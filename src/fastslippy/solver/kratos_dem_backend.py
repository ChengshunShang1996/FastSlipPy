"""Kratos DEM backend for the optional stress-controlled RVE coupling.

Kratos is imported lazily when the backend is constructed.  Merely importing
FastSlipPy therefore does not require KratosMultiphysics.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from fastslippy.pre_processing.model_parameters import DEMCouplingParameters
from fastslippy.solver.dem_coupling import DEMRVEResult, DEMWallState


class NormalConsolidationError(RuntimeError):
    """The RVE failed to establish the requested initial normal stress."""


class ShearPreloadError(RuntimeError):
    """The RVE failed to establish a near-static initial shear state."""


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _stage_class(KM, DEMAnalysisStage):
    class StressControlledShearStage(DEMAnalysisStage):
        """Rough-wall RVE with normal servo and a dynamic shear wall."""

        def __init__(self, model, parameters, settings: DEMCouplingParameters):
            self.settings = settings
            self.driving_shear_stress = 0.0
            self.effective_normal_stress = 1.0
            self.radiation_damping = 0.0
            self.wall_velocity = np.zeros(2, dtype=float)
            self.prescribed_wall_velocity = None
            self.lock_tangential_motion = False
            self.shear_displacement = 0.0
            self.measured_shear_traction = 0.0
            self.measured_normal_traction = 0.0
            self.measured_bottom_shear_traction = 0.0
            self.measured_bottom_normal_traction = 0.0
            self.normal_servo_command = 0.0
            self.normal_servo_saturated = False
            self.kinetic_energy = 0.0
            super().__init__(model, parameters)

        def Initialize(self):
            super().Initialize()
            self.top_wall = self.spheres_model_part.GetSubModelPart("TopWall")
            self.bottom_wall = self.spheres_model_part.GetSubModelPart("BottomWall")

        def _update_wall_velocity(self):
            if self.lock_tangential_motion:
                self.wall_velocity[0] = 0.0
            else:
                dt = float(self._GetSolver().dt)
                target_resistance = (
                    self.driving_shear_stress
                    - self.radiation_damping * self.wall_velocity[0]
                )
                acceleration = (
                    target_resistance - self.measured_shear_traction
                ) / self.settings.wall_mass_per_area
                self.wall_velocity[0] += acceleration * dt
                maximum = self.settings.maximum_wall_speed
                self.wall_velocity[0] = float(
                    np.clip(self.wall_velocity[0], -maximum, maximum)
                )

            normal_error = (
                self.effective_normal_stress - self.measured_normal_traction
            ) / max(self.effective_normal_stress, 1.0e-30)
            unrestricted_normal_velocity = (
                -self.settings.normal_servo_velocity * normal_error
            )
            desired_normal_velocity = float(
                np.clip(
                    unrestricted_normal_velocity,
                    -self.settings.normal_servo_velocity,
                    self.settings.normal_servo_velocity,
                )
            )
            self.normal_servo_command = desired_normal_velocity
            self.normal_servo_saturated = bool(
                abs(unrestricted_normal_velocity)
                > self.settings.normal_servo_velocity
            )
            relaxation = self.settings.servo_relaxation
            self.wall_velocity[1] = (
                (1.0 - relaxation) * self.wall_velocity[1]
                + relaxation * desired_normal_velocity
            )

        def InitializeSolutionStep(self):
            if self.prescribed_wall_velocity is None:
                self._update_wall_velocity()
            else:
                self.wall_velocity[:] = self.prescribed_wall_velocity
            velocity = [
                float(self.wall_velocity[0]),
                float(self.wall_velocity[1]),
                0.0,
            ]
            for node in self.top_wall.Nodes:
                node.SetSolutionStepValue(KM.VELOCITY, velocity)
            for node in self.bottom_wall.Nodes:
                node.SetSolutionStepValue(KM.VELOCITY, [0.0, 0.0, 0.0])
            super().InitializeSolutionStep()

        def FinalizeSolutionStep(self):
            super().FinalizeSolutionStep()
            total_force = np.zeros(3, dtype=float)
            for node in self.top_wall.Nodes:
                total_force += np.asarray(
                    node.GetSolutionStepValue(KM.TOTAL_FORCES), dtype=float
                )
            length = self.settings.layer_length
            self.top_wall_force = total_force
            self.measured_shear_traction = float(-total_force[0] / length)
            self.measured_normal_traction = max(float(total_force[1] / length), 0.0)
            bottom_force = np.zeros(3, dtype=float)
            for node in self.bottom_wall.Nodes:
                bottom_force += np.asarray(
                    node.GetSolutionStepValue(KM.TOTAL_FORCES), dtype=float
                )
            self.bottom_wall_force = bottom_force
            self.measured_bottom_shear_traction = float(bottom_force[0] / length)
            self.measured_bottom_normal_traction = max(
                float(-bottom_force[1] / length), 0.0
            )
            dt = float(self.spheres_model_part.ProcessInfo[KM.DELTA_TIME])
            self.shear_displacement += self.wall_velocity[0] * dt
            energy = 0.0
            for node in self.spheres_model_part.Nodes:
                velocity = np.asarray(node.GetSolutionStepValue(KM.VELOCITY))
                # Nodal mass is not uniformly exposed by every DEM element;
                # this velocity norm is retained as a robust activity proxy.
                energy += float(np.dot(velocity, velocity))
            self.kinetic_energy = 0.5 * energy

    return StressControlledShearStage


class KratosDEMRVE:
    """Persistent Kratos RVE associated with one FastSlipPy fault node."""

    def __init__(
        self,
        index: int,
        initial_normal_stress: float | None,
        initial_velocity: float,
        settings: DEMCouplingParameters,
        output_root: Path,
        *,
        initial_driving_shear_stress: float | None = None,
        radiation_damping: float = 0.0,
    ) -> None:
        try:
            import KratosMultiphysics as KM
            import KratosMultiphysics.DEMApplication  # noqa: F401
            from KratosMultiphysics.DEMApplication.DEM_analysis_stage import (
                DEMAnalysisStage,
            )
        except ImportError as exc:
            raise RuntimeError(
                "dem_multiscale requires a Kratos installation containing "
                "DEMApplication."
            ) from exc

        settings.validate()
        if initial_driving_shear_stress is not None:
            if (
                not math.isfinite(initial_velocity)
                or abs(initial_velocity) > settings.consolidation_speed_tol
            ):
                raise ValueError(
                    "Static shear preload requires a near-zero initial velocity."
                )
            if initial_normal_stress is None or not math.isfinite(
                initial_driving_shear_stress
            ):
                raise ValueError(
                    "Shear preload requires a normal stress and a finite shear target."
                )
            if not math.isfinite(radiation_damping) or radiation_damping < 0:
                raise ValueError("Radiation damping must be finite and non-negative.")
        if initial_normal_stress is not None and (
            not math.isfinite(initial_normal_stress) or initial_normal_stress < 0
        ):
            raise ValueError(
                "Initial DEM normal stress must be finite and compressive or zero."
            )
        if settings.consolidation_time:
            warnings.warn(
                "consolidation_time is deprecated and ignored. "
                "Normal consolidation ends when its equilibrium criteria are met.",
                FutureWarning,
                stacklevel=2,
            )
        self.KM = KM
        self.settings = settings
        self.index = index
        self.work_directory = output_root.resolve() / f"rve_{index:05d}"
        self.work_directory.mkdir(parents=True, exist_ok=True)
        self._prepare_case_files()

        parameters = KM.Parameters(
            (self.work_directory / "ProjectParametersDEM.json").read_text(
                encoding="utf-8"
            )
        )
        problem_name = f"rve_{index:05d}"
        parameters["problem_name"].SetString(problem_name)
        parameters["FinalTime"].SetDouble(1000)
        parameters["do_print_results_option"].SetBool(settings.output_enabled)
        if not settings.output_enabled:
            parameters["post_vtk_option"].SetBool(False)
            parameters["post_gid_option"].SetBool(False)

        stage_type = _stage_class(KM, DEMAnalysisStage)
        self.model = KM.Model()
        with _working_directory(self.work_directory):
            self.stage = stage_type(self.model, parameters, settings)
            self.stage.Initialize()
        self._closed = False
        self._capture_contact_diagnostics = False
        self._initialize_bulk_diagnostics()
        if initial_normal_stress is not None:
            try:
                with _working_directory(self.work_directory):
                    self._equilibrate(initial_normal_stress)
                    if initial_driving_shear_stress is not None:
                        self._equilibrate(
                            initial_normal_stress,
                            initial_driving_shear_stress,
                            radiation_damping,
                        )
            except BaseException:
                self.close()
                raise
        if initial_driving_shear_stress is None:
            self.stage.wall_velocity[0] = initial_velocity
        # Only the reference changes at handoff. Keep the Kratos time, geometry,
        # velocity and all contact history on this very same stage.
        self.loading_time_origin = self.stage.time
        self.loading_slip_origin = self.stage.shear_displacement
        self.loading_step_origin = self.stage.spheres_model_part.ProcessInfo[
            KM.TIME_STEPS
        ]
        self.loading_internal_slip_origin = self._raw_internal_slip()

    @property
    def loading_time(self) -> float:
        return self.stage.time - self.loading_time_origin

    @property
    def loading_slip(self) -> float:
        return self.stage.shear_displacement - self.loading_slip_origin

    @property
    def loading_internal_slip(self) -> float:
        return self._raw_internal_slip() - self.loading_internal_slip_origin

    def _initialize_bulk_diagnostics(self) -> None:
        stage = self.stage
        wall_ids = {
            node.Id
            for node in list(stage.top_wall.Nodes) + list(stage.bottom_wall.Nodes)
        }
        self._free_grains = [
            node for node in stage.spheres_model_part.Nodes if node.Id not in wall_ids
        ]
        ordered = sorted(self._free_grains, key=lambda node: (node.Y0, node.Id))
        split = max(1, len(ordered) // 2)
        self._lower_grains = ordered[:split]
        self._upper_grains = ordered[split:]
        if not self._upper_grains:
            self._upper_grains = self._lower_grains

        source = Path(self.settings.case_directory).expanduser().resolve()
        material_data = json.loads((source / "MaterialsDEM.json").read_text())
        friction_values = {
            float(relation["Variables"]["STATIC_FRICTION"])
            for relation in material_data.get("material_relations", [])
            if "STATIC_FRICTION" in relation.get("Variables", {})
        }
        self._contact_friction_limit = (
            friction_values.pop() if len(friction_values) == 1 else np.nan
        )

    @staticmethod
    def _mean_component(nodes, getter) -> float:
        return float(np.mean([getter(node) for node in nodes]))

    def _raw_internal_slip(self) -> float:
        km = self.KM
        upper = self._mean_component(
            self._upper_grains,
            lambda node: node.GetSolutionStepValue(km.DISPLACEMENT_X),
        )
        lower = self._mean_component(
            self._lower_grains,
            lambda node: node.GetSolutionStepValue(km.DISPLACEMENT_X),
        )
        return upper - lower

    def _internal_velocity(self) -> float:
        km = self.KM
        upper = self._mean_component(
            self._upper_grains, lambda node: node.GetSolutionStepValue(km.VELOCITY_X)
        )
        lower = self._mean_component(
            self._lower_grains, lambda node: node.GetSolutionStepValue(km.VELOCITY_X)
        )
        return upper - lower

    def _bulk_stress(self) -> tuple[float, float]:
        variable = self.KM.KratosGlobals.GetVariable("DEM_STRESS_TENSOR")
        weights, shear, normal = [], [], []
        for node in self._free_grains:
            matrix = np.asarray(node.GetSolutionStepValue(variable), dtype=float)
            radius = float(node.GetSolutionStepValue(self.KM.RADIUS))
            weights.append(radius * radius)
            shear.append(matrix[0, 1])
            normal.append(-matrix[1, 1])
        return (
            float(np.average(shear, weights=weights)),
            float(np.average(normal, weights=weights)),
        )

    def _contact_friction_diagnostics(self) -> tuple[float, float]:
        variable = self.KM.KratosGlobals.GetVariable("LOCAL_CONTACT_FORCE")
        ratios = []
        forces = []
        for element in self.stage.contact_model_part.Elements:
            force = np.asarray(element.GetValue(variable), dtype=float)
            normal = abs(float(force[2]))
            tangent = float(np.hypot(force[0], force[1]))
            forces.append((normal, tangent))
        if not forces:
            return np.nan, np.nan
        maximum_normal = max(normal for normal, _ in forces)
        threshold = max(np.finfo(float).eps, 1.0e-12 * maximum_normal)
        ratios = [tangent / normal for normal, tangent in forces if normal > threshold]
        if not ratios:
            return np.nan, np.nan
        limit = self._contact_friction_limit
        sliding_fraction = (
            float(np.mean(np.asarray(ratios) >= limit * (1.0 - 1.0e-6)))
            if np.isfinite(limit) and limit > 0.0
            else np.nan
        )
        return float(max(ratios)), sliding_fraction

    @property
    def wall_state(self) -> DEMWallState:
        solver = self.stage._GetSolver()
        solver.PrepareElementsForPrinting()
        solver.PrepareContactElementsForPrinting()
        bulk_shear, bulk_normal = self._bulk_stress()
        maximum_ratio, sliding_fraction = self._contact_friction_diagnostics()
        return DEMWallState(
            float(self.stage.wall_velocity[0]),
            self.stage.measured_shear_traction,
            self.stage.measured_normal_traction,
            self.loading_time,
            self.loading_slip,
            self.stage.spheres_model_part.ProcessInfo[self.KM.TIME_STEPS]
            - self.loading_step_origin,
            internal_slip=self.loading_internal_slip,
            internal_velocity=self._internal_velocity(),
            top_shear_traction=self.stage.measured_shear_traction,
            bottom_shear_traction=self.stage.measured_bottom_shear_traction,
            top_normal_traction=self.stage.measured_normal_traction,
            bottom_normal_traction=self.stage.measured_bottom_normal_traction,
            bulk_shear_stress=bulk_shear,
            bulk_normal_stress=bulk_normal,
            maximum_contact_friction_ratio=maximum_ratio,
            contact_friction_limit=self._contact_friction_limit,
            sliding_contact_fraction=sliding_fraction,
            target_normal_traction=self.stage.effective_normal_stress,
            target_driving_shear=self.stage.driving_shear_stress,
            normal_wall_velocity=float(self.stage.wall_velocity[1]),
            normal_servo_saturated=self.stage.normal_servo_saturated,
        )

    def _wall_and_grain_measurements(self, top, bottom, grains):
        km = self.KM
        length = self.settings.layer_length
        top_force = self.stage.top_wall_force
        bottom_force = np.sum(
            [np.asarray(n.GetSolutionStepValue(km.TOTAL_FORCES)) for n in bottom],
            axis=0,
        )
        return dict(
            top_Pa=top_force[1] / length,
            bottom_Pa=-bottom_force[1] / length,
            imbalance_Pa=np.linalg.norm((top_force + bottom_force)[:2]) / length,
            top_shear_Pa=-top_force[0] / length,
            bottom_shear_Pa=bottom_force[0] / length,
            top_vx_m_s=top[0].GetSolutionStepValue(km.VELOCITY_X),
            top_vy_m_s=top[0].GetSolutionStepValue(km.VELOCITY_Y),
            wall_speed_m_s=np.max(
                [abs(n.GetSolutionStepValue(km.VELOCITY_Y)) for n in top + bottom]
            ),
            grain_speed_m_s=np.max(
                [np.linalg.norm(n.GetSolutionStepValue(km.VELOCITY)) for n in grains]
            ),
            rim_speed_m_s=np.max(
                [
                    n.GetSolutionStepValue(km.RADIUS)
                    * abs(n.GetSolutionStepValue(km.ANGULAR_VELOCITY_Z))
                    for n in grains
                ]
            ),
        )

    def _equilibrate(
        self, target: float, shear: float | None = None, radiation_damping: float = 0.0
    ) -> None:
        stage, cfg = self.stage, self.settings
        top = list(stage.top_wall.Nodes)
        bottom = list(stage.bottom_wall.Nodes)
        wall_ids = {n.Id for n in top + bottom}
        grains = [n for n in stage.spheres_model_part.Nodes if n.Id not in wall_ids]
        if not top or not bottom or not grains:
            raise ValueError("Normal consolidation requires two walls and free grains.")

        stage.lock_tangential_motion = shear is None
        stage.effective_normal_stress = target
        stage.radiation_damping = radiation_damping
        start_shear = stage.measured_shear_traction
        stress_tol = (
            cfg.consolidation_stress_atol + cfg.consolidation_stress_rtol * target
        )
        stable_steps = 0
        sliding_steps = 0
        last_vx = 0.0
        phase = "normal_consolidation" if shear is None else "shear_preload"
        path = self.work_directory / f"{phase}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            for step in range(1, cfg.consolidation_max_steps + 1):
                if shear is not None:
                    fraction = min(step / cfg.shear_preload_ramp_steps, 1.0)
                    weight = fraction * fraction * (3.0 - 2.0 * fraction)
                    stage.driving_shear_stress = (
                        shear
                        if fraction == 1.0
                        else start_shear + weight * (shear - start_shear)
                    )
                self._single_step()
                measured = self._wall_and_grain_measurements(top, bottom, grains)
                top_normal = measured["top_Pa"]
                bottom_normal = measured["bottom_Pa"]
                vx = measured["top_vx_m_s"]
                in_band = (
                    max(
                        abs(top_normal - target),
                        abs(bottom_normal - target),
                        measured["imbalance_Pa"],
                    )
                    <= stress_tol
                    # and max(
                    #     measured["wall_speed_m_s"],
                    #     measured["grain_speed_m_s"],
                    #     measured["rim_speed_m_s"],
                    # )
                    # <= cfg.consolidation_speed_tol
                    and (target == 0 or min(top_normal, bottom_normal) > 0)
                )
                if shear is not None:
                    at_target = step >= cfg.shear_preload_ramp_steps
                    contact_target = stage.driving_shear_stress - radiation_damping * vx
                    shear_tol = (
                        cfg.consolidation_stress_atol
                        + cfg.consolidation_stress_rtol * abs(shear)
                    )
                    in_band = (
                        in_band
                        and at_target
                        and abs(vx) <= cfg.consolidation_speed_tol
                        and abs(radiation_damping * vx) <= shear_tol
                        and abs(measured["top_shear_Pa"] - contact_target) <= shear_tol
                        and abs(measured["bottom_shear_Pa"] - contact_target)
                        <= shear_tol
                    )
                    if at_target and abs(vx) > cfg.consolidation_speed_tol:
                        sliding_steps = sliding_steps + 1 if vx * last_vx > 0 else 1
                    else:
                        sliding_steps = 0
                    last_vx = vx
                    measured.update(
                        target_drive_Pa=shear,
                        applied_drive_Pa=stage.driving_shear_stress,
                        damping_Pa_s_m=radiation_damping,
                        contact_target_Pa=contact_target,
                        preload_slip_m=stage.shear_displacement,
                    )
                finite = all(math.isfinite(value) for value in measured.values())
                in_band = in_band and finite
                stable_steps = stable_steps + 1 if in_band else 0
                if stable_steps >= cfg.consolidation_window_steps:
                    status = "converged"
                elif not finite:
                    status = "nonfinite"
                elif step == cfg.consolidation_max_steps:
                    status = (
                        "sustained_sliding"
                        if sliding_steps >= cfg.consolidation_window_steps
                        else "max_steps"
                    )
                else:
                    status = "monitoring"
                row = dict(
                    step=step,
                    time_s=stage.time,
                    target_Pa=target,
                    stable_steps=stable_steps,
                    status=status,
                    **measured,
                )
                if step == 1:
                    writer = csv.DictWriter(stream, fieldnames=row)
                    writer.writeheader()
                writer.writerow(row)
                if status == "converged":
                    stage.lock_tangential_motion = False
                    return
                if status != "monitoring":
                    if shear is not None:
                        raise ShearPreloadError(
                            f"RVE {self.index} shear preload stopped ({status}) "
                            f"at step {step}, drive={shear:g} Pa, "
                            f"contact={stage.measured_shear_traction:g} Pa, "
                            f"normal={top_normal:g} Pa, vx={vx:g} m/s. See {path}"
                        )
                    raise NormalConsolidationError(
                        f"RVE {self.index} normal consolidation stopped ({status}) at "
                        f"step {step}, target={target:g} Pa, top={top_normal:g} Pa, "
                        f"bottom={bottom_normal:g} Pa, "
                        f"imbalance={measured['imbalance_Pa']:g} Pa, "
                        f"wall={measured['wall_speed_m_s']:g}, "
                        f"grain={measured['grain_speed_m_s']:g}, "
                        f"rim={measured['rim_speed_m_s']:g} m/s, "
                        f"stable steps={stable_steps}. See {path}"
                    )

    def _prepare_case_files(self) -> None:
        source = Path(self.settings.case_directory).expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"DEM case directory does not exist: {source}")
        required = (
            "dem_shearDEM.mdpa",
            "dem_shearDEM_FEM_boundary.mdpa",
            "MaterialsDEM.json",
            "ProjectParametersDEM.json",
        )
        problem_name = f"rve_{self.index:05d}"
        destinations = {
            "dem_shearDEM.mdpa": f"{problem_name}DEM.mdpa",
            "dem_shearDEM_FEM_boundary.mdpa": (f"{problem_name}DEM_FEM_boundary.mdpa"),
        }
        for name in required:
            path = source / name
            if not path.is_file():
                raise FileNotFoundError(f"Required DEM case file is missing: {path}")
            shutil.copy2(path, self.work_directory / destinations.get(name, name))

    def _single_step(self) -> None:
        solver = self.stage._GetSolver()
        self.stage.time = self.stage._AdvanceTime()
        self.stage.InitializeSolutionStep()
        update_variable = self.KM.KratosGlobals.GetVariable(
            "IS_TIME_TO_UPDATE_CONTACT_ELEMENT"
        )
        if self._capture_contact_diagnostics:
            self.stage.spheres_model_part.ProcessInfo[update_variable] = True
        solver.Predict()
        solver.SolveSolutionStep()
        self.stage.FinalizeSolutionStep()
        if self._capture_contact_diagnostics:
            self.stage.spheres_model_part.ProcessInfo[update_variable] = False
        if self.settings.output_enabled:
            self.stage.OutputSolutionStep()

    def _advance_interval(
        self,
        *,
        duration: float,
        driving_shear_stress: float,
        effective_normal_stress: float,
        radiation_damping: float,
    ) -> tuple[float, float, float]:
        stage = self.stage
        stage.driving_shear_stress = driving_shear_stress
        stage.effective_normal_stress = effective_normal_stress
        stage.radiation_damping = radiation_damping
        solver = stage._GetSolver()
        nominal_dt = float(solver.dt)
        if not math.isfinite(nominal_dt) or nominal_dt <= 0.0:
            raise ValueError("DEM solver time step must be finite and positive.")
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("DEM interval must be finite and positive.")
        # Repeatedly summing dt can produce a spurious near-zero final solve.
        full_steps = int(duration / nominal_dt)
        remainder = duration - full_steps * nominal_dt
        step_count = max(1, full_steps + int(remainder > math.ulp(duration)))
        traction_integral = 0.0
        normal_integral = 0.0
        with _working_directory(self.work_directory):
            try:
                for i in range(step_count):
                    step_dt = nominal_dt
                    if i + 1 == step_count:
                        step_dt = duration - i * nominal_dt
                    solver.dt = step_dt
                    self._capture_contact_diagnostics = i + 1 == step_count
                    self._single_step()
                    traction_integral += stage.measured_shear_traction * step_dt
                    normal_integral += stage.measured_normal_traction * step_dt
            finally:
                self._capture_contact_diagnostics = False
                solver.dt = nominal_dt
        return (
            traction_integral / duration,
            normal_integral / duration,
            nominal_dt,
        )

    def advance(
        self,
        *,
        driving_shear_stress: float,
        effective_normal_stress: float,
        radiation_damping: float,
        duration: float,
        initial_velocity: float,
    ) -> DEMRVEResult:
        if self._closed:
            raise RuntimeError("Cannot advance a closed Kratos DEM RVE.")
        start = self.stage.shear_displacement
        average_shear, average_normal, _ = self._advance_interval(
            duration=duration,
            driving_shear_stress=driving_shear_stress,
            effective_normal_stress=effective_normal_stress,
            radiation_damping=radiation_damping,
        )
        slip = self.stage.shear_displacement - start
        return DEMRVEResult(
            slip_increment=slip,
            average_velocity=slip / duration,
            shear_traction=average_shear,
            normal_traction=average_normal,
            kinetic_energy=self.stage.kinetic_energy,
            # The DEM stability step is handled by internal subcycling; macro
            # accuracy is governed by the coupler's stress/slip limits.
            suggested_dt=np.inf,
        )

    def save_checkpoint(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        serializer = self.KM.FileSerializer(
            str(directory / "kratos_model"),
            self.KM.SerializerTraceType.SERIALIZER_NO_TRACE,
        )
        serializer.Save("Model", self.model)
        metadata = {
            "time": float(self.stage.time),
            "wall_velocity": self.stage.wall_velocity.tolist(),
            "shear_displacement": float(self.stage.shear_displacement),
            "loading_slip_origin": float(self.loading_slip_origin),
            "loading_time_origin": float(self.loading_time_origin),
            "loading_step_origin": int(self.loading_step_origin),
            "measured_shear_traction": float(self.stage.measured_shear_traction),
            "measured_normal_traction": float(self.stage.measured_normal_traction),
        }
        (directory / "state.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    def close(self) -> None:
        if not self._closed:
            with _working_directory(self.work_directory):
                self.stage.Finalize()
            self._closed = True


class KratosDEMBackendFactory:
    def __init__(self, settings: DEMCouplingParameters, output_root: Path) -> None:
        self.settings = settings
        self.output_root = output_root

    def __call__(
        self,
        index: int,
        initial_normal_stress: float,
        initial_velocity: float,
        *,
        initial_driving_shear_stress: float | None = None,
        radiation_damping: float = 0.0,
    ) -> KratosDEMRVE:
        return KratosDEMRVE(
            index,
            initial_normal_stress,
            initial_velocity,
            self.settings,
            self.output_root,
            initial_driving_shear_stress=initial_driving_shear_stress,
            radiation_damping=radiation_damping,
        )

