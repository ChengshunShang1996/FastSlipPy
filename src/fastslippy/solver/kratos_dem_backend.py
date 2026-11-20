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
from fastslippy.solver.dem_coupling import DEMRVEResult


class NormalConsolidationError(RuntimeError):
    """The RVE failed to establish the requested initial normal stress."""


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
            desired_normal_velocity = -self.settings.normal_servo_velocity * np.clip(
                normal_error, -1.0, 1.0
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
        parameters["FinalTime"].SetDouble(1.0e300)
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
        if initial_normal_stress is not None:
            try:
                with _working_directory(self.work_directory):
                    self._consolidate_normal(initial_normal_stress)
            except BaseException:
                self.close()
                raise
        self.stage.wall_velocity[0] = initial_velocity

    def _consolidate_normal(self, target: float) -> None:
        stage, km, cfg = self.stage, self.KM, self.settings
        top = list(stage.top_wall.Nodes)
        bottom = list(stage.bottom_wall.Nodes)
        wall_ids = {n.Id for n in top + bottom}
        grains = [n for n in stage.spheres_model_part.Nodes if n.Id not in wall_ids]
        if not top or not bottom or not grains:
            raise ValueError("Normal consolidation requires two walls and free grains.")

        stage.lock_tangential_motion = True
        stage.effective_normal_stress = target
        stress_tol = (
            cfg.consolidation_stress_atol + cfg.consolidation_stress_rtol * target
        )
        stable_steps = 0
        path = self.work_directory / "normal_consolidation.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            for step in range(1, cfg.consolidation_max_steps + 1):
                self._single_step()
                bottom_force = np.sum(
                    [
                        np.asarray(n.GetSolutionStepValue(km.TOTAL_FORCES))
                        for n in bottom
                    ],
                    axis=0,
                )
                top_normal = stage.top_wall_force[1] / cfg.layer_length
                bottom_normal = -bottom_force[1] / cfg.layer_length
                imbalance = (
                    np.linalg.norm((stage.top_wall_force + bottom_force)[:2])
                    / cfg.layer_length
                )
                wall_speed = np.max(
                    [abs(n.GetSolutionStepValue(km.VELOCITY_Y)) for n in top + bottom]
                )
                grain_speed = np.max(
                    [
                        np.linalg.norm(n.GetSolutionStepValue(km.VELOCITY))
                        for n in grains
                    ]
                )
                rim_speed = np.max(
                    [
                        n.GetSolutionStepValue(km.RADIUS)
                        * abs(n.GetSolutionStepValue(km.ANGULAR_VELOCITY_Z))
                        for n in grains
                    ]
                )
                finite = np.isfinite(
                    [
                        top_normal,
                        bottom_normal,
                        imbalance,
                        wall_speed,
                        grain_speed,
                        rim_speed,
                    ]
                ).all()
                in_band = (
                    finite
                    and max(
                        abs(top_normal - target), abs(bottom_normal - target), imbalance
                    )
                    <= stress_tol
                    and max(wall_speed, grain_speed, rim_speed)
                    <= cfg.consolidation_speed_tol
                    and (target == 0 or min(top_normal, bottom_normal) > 0)
                )
                stable_steps = stable_steps + 1 if in_band else 0
                if stable_steps >= cfg.consolidation_window_steps:
                    status = "converged"
                elif not finite:
                    status = "nonfinite"
                elif step == cfg.consolidation_max_steps:
                    status = "max_steps"
                else:
                    status = "monitoring"
                row = dict(
                    step=step,
                    time_s=stage.time,
                    target_Pa=target,
                    top_Pa=top_normal,
                    bottom_Pa=bottom_normal,
                    imbalance_Pa=imbalance,
                    wall_speed_m_s=wall_speed,
                    top_vy_m_s=top[0].GetSolutionStepValue(km.VELOCITY_Y),
                    grain_speed_m_s=grain_speed,
                    rim_speed_m_s=rim_speed,
                    stable_steps=stable_steps,
                    status=status,
                )
                if step == 1:
                    writer = csv.DictWriter(stream, fieldnames=row)
                    writer.writeheader()
                writer.writerow(row)
                if status == "converged":
                    stage.lock_tangential_motion = False
                    return
                if status != "monitoring":
                    raise NormalConsolidationError(
                        f"RVE {self.index} normal consolidation stopped ({status}) at "
                        f"step {step}, target={target:g} Pa, top={top_normal:g} Pa, "
                        f"bottom={bottom_normal:g} Pa, imbalance={imbalance:g} Pa, "
                        f"wall={wall_speed:g}, grain={grain_speed:g}, "
                        f"rim={rim_speed:g} m/s, stable steps={stable_steps}. "
                        f"See {path}"
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
            "dem_shearDEM_FEM_boundary.mdpa": (
                f"{problem_name}DEM_FEM_boundary.mdpa"
            ),
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
        solver.Predict()
        solver.SolveSolutionStep()
        self.stage.FinalizeSolutionStep()
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
                    self._single_step()
                    traction_integral += stage.measured_shear_traction * step_dt
                    normal_integral += stage.measured_normal_traction * step_dt
            finally:
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
        self, index: int, initial_normal_stress: float, initial_velocity: float
    ) -> KratosDEMRVE:
        return KratosDEMRVE(
            index,
            initial_normal_stress,
            initial_velocity,
            self.settings,
            self.output_root,
        )

