"""Optional stress-controlled DEM/continuum coupling.

The module deliberately has no Kratos imports.  This keeps the established
rate-and-state path usable in Python environments where Kratos is absent and
also makes the coupling contract independently testable.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from fastslippy.pre_processing.model_parameters import DEMCouplingParameters


@dataclass(frozen=True)
class DEMRVEResult:
    """Homogenized response of one RVE over one macro time interval."""

    slip_increment: float
    average_velocity: float
    shear_traction: float
    normal_traction: float
    kinetic_energy: float = 0.0
    suggested_dt: float = np.inf


@dataclass(frozen=True)
class DEMWallState:
    """Instantaneous wall response relative to the accepted preload."""

    velocity: float
    shear_traction: float
    normal_traction: float
    loading_time: float
    loading_slip: float
    loading_steps: int


class DEMRVEBackend(Protocol):
    @property
    def wall_state(self) -> DEMWallState: ...

    def advance(
        self,
        *,
        driving_shear_stress: float,
        effective_normal_stress: float,
        radiation_damping: float,
        duration: float,
        initial_velocity: float,
    ) -> DEMRVEResult: ...

    def save_checkpoint(self, directory: Path) -> None: ...
    def close(self) -> None: ...


class BackendFactory(Protocol):
    def __call__(
        self,
        index: int,
        initial_normal_stress: float,
        initial_velocity: float,
        *,
        initial_driving_shear_stress: float,
        radiation_damping: float,
    ) -> DEMRVEBackend: ...


@dataclass(frozen=True)
class DEMCouplingResult:
    slip_increment: np.ndarray
    average_velocity: np.ndarray
    average_shear: np.ndarray
    average_normal: np.ndarray
    end_velocity: np.ndarray
    end_shear: np.ndarray
    end_normal: np.ndarray


class DEMMultiscaleCoupler:
    """One preloaded, continuously evolving RVE per constitutive fault node."""

    def __init__(
        self,
        node_count: int,
        settings: DEMCouplingParameters,
        backend_factory: BackendFactory,
        *,
        creep_start: int | None = None,
        creep_velocity: float = 0.0,
        extrapolate_surface: bool = False,
    ):
        self.settings = settings
        self._factory = backend_factory
        stop = node_count if creep_start is None else creep_start
        self._copy_surface = extrapolate_surface and stop > 1
        self.nodes = range(int(self._copy_surface), stop)
        self.backends: dict[int, DEMRVEBackend] = {}
        self.last_velocity = np.zeros(node_count)
        self.last_velocity[stop:] = creep_velocity
        self._suggested_dt = np.inf
        self._ready = False

    def _check_loads(self, driving, normal, eta):
        if driving.shape != self.last_velocity.shape or normal.shape != driving.shape:
            raise ValueError("DEM stress arrays must have one value per fault node.")
        if not (
            np.all(np.isfinite(driving))
            and np.all(np.isfinite(normal))
            and np.isfinite(eta)
            and eta >= 0
        ):
            raise ValueError("Invalid DEM loading.")
        if np.any(normal[self.nodes.start : self.nodes.stop] < 0):
            raise ValueError("DEM requires non-negative effective normal stress.")

    def initialize(self, driving, normal, eta):
        """Finish every preload before returning the time-zero fault state."""
        if self._ready:
            raise RuntimeError("DEM RVEs have already been initialized.")
        self._check_loads(driving, normal, eta)
        shear = driving - eta * self.last_velocity
        try:
            for i in self.nodes:
                rve = self._factory(
                    i,
                    float(normal[i]),
                    0.0,
                    initial_driving_shear_stress=float(driving[i]),
                    radiation_damping=eta,
                )
                self.backends[i] = rve
                state = rve.wall_state
                self.last_velocity[i] = state.velocity
                shear[i] = state.shear_traction
            if self._copy_surface:
                self.last_velocity[0] = self.last_velocity[1]
                shear[0] = driving[0] - eta * self.last_velocity[0]
        except BaseException:
            self.close()
            raise
        self._ready = True
        return self.last_velocity.copy(), shear

    def limit_dt(self, proposed_dt: float) -> float:
        dt = min(proposed_dt, self.settings.max_macro_dt, self._suggested_dt)
        speed = np.max(np.abs(self.last_velocity))
        if speed > 0:
            dt = min(dt, self.settings.max_slip_increment / speed)
        return dt

    def advance(
        self,
        *,
        driving_shear_stress,
        effective_normal_stress,
        radiation_damping: float,
        dt: float,
    ) -> DEMCouplingResult:
        if not self._ready:
            raise RuntimeError("Initialize DEM before advancing.")
        drive, normal, eta = (
            driving_shear_stress,
            effective_normal_stress,
            radiation_damping,
        )
        self._check_loads(drive, normal, eta)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("DEM macro time step must be finite and positive.")
        velocity = self.last_velocity.copy()
        slip = velocity * dt
        shear = drive - eta * velocity
        avg_normal = normal.copy()
        end_velocity, end_shear, end_normal = (
            velocity.copy(),
            shear.copy(),
            normal.copy(),
        )
        suggested_dt = np.inf
        for i, rve in self.backends.items():
            result = rve.advance(
                driving_shear_stress=float(drive[i]),
                effective_normal_stress=float(normal[i]),
                radiation_damping=eta,
                duration=dt,
                initial_velocity=self.last_velocity[i],
            )
            state = rve.wall_state
            if not np.isclose(
                result.slip_increment,
                result.average_velocity * dt,
                atol=1e-20,
                rtol=1e-10,
            ):
                raise RuntimeError(
                    f"Inconsistent slip increment and velocity at RVE {i}."
                )
            slip[i] = result.slip_increment
            velocity[i] = slip[i] / dt
            shear[i], avg_normal[i] = result.shear_traction, result.normal_traction
            end_velocity[i], end_shear[i], end_normal[i] = (
                state.velocity,
                state.shear_traction,
                state.normal_traction,
            )
            suggested_dt = min(suggested_dt, result.suggested_dt)
        if self._copy_surface:
            slip[0], velocity[0], end_velocity[0] = (
                slip[1],
                velocity[1],
                end_velocity[1],
            )
            shear[0] = drive[0] - eta * velocity[0]
            end_shear[0] = drive[0] - eta * end_velocity[0]
        self.last_velocity = end_velocity
        self._suggested_dt = suggested_dt
        return DEMCouplingResult(
            slip, velocity, shear, avg_normal, end_velocity, end_shear, end_normal
        )

    def save_checkpoint(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(
            directory / "coupler_state.npz",
            nodes=list(self.nodes),
            last_velocity=self.last_velocity,
            suggested_dt=self._suggested_dt,
        )
        for i, backend in self.backends.items():
            backend.save_checkpoint(directory / f"rve_{i:05d}")

    def write_history(self, directory, time, fault_slip, result=None):
        """Record accepted states; averages refer to the last Euler interval."""
        fields = (
            "node",
            "fdm_time_s",
            "dem_time_s",
            "dem_steps",
            "fdm_slip_m",
            "dem_slip_m",
            "end_velocity_m_s",
            "end_shear_Pa",
            "end_normal_Pa",
            "interval_slip_m",
            "average_velocity_m_s",
            "average_shear_Pa",
            "average_normal_Pa",
        )
        with (directory / "dem_history.csv").open(
            "w" if result is None else "a", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            if result is None:
                writer.writerow(fields)
            for i, rve in self.backends.items():
                state = rve.wall_state
                interval = (
                    ("", "", "", "")
                    if result is None
                    else (
                        result.slip_increment[i],
                        result.average_velocity[i],
                        result.average_shear[i],
                        result.average_normal[i],
                    )
                )
                writer.writerow(
                    (
                        i,
                        time,
                        state.loading_time,
                        state.loading_steps,
                        fault_slip[i],
                        state.loading_slip,
                        state.velocity,
                        state.shear_traction,
                        state.normal_traction,
                        *interval,
                    )
                )

    def close(self) -> None:
        self._ready = False
        for backend in self.backends.values():
            backend.close()
        self.backends.clear()
