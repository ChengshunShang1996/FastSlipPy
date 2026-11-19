"""Optional stress-controlled DEM/continuum coupling.

The module deliberately has no Kratos imports.  This keeps the established
rate-and-state path usable in Python environments where Kratos is absent and
also makes the coupling contract independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Protocol

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


class DEMRVEBackend(Protocol):
    """Stateful scalar RVE used at one fault node."""

    def advance(
        self,
        *,
        driving_shear_stress: float,
        effective_normal_stress: float,
        radiation_damping: float,
        duration: float,
        initial_velocity: float,
    ) -> DEMRVEResult:
        ...

    def save_checkpoint(self, directory: Path) -> None:
        ...

    def close(self) -> None:
        ...


BackendFactory = Callable[[int, float, float], DEMRVEBackend]


@dataclass(frozen=True)
class DEMCouplingResult:
    velocity: np.ndarray
    traction: np.ndarray
    slip_increment: np.ndarray
    active: np.ndarray
    traction_residual: np.ndarray
    kinetic_energy: np.ndarray


class DEMMultiscaleCoupler:
    """Switch fault nodes between an RSF predictor and dynamic DEM RVEs."""

    def __init__(
        self,
        node_count: int,
        settings: DEMCouplingParameters,
        backend_factory: BackendFactory,
    ) -> None:
        self.settings = settings
        self._factory = backend_factory
        self._backends: list[Optional[DEMRVEBackend]] = [None] * node_count
        self.active = np.zeros(node_count, dtype=bool)
        self._below_threshold_steps = np.zeros(node_count, dtype=np.int64)
        self.last_velocity = np.zeros(node_count, dtype=float)
        self.last_kinetic_energy = np.zeros(node_count, dtype=float)
        self.last_suggested_dt = np.full(node_count, np.inf, dtype=float)

    def fast_step_expected(self, predictor_velocity: np.ndarray) -> bool:
        predictor = np.asarray(predictor_velocity, dtype=float)
        return bool(
            np.any(self.active)
            or np.any(np.abs(predictor) >= self.settings.activation_velocity)
        )

    def limit_dt(
        self,
        proposed_dt: float,
        predictor_velocity: Optional[np.ndarray] = None,
    ) -> float:
        """Apply DEM-specific macro limits without changing inactive RSF runs."""

        dt = min(float(proposed_dt), float(self.settings.max_macro_dt))
        speeds = np.abs(self.last_velocity[self.active])
        if predictor_velocity is not None:
            predictor = np.asarray(predictor_velocity, dtype=float)
            candidate = self.active | (
                np.abs(predictor) >= self.settings.activation_velocity
            )
            speeds = np.concatenate((speeds, np.abs(predictor[candidate])))
        if speeds.size and np.isfinite(self.settings.max_slip_increment):
            speed = np.max(speeds)
            if speed > 0.0:
                dt = min(dt, self.settings.max_slip_increment / speed)
            if np.any(self.active):
                dt = min(dt, float(np.min(self.last_suggested_dt[self.active])))
        return dt

    def advance(
        self,
        *,
        predictor_velocity: np.ndarray,
        driving_shear_stress: np.ndarray,
        effective_normal_stress: np.ndarray,
        radiation_damping: float,
        dt: float,
    ) -> DEMCouplingResult:
        predictor = np.asarray(predictor_velocity, dtype=float)
        driving = np.asarray(driving_shear_stress, dtype=float)
        normal = np.asarray(effective_normal_stress, dtype=float)
        if predictor.shape != self.active.shape:
            raise ValueError("DEM predictor velocity has the wrong shape.")
        if driving.shape != self.active.shape or normal.shape != self.active.shape:
            raise ValueError("DEM stress arrays have the wrong shape.")
        if dt <= 0.0 or not np.isfinite(dt):
            raise ValueError("DEM macro time step must be finite and positive.")

        newly_active = (~self.active) & (
            np.abs(predictor) >= self.settings.activation_velocity
        )
        self.active[newly_active] = True

        velocity = predictor.copy()
        traction = driving - radiation_damping * predictor
        slip = predictor * dt
        residual = np.zeros_like(predictor)
        kinetic = np.zeros_like(predictor)

        for index in np.flatnonzero(self.active):
            backend = self._backends[index]
            if backend is None:
                backend = self._factory(
                    int(index), float(normal[index]), float(predictor[index])
                )
                self._backends[index] = backend
            result = backend.advance(
                driving_shear_stress=float(driving[index]),
                effective_normal_stress=float(normal[index]),
                radiation_damping=float(radiation_damping),
                duration=float(dt),
                initial_velocity=float(self.last_velocity[index] or predictor[index]),
            )
            if not np.isclose(result.slip_increment, result.average_velocity * dt):
                raise RuntimeError(
                    f"DEM RVE {index} returned inconsistent slip and velocity."
                )
            velocity[index] = result.average_velocity
            traction[index] = result.shear_traction
            slip[index] = result.slip_increment
            kinetic[index] = result.kinetic_energy
            residual[index] = (
                result.shear_traction
                + radiation_damping * result.average_velocity
                - driving[index]
            )
            self.last_velocity[index] = result.average_velocity
            self.last_kinetic_energy[index] = result.kinetic_energy
            self.last_suggested_dt[index] = result.suggested_dt

            if abs(result.average_velocity) <= self.settings.deactivation_velocity:
                self._below_threshold_steps[index] += 1
            else:
                self._below_threshold_steps[index] = 0
            if self._below_threshold_steps[index] >= self.settings.deactivation_steps:
                self.active[index] = False
                self._below_threshold_steps[index] = 0

        return DEMCouplingResult(
            velocity=velocity,
            traction=traction,
            slip_increment=slip,
            active=self.active.copy(),
            traction_residual=residual,
            kinetic_energy=kinetic,
        )

    def save_checkpoint(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(
            directory / "coupler_state.npz",
            active=self.active,
            below_threshold_steps=self._below_threshold_steps,
            last_velocity=self.last_velocity,
            last_kinetic_energy=self.last_kinetic_energy,
            last_suggested_dt=self.last_suggested_dt,
        )
        for index, backend in enumerate(self._backends):
            if backend is not None:
                backend.save_checkpoint(directory / f"rve_{index:05d}")

    def close(self) -> None:
        for backend in self._backends:
            if backend is not None:
                backend.close()

