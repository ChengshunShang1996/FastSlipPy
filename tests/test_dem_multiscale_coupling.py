from pathlib import Path

import numpy as np
import pytest

from fastslippy.fast_slip_py import FastSlipPy
from fastslippy.pre_processing.model_parameters import (
    CaseType,
    DEMCouplingParameters,
    FrictionLaw,
    ModelParameters,
    TimeIntegrator,
)
from fastslippy.solver.dem_coupling import (
    DEMMultiscaleCoupler,
    DEMRVEResult,
    DEMWallState,
)


class DeterministicRVE:
    def __init__(self, index, calls, normal, drive, returned_velocity=2.0e-5):
        self.index = index
        self.calls = calls
        self.returned_velocity = returned_velocity
        self.saved = []
        self.closed = False
        self.wall_state = DEMWallState(0.0, drive, normal, 0.0, 0.0, 0)

    def advance(
        self,
        *,
        driving_shear_stress,
        effective_normal_stress,
        radiation_damping,
        duration,
        initial_velocity,
    ):
        self.calls.append(
            (
                self.index,
                driving_shear_stress,
                effective_normal_stress,
                radiation_damping,
                duration,
                initial_velocity,
            )
        )
        velocity = self.returned_velocity
        previous = self.wall_state
        self.wall_state = DEMWallState(
            2 * velocity,
            driving_shear_stress - 2 * radiation_damping * velocity,
            effective_normal_stress,
            previous.loading_time + duration,
            previous.loading_slip + velocity * duration,
            previous.loading_steps + 1,
        )
        return DEMRVEResult(
            slip_increment=velocity * duration,
            average_velocity=velocity,
            shear_traction=driving_shear_stress - radiation_damping * velocity,
            normal_traction=effective_normal_stress,
        )

    def save_checkpoint(self, directory: Path):
        self.saved.append(directory)
        directory.mkdir(parents=True, exist_ok=True)

    def close(self):
        self.closed = True


class DeterministicFactory:
    def __init__(self, returned_velocity=2.0e-5):
        self.calls = []
        self.instances = []
        self.returned_velocity = returned_velocity

    def __call__(
        self,
        index,
        initial_normal_stress,
        initial_velocity,
        *,
        initial_driving_shear_stress,
        radiation_damping,
    ):
        backend = DeterministicRVE(
            index,
            self.calls,
            initial_normal_stress,
            initial_driving_shear_stress,
            returned_velocity=self.returned_velocity,
        )
        self.instances.append(backend)
        return backend


def test_all_nodes_preloaded_before_quiet_step_and_return_mean_and_end_state():
    factory = DeterministicFactory()
    coupler = DEMMultiscaleCoupler(3, DEMCouplingParameters(), factory)
    drive = np.array([0.0, 500.0, -500.0])
    normal = np.array([5000.0, 10000.0, 15000.0])
    v0, tau0 = coupler.initialize(drive, normal, 4.0)
    np.testing.assert_array_equal(v0, 0)
    np.testing.assert_array_equal(tau0, drive)
    assert len(factory.instances) == 3 and not factory.calls
    for _ in range(3):
        result = coupler.advance(
            driving_shear_stress=drive,
            effective_normal_stress=normal,
            radiation_damping=4.0,
            dt=0.25,
        )
    assert len(factory.calls) == 9
    np.testing.assert_allclose(result.average_velocity, 2e-5)
    np.testing.assert_allclose(result.end_velocity, 4e-5)
    np.testing.assert_allclose(result.slip_increment, result.average_velocity * 0.25)
    np.testing.assert_array_equal(result.average_normal, normal)


def test_preload_failure_closes_prepared_nodes_and_blocks_advance():
    factory = DeterministicFactory()

    def fail_second(index, *args, **kwargs):
        if index == 1:
            raise RuntimeError("preload failed")
        return factory(index, *args, **kwargs)

    coupler = DEMMultiscaleCoupler(3, DEMCouplingParameters(), fail_second)
    normal = np.ones(3)
    with pytest.raises(RuntimeError, match="preload failed"):
        coupler.initialize(normal, normal, 0.0)
    assert factory.instances[0].closed
    with pytest.raises(RuntimeError, match="Initialize DEM"):
        coupler.advance(
            driving_shear_stress=normal,
            effective_normal_stress=normal,
            radiation_damping=0.0,
            dt=0.1,
        )


def test_dem_configuration_is_optional_and_rejects_midpoint():
    assert ModelParameters().friction_law is FrictionLaw.RATE_STATE
    with pytest.raises(ValueError, match="Euler"):
        ModelParameters(
            friction_law="dem_multiscale",
            time_integrator=TimeIntegrator.RK2_MIDPOINT,
        )


def test_prescribed_creep_and_extrapolated_surface_are_kinematic():
    factory = DeterministicFactory()
    coupler = DEMMultiscaleCoupler(
        5, DEMCouplingParameters(), factory, creep_start=3, creep_velocity=-1e-8,
        extrapolate_surface=True,
    )
    normal = np.full(5, 1e4)
    drive = np.arange(5, dtype=float)
    coupler.initialize(drive, normal, 4.0)
    result = coupler.advance(driving_shear_stress=drive, effective_normal_stress=normal,
                             radiation_damping=4.0, dt=0.25)
    assert list(coupler.backends) == [1, 2]
    np.testing.assert_equal(result.slip_increment[0], result.slip_increment[1])
    np.testing.assert_equal(result.end_velocity[3:], -1e-8)
    np.testing.assert_equal(result.slip_increment[3:], -0.25e-8)


def test_fastslippy_dem_output_does_not_advance_rve_twice(tmp_path, monkeypatch):
    dem = DEMCouplingParameters(
        activation_velocity=1.0e-300,
        deactivation_velocity=0.0,
        deactivation_steps=5,
        max_macro_dt=1.0e-4,
        max_slip_increment=1.0,
    )
    params = ModelParameters(
        case_type=CaseType.LAB,
        friction_law=FrictionLaw.DEM_MULTISCALE,
        dem=dem,
        Nx=5,
        Ny=5,
        Nt=1,
        dt_init=1.0e-4,
        dt_max=1.0e-4,
        tfinal=1.0e-4,
        output_interval=1,
        checkpoint_interval=10,
        output_vtk_option=False,
        Vi=1.0e-40,
    )
    factory = DeterministicFactory(returned_velocity=2.0e-6)
    model = FastSlipPy(
        params,
        output_dir=str(tmp_path),
        dem_backend_factory=factory,
    )
    model.figure_creator.plot_results_shear = lambda *args, **kwargs: None

    def unexpected(*args, **kwargs):
        pytest.fail("DEM called an RSF solver or state update")

    monkeypatch.setattr(model, "_solve_fault_slip_rate", unexpected)
    monkeypatch.setattr(model.fault, "advance", unexpected)
    assert model.fric is None and model.ksi is None
    model.run()

    # Each node advances once.  Synchronized output and the final
    # checkpoint only read/cache state and must not run another DEM interval.
    assert len(factory.calls) == params.Ny
    np.testing.assert_allclose(model.fault.V, 4.0e-6)
    np.testing.assert_allclose(model.fault.U, 2.0e-10)
