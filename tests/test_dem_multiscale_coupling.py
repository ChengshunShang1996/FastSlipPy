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
from fastslippy.solver.dem_coupling import DEMMultiscaleCoupler, DEMRVEResult


class DeterministicRVE:
    def __init__(self, index, calls, returned_velocity=2.0e-5):
        self.index = index
        self.calls = calls
        self.returned_velocity = returned_velocity
        self.saved = []

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
        pass


class DeterministicFactory:
    def __init__(self, returned_velocity=2.0e-5):
        self.calls = []
        self.instances = []
        self.returned_velocity = returned_velocity

    def __call__(self, index, initial_normal_stress, initial_velocity):
        backend = DeterministicRVE(
            index, self.calls, returned_velocity=self.returned_velocity
        )
        self.instances.append(backend)
        return backend


def test_dem_coupler_activates_only_fast_nodes_and_balances_radiation_damping():
    settings = DEMCouplingParameters(
        activation_velocity=1.0e-4,
        deactivation_velocity=1.0e-6,
    )
    factory = DeterministicFactory(returned_velocity=2.0e-5)
    coupler = DEMMultiscaleCoupler(3, settings, factory)
    predictor = np.array([1.0e-5, 2.0e-4, -3.0e-4])
    driving = np.array([10.0, 20.0, -30.0])
    normal = np.full(3, 50.0)

    result = coupler.advance(
        predictor_velocity=predictor,
        driving_shear_stress=driving,
        effective_normal_stress=normal,
        radiation_damping=4.0,
        dt=0.25,
    )

    assert [call[0] for call in factory.calls] == [1, 2]
    assert result.velocity[0] == predictor[0]
    np.testing.assert_allclose(result.velocity[1:], 2.0e-5)
    np.testing.assert_allclose(result.traction_residual[1:], 0.0)
    np.testing.assert_allclose(result.slip_increment, result.velocity * 0.25)


def test_dem_configuration_is_optional_and_rejects_midpoint():
    assert ModelParameters().friction_law is FrictionLaw.RATE_STATE
    with pytest.raises(ValueError, match="Euler"):
        ModelParameters(
            friction_law="dem_multiscale",
            time_integrator=TimeIntegrator.RK2_MIDPOINT,
        )


def test_fastslippy_dem_output_does_not_advance_rve_twice(tmp_path):
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

    model.run()

    # Each active node advances once.  Synchronized output and the final
    # checkpoint only read/cache state and must not run another DEM interval.
    assert len(factory.calls) == params.Ny
    np.testing.assert_allclose(model.fault.V, 2.0e-6)
    np.testing.assert_allclose(model.fault.U, 2.0e-10)

