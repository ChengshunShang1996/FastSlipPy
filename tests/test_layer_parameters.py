import numpy as np
import pytest

from fastslippy.pre_processing.frictional_zones import FrictionalZones
from fastslippy.pre_processing.grid import Grid
from fastslippy.pre_processing.model_parameters import ModelParameters
from fastslippy.solver.fault_state import FaultState
from fastslippy.solver.stress_state import StressState


def _layered_parameters() -> ModelParameters:
    params = ModelParameters(
        case_type="lab",
        xsize=1.0,
        ysize=10.0,
        Nx=11,
        Ny=6,
        V0=0.1,
        D_rs=0.4,
    )
    params.layers.add("Upper", 10.0, 15.0, 0.01, 0.015)
    params.layers.add("Lower", 15.0, 20.0, 0.02, 0.025, D_rs=0.2)
    return params


def test_layers_inherit_or_override_characteristic_distance():
    params = _layered_parameters()
    grid = Grid(params)
    friction = FrictionalZones(params, grid.y)

    np.testing.assert_allclose(friction.D_rs[:3], params.D_rs)
    np.testing.assert_allclose(friction.D_rs[3:], 0.2)


def test_fault_state_uses_local_characteristic_distance():
    params = _layered_parameters()
    grid = Grid(params)
    friction = FrictionalZones(params, grid.y)
    stress = StressState(params, grid.y)
    fault = FaultState(params, stress, friction, fault_y=grid.y)

    np.testing.assert_allclose(fault.theta, friction.D_rs / params.V0)

    theta = np.ones(params.Ny)
    velocity = np.full(params.Ny, 0.2)
    dt = 0.5
    exponent = velocity * dt / friction.D_rs
    expected = (
        friction.D_rs / velocity * (1.0 - np.exp(-exponent))
        + theta * np.exp(-exponent)
    )
    np.testing.assert_allclose(
        fault.theta_after_constant_velocity(theta, velocity, dt), expected
    )


def test_legacy_two_profile_builder_inherits_global_characteristic_distance():
    class LegacyFrictionalZones(FrictionalZones):
        def build(self):
            return (
                np.full_like(self.y, self.p.a0),
                np.full_like(self.y, self.p.b0),
            )

    params = ModelParameters(Ny=6, D_rs=0.3)
    grid = Grid(params)
    friction = LegacyFrictionalZones(params, grid.y)

    np.testing.assert_allclose(friction.D_rs, params.D_rs)


@pytest.mark.parametrize("D_rs", [0.0, -1.0, np.inf, np.nan])
def test_model_rejects_invalid_characteristic_distance(D_rs):
    with pytest.raises(ValueError, match="D_rs must be finite and positive"):
        ModelParameters(D_rs=D_rs)


@pytest.mark.parametrize("D_rs", [0.0, -1.0, np.inf, np.nan])
def test_layer_rejects_invalid_characteristic_distance(D_rs):
    with pytest.raises(
        ValueError, match="Layer D_rs must be finite and positive"
    ):
        ModelParameters().layers.add(
            "Invalid", 0.0, 1.0, 0.01, 0.015, D_rs=D_rs
        )
