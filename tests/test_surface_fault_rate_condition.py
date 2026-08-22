import numpy as np
import pytest

from fastslippy.pre_processing.frictional_zones import FrictionalZones
from fastslippy.pre_processing.grid import Grid
from fastslippy.pre_processing.model_parameters import ModelParameters
from fastslippy.solver.fault_state import FaultState
from fastslippy.solver.stress_state import StressState


def _vertical_fault(extrapolate: bool):
    params = ModelParameters(
        case_type="california",
        alpha=90.0,
        Nx=11,
        Ny=9,
        xsize=1000.0,
        ysize=800.0,
        W_f=700.0,
        H=300.0,
        h=100.0,
        rho=2670.0,
        cs=3464.0,
        nu=0.25,
        mu0=0.6,
        V0=1e-6,
        a0=0.01,
        a_max=0.025,
        b0=0.015,
        L=0.008,
        Vi=1e-9,
        extrapolate_surface_fault_rate=extrapolate,
    )
    params.loading.V_p = 1e-9
    params.loading.V_L = 1e-9
    grid = Grid(params)
    friction = FrictionalZones(params, grid.y)
    stress = StressState(params, grid.y)
    fault = FaultState(params, stress, friction, fault_y=grid.y)
    return params, friction, stress, fault


@pytest.mark.parametrize(
    "solver_name",
    ["solve_slip_rate_newton_v2", "solve_slip_rate_matlab"],
)
def test_bp3_surface_rate_uses_first_interior_limit(solver_name):
    _, friction, stress, fault = _vertical_fault(extrapolate=True)
    stress_step = np.zeros(fault.p.Ny)
    stress_step[0] = -1.0e5
    stress_step[1] = 1.0e5

    getattr(fault, solver_name)(stress_step, stress, friction)

    assert fault.V[0] == fault.V[1]


def test_surface_rate_extrapolation_can_be_disabled_for_legacy_parity():
    _, friction, stress, fault = _vertical_fault(extrapolate=False)
    stress_step = np.zeros(fault.p.Ny)
    stress_step[0] = -1.0e5
    stress_step[1] = 1.0e5

    fault.solve_slip_rate_newton_v2(stress_step, stress, friction)

    assert not np.isclose(fault.V[0], fault.V[1], rtol=1e-3, atol=0.0)
