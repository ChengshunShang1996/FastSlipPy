#/////////////////////////////////////////////////
__author__      = "Chengshun Shang (Utrecht University)"
__copyright__   = "Copyright (C) 2026-present by Chengshun Shang"
__version__     = "0.0.1"
__maintainer__  = "Chengshun Shang"
__email__       = "c.shang@uu.nl"
__status__      = "development"
__date__        = "Aug 14, 2026"
__license__     = "MIT License"
#/////////////////////////////////////////////////

import numpy as np

from fastslippy.fast_slip_py import FastSlipPy
from fastslippy.pre_processing.frictional_zones import FrictionalZones
from fastslippy.pre_processing.grid import Grid
from fastslippy.pre_processing.model_parameters import ModelParameters
from fastslippy.solver.fault_state import FaultState
from fastslippy.solver.stress_state import StressState


def _build_california_params() -> ModelParameters:
    params = ModelParameters(
        case_type="california",
        alpha=60.0,
        xsize=320e3,
        ysize=160e3,
        Nx=101,
        Ny=61,
        Vi=1e-9,
        W_f=40e3,
        x_stretch_enabled=True,
        y_stretch_enabled=True,
        x_stretch_inner_size=20e3,
        y_stretch_inner_size=20e3,
        x_stretch_inner_points=21,
        y_stretch_inner_points=21,
        x_stretch_power=2,
        y_stretch_power=2,
        allow_nonuniform_solver=True,
    )
    params.loading.V_L = 2.0e-9
    params.layers.set_homogeneous(top=params.ysize, bottom=params.ysize * 2.0, a=params.a0, b=params.b0)
    return params


def test_fault_loading_start_index_uses_grid_coordinates():
    params = _build_california_params()
    grid = Grid(params)
    fric = FrictionalZones(params, grid.y)
    stress = StressState(params, grid.y)
    fault = FaultState(params, stress, fric, fault_y=grid.y)

    expected = int(np.searchsorted(grid.y, params.W_f, side="left"))
    assert fault.california_loading_start_idx() == expected


def test_fault_velocity_loading_region_matches_wf_threshold():
    params = _build_california_params()
    grid = Grid(params)
    fric = FrictionalZones(params, grid.y)
    stress = StressState(params, grid.y)
    fault = FaultState(params, stress, fric, fault_y=grid.y)

    fault.solve_slip_rate_newton(np.zeros(params.Ny), stress, fric)
    start_idx = fault.california_loading_start_idx()

    np.testing.assert_allclose(fault.V[start_idx:], params.loading.V_L, rtol=0.0, atol=0.0)
    if start_idx > 1:
        assert np.any(np.abs(fault.V[1:start_idx] - params.loading.V_L) > 1e-20)


def test_adaptive_dt_window_uses_same_wf_indexing(tmp_path):
    params = _build_california_params()
    model = FastSlipPy(params=params, output_dir=str(tmp_path))
    start_idx = model.fault.california_loading_start_idx()

    model.fault.V = np.arange(params.Ny, dtype=float) + 1.0
    model.ksi = np.arange(params.Ny, dtype=float) + 101.0

    v_inner, ksi_inner = model._select_adaptive_fault_window()

    expected_upper = min(max(start_idx, 1), params.Ny - 1)
    if expected_upper > 1:
        np.testing.assert_allclose(v_inner, model.fault.V[1:expected_upper], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(ksi_inner, model.ksi[1:expected_upper], rtol=0.0, atol=0.0)
    else:
        np.testing.assert_allclose(v_inner, model.fault.V[1:params.Ny - 1], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(ksi_inner, model.ksi[1:params.Ny - 1], rtol=0.0, atol=0.0)
