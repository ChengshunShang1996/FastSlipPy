import numpy as np

from examples.run_case_bp3_real_case import YR, build_bp3_90_parameters
from fastslippy.pre_processing.grid import Grid


def test_real_90_degree_case_resolves_complete_frictional_fault_uniformly():
    params = build_bp3_90_parameters()
    grid = Grid(params)

    assert params.alpha == 90.0
    assert params.motion_sign == -1
    assert params.tfinal == 500.0 * YR
    assert params.dt_max == 0.1 * YR
    assert params.friction_tolerance == 5.0
    assert params.extrapolate_surface_fault_rate
    assert params.y_stretch_inner_size >= params.W_f

    rate_state_nodes = grid.y <= params.W_f
    np.testing.assert_allclose(
        np.diff(grid.y[rate_state_nodes]),
        100.0,
        rtol=0.0,
        atol=1e-11,
    )
    assert grid.y[np.searchsorted(grid.y, params.W_f)] == params.W_f
    assert np.max(grid.dy_edges) <= params.y_stretch_max_cell_size


def test_real_90_degree_case_uses_side_loading_and_free_bottom():
    params = build_bp3_90_parameters()

    assert params.bc.left.uy.value == -0.5 * params.loading.V_p
    assert params.bc.right.uy.value == 0.5 * params.loading.V_p
    assert params.bc.bottom.ux.type.name == "TRACTION_FREE"
    assert params.bc.bottom.uy.type.name == "TRACTION_FREE"
