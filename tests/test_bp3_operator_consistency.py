"""Checkpoint equilibrium/projection tests for the BP3 static operator."""

import numpy as np
from scipy.sparse.linalg import spsolve

from examples.run_bp3_small_peak_diagnostic import build_small_bp3_parameters
from fastslippy.pre_processing.grid import Grid
from fastslippy.solver.matrix_builder import MatrixBuilder
from fastslippy.utilities.bp3_operator_consistency import (
    absolute_displacement_rhs,
    diagnose_bp3_operator_consistency,
    pack_staggered_fields,
)


def _unpack_physical(solution, params):
    ux_full = solution[0::2].reshape(
        (params.Nx + 1, params.Ny + 1)
    ).T
    uy_full = solution[1::2].reshape(
        (params.Nx + 1, params.Ny + 1)
    ).T
    return ux_full[:, : params.Nx], uy_full[: params.Ny, :]


def test_absolute_rhs_separates_fault_slip_from_elapsed_loading_time():
    params = build_small_bp3_parameters()
    params.bc.left.uy.set_velocity(-0.5 * params.loading.V_p)
    params.bc.right.uy.set_velocity(0.5 * params.loading.V_p)
    builder = MatrixBuilder(params, Grid(params))
    slip = np.linspace(0.0, 0.02, params.Ny)
    time = 7.5e6
    zero = np.zeros(params.Ny)
    loading = builder.build_RH(0.0, zero).copy()
    combined = builder.build_RH(0.0, slip).copy()

    actual = absolute_displacement_rhs(builder, slip, time)
    expected = (combined - loading) + time * loading
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_static_checkpoint_has_small_residual_and_projection_error():
    params = build_small_bp3_parameters()
    grid = Grid(params)
    builder = MatrixBuilder(params, grid)
    lhs = builder.build_LH().tocsc()
    slip = 0.01 * np.sin(np.pi * grid.y / grid.y[-1])
    rhs = absolute_displacement_rhs(builder, slip, time=0.0)
    solution = spsolve(lhs, rhs)
    ux, uy = _unpack_physical(solution, params)
    checkpoint = {"U": slip, "ux": ux, "uy": uy, "t": np.array(0.0)}
    modes = np.column_stack((
        np.sin(np.pi * grid.y / grid.y[-1]),
        np.sin(2.0 * np.pi * grid.y / grid.y[-1]),
    ))

    result = diagnose_bp3_operator_consistency(
        params, checkpoint, modes=modes, project=True
    )

    assert (
        result.checkpoint.residual_groups["physical_rows"].relative_l2
        < 2e-12
    )
    assert result.checkpoint.displacement_relative_l2 < 2e-12
    assert result.reciprocity is not None
    assert np.all(np.isfinite(result.reciprocity.work_matrix))

    repacked = pack_staggered_fields(params, ux, uy)
    np.testing.assert_allclose(repacked, solution, rtol=0.0, atol=2e-12)
