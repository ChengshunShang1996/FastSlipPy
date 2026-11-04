import numpy as np

from fastslippy.pre_processing.grid import Grid
from fastslippy.pre_processing.model_parameters import BCType, ModelParameters
from fastslippy.solver.matrix_builder import MatrixBuilder


def test_horizontal_section_override_can_be_cleared():
    params = ModelParameters(Nx=11, Ny=9)
    params.bc.top.uy.set_velocity(2.0)
    params.bc.top.left.uy.set_traction_free()

    assert params.bc.top.left.uy.type is BCType.TRACTION_FREE
    assert params.bc.top.left.uy.is_set

    params.bc.top.left.uy.clear()

    assert params.bc.top.left.uy.type is None
    assert not params.bc.top.left.uy.is_set


def test_split_top_uy_supports_different_operator_types_and_values():
    params = ModelParameters(Nx=11, Ny=9)
    velocity = 2.5e-6
    traction_rate = 3.0e5
    params.bc.top.left.uy.set_velocity(velocity)
    params.bc.top.right.uy.set_traction(traction_rate)

    builder = MatrixBuilder(params, Grid(params))
    matrix = builder.build_LH()
    rhs = builder.build_RH(0.0, np.zeros(params.Ny)).copy()
    mid = params.Nx // 2
    left_ix = mid - 1
    right_ix = mid + 2

    left_row = builder._kuy[0, left_ix]
    right_row = builder._kuy[0, right_ix]
    assert matrix.getrow(left_row).nnz == 1
    assert matrix[left_row, left_row] == 1.0
    assert matrix.getrow(right_row).nnz > 1
    assert rhs[left_row] == velocity
    np.testing.assert_allclose(
        rhs[right_row],
        -builder._dx_xuy[right_ix] / params.G * traction_rate,
    )


def test_split_bottom_ux_supports_free_and_velocity_sections():
    params = ModelParameters(Nx=11, Ny=9)
    velocity = -4.0e-6
    params.bc.bottom.left.ux.set_free()
    params.bc.bottom.right.ux.set_velocity(velocity)

    builder = MatrixBuilder(params, Grid(params))
    matrix = builder.build_LH()
    rhs = builder.build_RH(0.0, np.zeros(params.Ny)).copy()
    mid = params.Nx // 2
    left_ix = mid - 1
    right_ix = mid + 2
    left_row = builder._kux[params.Ny, left_ix]
    right_row = builder._kux[params.Ny, right_ix]

    left_entries = dict(
        zip(matrix.getrow(left_row).indices, matrix.getrow(left_row).data)
    )
    assert left_entries == {left_row - 2: -1.0, left_row: 1.0}
    assert matrix.getrow(right_row).nnz == 1
    assert matrix[right_row, right_row] == 1.0
    assert rhs[left_row] == 0.0
    assert rhs[right_row] == velocity


def test_unset_section_inherits_effective_legacy_lab_loading():
    params = ModelParameters(Nx=11, Ny=9)
    params.bc.top.uy.set_velocity(7.0e-6)
    # Activates split resolution while leaving the left section inherited.
    params.bc.top.right.uy.set_velocity(9.0e-6)

    builder = MatrixBuilder(params, Grid(params))
    rhs = builder.build_RH(0.0, np.zeros(params.Ny)).copy()
    mid = params.Nx // 2

    assert rhs[builder._kuy[0, mid - 1]] == 0.0
    assert rhs[builder._kuy[0, mid + 2]] == 9.0e-6


def test_split_values_can_change_without_rebuilding_matrix():
    params = ModelParameters(Nx=11, Ny=9)
    params.bc.top.left.uy.set_velocity(1.0)
    params.bc.top.right.uy.set_velocity(2.0)
    builder = MatrixBuilder(params, Grid(params))
    builder.build_LH()
    mid = params.Nx // 2

    rhs_1 = builder.build_RH(0.0, np.zeros(params.Ny)).copy()
    params.bc.top.left.uy.value = 3.0
    params.bc.top.right.uy.value = 4.0
    rhs_2 = builder.build_RH(0.0, np.zeros(params.Ny)).copy()

    assert rhs_1[builder._kuy[0, mid - 1]] == 1.0
    assert rhs_1[builder._kuy[0, mid + 2]] == 2.0
    assert rhs_2[builder._kuy[0, mid - 1]] == 3.0
    assert rhs_2[builder._kuy[0, mid + 2]] == 4.0


def test_fault_owned_surface_rows_take_precedence_over_split_bc():
    params = ModelParameters(
        Nx=11,
        Ny=9,
        fault_reaches_surface=True,
        fault_reaches_bottom=True,
    )
    params.bc.top.left.uy.set_velocity(-10.0)
    params.bc.top.right.uy.set_velocity(10.0)
    builder = MatrixBuilder(params, Grid(params))
    matrix = builder.build_LH()
    fault_velocity = np.linspace(1.0, 2.0, params.Ny)
    rhs = builder.build_RH(0.0, fault_velocity).copy()
    mid = params.Nx // 2

    jump_row = builder._kuy[0, mid]
    continuity_row = builder._kuy[0, mid + 1]
    jump_entries = dict(
        zip(matrix.getrow(jump_row).indices, matrix.getrow(jump_row).data)
    )
    assert jump_entries == {
        jump_row: -1.0,
        jump_row + 2 * (params.Ny + 1): 1.0,
    }
    assert matrix.getrow(continuity_row).nnz > 1
    assert rhs[jump_row] == fault_velocity[0]
    assert rhs[continuity_row] == 0.0
