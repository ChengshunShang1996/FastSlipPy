import numpy as np

from fastslippy.pre_processing.grid import Grid
from fastslippy.pre_processing.model_parameters import (
    BoundaryProfile,
    FaultBottomTreatment,
    FaultSurfaceTreatment,
    ModelParameters,
)
from fastslippy.solver.matrix_builder import MatrixBuilder


def _row_coefficients(matrix, row):
    sparse_row = matrix.getrow(row)
    return dict(zip(sparse_row.indices, sparse_row.data))


def test_case_presets_keep_existing_fault_endpoint_treatments():
    bp3 = ModelParameters(case_type="california", Nx=7, Ny=7)
    lab = ModelParameters(case_type="lab", Nx=7, Ny=7)
    groningen = ModelParameters(case_type="groningen", Nx=7, Ny=7)

    assert (
        bp3.fault_surface_treatment
        is FaultSurfaceTreatment.FREE_SURFACE_FAULT_INTERSECTION
    )
    assert (
        bp3.fault_bottom_treatment
        is FaultBottomTreatment.DEEP_BOUNDARY_FAULT_INTERSECTION
    )

    for general_case in (lab, groningen):
        assert (
            general_case.fault_surface_treatment
            is FaultSurfaceTreatment.EXTERNAL_BOUNDARY
        )
        assert (
            general_case.fault_bottom_treatment
            is FaultBottomTreatment.EXTERNAL_BOUNDARY
        )


def test_staggered_dirichlet_values_are_imposed_at_physical_faces():
    params = ModelParameters(case_type="lab", Nx=7, Ny=7)
    params.bc.left.uy.set_velocity(2.0)
    params.bc.right.uy.set_velocity(3.0)
    params.bc.top.ux.set_velocity(5.0)
    params.bc.bottom.ux.set_velocity(7.0)

    grid = Grid(params)
    builder = MatrixBuilder(params, grid)
    matrix = builder.build_LH()
    rhs = builder.build_RH(0.0, np.zeros(params.Ny))
    iy = 2
    ix = 2

    _, left_uy = builder._dofs(0, iy, params.Ny)
    _, left_uy_inner = builder._dofs(1, iy, params.Ny)
    assert _row_coefficients(matrix, left_uy) == {
        left_uy: 1.0,
        left_uy_inner: 1.0,
    }
    assert rhs[left_uy] == 4.0

    _, right_uy = builder._dofs(params.Nx, iy, params.Ny)
    _, right_uy_inner = builder._dofs(params.Nx - 1, iy, params.Ny)
    assert _row_coefficients(matrix, right_uy) == {
        right_uy_inner: 1.0,
        right_uy: 1.0,
    }
    assert rhs[right_uy] == 6.0

    top_ux, _ = builder._dofs(ix, 0, params.Ny)
    top_ux_inner, _ = builder._dofs(ix, 1, params.Ny)
    assert _row_coefficients(matrix, top_ux) == {
        top_ux: 1.0,
        top_ux_inner: 1.0,
    }
    assert rhs[top_ux] == 10.0

    bottom_ux, _ = builder._dofs(ix, params.Ny, params.Ny)
    bottom_ux_inner, _ = builder._dofs(ix, params.Ny - 1, params.Ny)
    assert _row_coefficients(matrix, bottom_ux) == {
        bottom_ux_inner: 1.0,
        bottom_ux: 1.0,
    }
    assert rhs[bottom_ux] == 14.0


def test_fault_and_traction_free_discretisation_no_longer_depend_on_case_name():
    common = dict(
        alpha=60.0,
        xsize=12.0,
        ysize=10.0,
        Nx=7,
        Ny=7,
        fault_surface_treatment="external_boundary",
        fault_bottom_treatment="external_boundary",
    )
    lab = ModelParameters(case_type="lab", **common)
    bp3_named = ModelParameters(case_type="california", **common)
    for params in (lab, bp3_named):
        params.bc.top.ux.set_traction_free()
        params.bc.top.uy.set_traction_free()

    lab_matrix = MatrixBuilder(lab, Grid(lab)).build_LH()
    bp3_named_matrix = MatrixBuilder(bp3_named, Grid(bp3_named)).build_LH()

    np.testing.assert_allclose(
        lab_matrix.toarray(), bp3_named_matrix.toarray(), rtol=0.0, atol=0.0
    )


def test_velocity_profile_is_owned_by_each_boundary_condition():
    params = ModelParameters(case_type="lab", Nx=7, Ny=7)
    params.bc.top.uy.set_velocity(2.0, profile="positive_fault_block")
    params.bc.bottom.uy.set_velocity(
        3.0, profile="antisymmetric_about_fault"
    )

    assert params.bc.top.uy.profile is BoundaryProfile.POSITIVE_FAULT_BLOCK
    assert (
        params.bc.bottom.uy.profile
        is BoundaryProfile.ANTISYMMETRIC_ABOUT_FAULT
    )

    builder = MatrixBuilder(params, Grid(params))
    rhs = builder.build_RH(0.0, np.zeros(params.Ny))
    mid = params.Nx // 2

    np.testing.assert_allclose(rhs[builder._kuy[0, 1:mid + 1]], 0.0)
    np.testing.assert_allclose(rhs[builder._kuy[0, mid + 1:params.Nx]], 2.0)
    np.testing.assert_allclose(rhs[builder._kuy[-1, 1:mid + 1]], -3.0)
    np.testing.assert_allclose(rhs[builder._kuy[-1, mid + 1:params.Nx]], 3.0)
