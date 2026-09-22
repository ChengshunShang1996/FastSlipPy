import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from fastslippy.pre_processing.grid import Grid
from fastslippy.pre_processing.model_parameters import FaultMode, ModelParameters
from fastslippy.solver.matrix_builder import MatrixBuilder


@pytest.mark.parametrize("loaded_side", ["left", "right"])
def test_normal_traction_rate_drives_expected_uniaxial_velocity(loaded_side):
    params = ModelParameters(
        fault_mode=FaultMode.NONE,
        alpha=90.0,
        xsize=0.1,
        ysize=0.05,
        Nx=10,
        Ny=9,
        E=1.0e10,
    )
    compression_rate = 2.0e6
    params.bc.left.uy.set_fixed()
    params.bc.right.uy.set_fixed()
    params.bc.top.uy.set_fixed()
    params.bc.bottom.uy.set_fixed()
    params.bc.top.ux.set_free()
    params.bc.bottom.ux.set_free()
    if loaded_side == "right":
        params.bc.left.ux.set_fixed()
        params.bc.right.ux.set_traction(-compression_rate)
    else:
        params.bc.left.ux.set_traction(compression_rate)
        params.bc.right.ux.set_fixed()

    grid = Grid(params)
    builder = MatrixBuilder(params, grid)
    solution = spsolve(
        builder.build_LH().tocsc(),
        builder.build_RH(0.0, np.zeros(params.Ny)),
    )
    vx = np.reshape(
        solution[0::2], (params.Nx + 1, params.Ny + 1), order="C"
    ).T[:, :params.Nx]

    strain_rate = -compression_rate / (params.lam + 2.0 * params.G)
    anchor = grid.x[0] if loaded_side == "right" else grid.x[-1]
    expected = strain_rate * (grid.x - anchor)
    np.testing.assert_allclose(
        vx, np.broadcast_to(expected, vx.shape), rtol=2.0e-12, atol=1.0e-15
    )


def test_nonzero_traction_rows_match_affine_stress_rate():
    """All four boundaries use outward-traction component rates."""
    params = ModelParameters(
        fault_mode=FaultMode.NONE,
        alpha=90.0,
        xsize=0.1,
        ysize=0.05,
        Nx=10,
        Ny=9,
        E=1.0e10,
    )
    exx, dux_dy = -2.0e-4, 3.0e-5
    duy_dx, eyy = -1.0e-5, 8.0e-5
    sigma_xx = (params.lam + 2.0 * params.G) * exx + params.lam * eyy
    sigma_yy = params.lam * exx + (params.lam + 2.0 * params.G) * eyy
    sigma_xy = params.G * (dux_dy + duy_dx)

    # t = sigma . n, with outward normals left/right/top/bottom equal to
    # (-1, 0), (1, 0), (0, -1), and (0, 1), respectively.
    params.bc.left.ux.set_traction(-sigma_xx)
    params.bc.left.uy.set_traction(-sigma_xy)
    params.bc.right.ux.set_traction(sigma_xx)
    params.bc.right.uy.set_traction(sigma_xy)
    params.bc.top.ux.set_traction(-sigma_xy)
    params.bc.top.uy.set_traction(-sigma_yy)
    params.bc.bottom.ux.set_traction(sigma_xy)
    params.bc.bottom.uy.set_traction(sigma_yy)

    grid = Grid(params)
    builder = MatrixBuilder(params, grid)
    matrix = builder.build_LH()
    rhs = builder.build_RH(0.0, np.zeros(params.Ny)).copy()
    state = np.zeros(grid.N)

    for ix in range(params.Nx + 1):
        for iy in range(params.Ny + 1):
            kux, kuy = builder._dofs(ix, iy, params.Ny)
            if ix < params.Nx:
                state[kux] = exx * grid.x[ix] + dux_dy * grid.yp[iy]
            if iy < params.Ny:
                state[kuy] = duy_dx * grid.xp[ix] + eyy * grid.y[iy]

    np.testing.assert_allclose(matrix @ state, rhs, atol=2.0e-14)


def test_traction_rhs_tracks_updated_rate_without_rebuilding_matrix():
    params = ModelParameters(
        fault_mode=FaultMode.NONE,
        alpha=90.0,
        Nx=10,
        Ny=9,
    )
    params.bc.left.set_traction_free()
    params.bc.top.set_traction_free()
    params.bc.bottom.set_traction_free()
    params.bc.right.ux.set_traction(-2.0e6)
    params.bc.right.uy.set_traction(3.0e5)

    builder = MatrixBuilder(params, Grid(params))
    builder.build_LH()
    rhs_1 = builder.build_RH(0.0, np.zeros(params.Ny)).copy()
    params.bc.right.ux.value *= 2.0
    params.bc.right.uy.value *= 2.0
    rhs_2 = builder.build_RH(0.0, np.zeros(params.Ny)).copy()

    np.testing.assert_allclose(rhs_2, 2.0 * rhs_1)
