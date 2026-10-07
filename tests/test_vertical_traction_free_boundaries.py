import numpy as np

from fastslippy.pre_processing.grid import Grid
from fastslippy.pre_processing.model_parameters import FaultMode, ModelParameters
from fastslippy.solver.matrix_builder import MatrixBuilder


def test_vertical_traction_free_rows_accept_zero_traction_affine_field():
    """Left/right traction rows vanish for sigma_xx = sigma_xy = 0."""
    params = ModelParameters(
        fault_mode=FaultMode.NONE,
        alpha=90.0,
        xsize=0.1,
        ysize=0.05,
        Nx=10,
        Ny=9,
        E=1.0e10,
    )
    params.bc.left.set_traction_free()
    params.bc.right.set_traction_free()
    params.bc.top.ux.set_free()
    params.bc.bottom.ux.set_free()

    strain_yy = 2.0e-4
    strain_xx = -params.lam / (params.lam + 2.0 * params.G) * strain_yy
    params.bc.top.uy.set_velocity(strain_yy * 0.0)
    params.bc.bottom.uy.set_velocity(strain_yy * params.ysize)

    grid = Grid(params)
    builder = MatrixBuilder(params, grid)
    matrix = builder.build_LH()
    rhs = builder.build_RH(0.0, np.zeros(params.Ny))
    state = np.zeros(grid.N)

    for ix in range(params.Nx + 1):
        for iy in range(params.Ny + 1):
            kux, kuy = builder._dofs(ix, iy, params.Ny)
            if ix < params.Nx:
                state[kux] = strain_xx * grid.x[ix]
            if iy < params.Ny:
                state[kuy] = strain_yy * grid.y[iy]

    residual = matrix @ state - rhs
    side_rows = []
    for iy in range(params.Ny):
        side_rows.extend((builder._kuy[iy, 0], builder._kuy[iy, params.Nx]))
    for iy in range(1, params.Ny):
        side_rows.extend((builder._kux[iy, 0], builder._kux[iy, params.Nx - 1]))

    np.testing.assert_allclose(residual[side_rows], 0.0, atol=1.0e-14)

