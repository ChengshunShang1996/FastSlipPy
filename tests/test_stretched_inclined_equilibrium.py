import numpy as np

from fastslippy.pre_processing.grid import Grid
from fastslippy.pre_processing.model_parameters import ModelParameters
from fastslippy.solver.matrix_builder import MatrixBuilder


def test_stretched_inclined_bulk_annuls_global_harmonic_solution():
    """Stretched inclined bulk rows must preserve a Navier equilibrium.

    ``phi = X**3 - 3*X*Z**2`` is harmonic, so ``grad(phi)`` is an exact
    homogeneous isotropic-elastic equilibrium.  This detects treating the
    mixed derivatives as if the stretched grid had a single local dx/dy.
    """
    for alpha in (30.0, 60.0, 90.0):
        params = ModelParameters(
            case_type="california",
            alpha=alpha,
            xsize=80e3,
            ysize=80e3,
            Nx=81,
            Ny=81,
            x_stretch_enabled=True,
            y_stretch_enabled=True,
            x_stretch_inner_size=20e3,
            y_stretch_inner_size=20e3,
            x_stretch_inner_points=41,
            y_stretch_inner_points=41,
            allow_nonuniform_solver=True,
        )
        grid = Grid(params)
        builder = MatrixBuilder(params, grid)

        ux_global = 3.0 * grid.Xux**2 - 3.0 * grid.Yux**2
        uz_global = -6.0 * grid.Xux * grid.Yux
        ux = ux_global - grid.cosa * uz_global / grid.sina
        uy = -6.0 * grid.Xuy * grid.Yuy / grid.sina

        solution = np.zeros(grid.N)
        for ix in range(params.Nx + 1):
            for iy in range(params.Ny + 1):
                kux, kuy = builder._dofs(ix, iy, params.Ny)
                if ix < params.Nx:
                    solution[kux] = ux[iy, ix]
                if iy < params.Ny:
                    solution[kuy] = uy[iy, ix]

        residual = builder.build_LH() @ solution
        bulk_rows = []
        mid = params.Nx // 2
        for ix in range(2, params.Nx - 1):
            if ix in (mid, mid + 1):
                continue
            for iy in range(2, params.Ny - 2):
                kux, kuy = builder._dofs(ix, iy, params.Ny)
                bulk_rows.extend((kux, kuy))

        relative_residual = (
            np.max(np.abs(residual[bulk_rows])) / np.max(np.abs(solution))
        )
        # Before the coordinate-aware bulk operator, the 30°/60° residual
        # was about 8e-3 on this mesh.
        assert relative_residual < 1e-12
