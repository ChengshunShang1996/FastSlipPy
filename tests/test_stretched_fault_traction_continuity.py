import numpy as np
from scipy.sparse.linalg import spsolve

from fastslippy.pre_processing.grid import Grid
from fastslippy.pre_processing.model_parameters import ModelParameters
from fastslippy.solver.matrix_builder import MatrixBuilder
from fastslippy.utilities.stress_cal_util import StressCalUtil


def test_stretched_inclined_fault_uses_recovered_traction_at_interface():
    """Inclined-fault matrix rows must constrain the recovered tractions."""
    for alpha in (30.0, 60.0):
        params = ModelParameters(
            case_type="california",
            alpha=alpha,
            xsize=80e3,
            ysize=80e3,
            Nx=81,
            Ny=81,
            W_f=40e3,
            rho=2670.0,
            cs=3464.0,
            nu=0.25,
            x_stretch_enabled=True,
            y_stretch_enabled=True,
            x_stretch_inner_size=20e3,
            y_stretch_inner_size=20e3,
            x_stretch_inner_points=41,
            y_stretch_inner_points=41,
            x_stretch_power=2,
            y_stretch_power=2,
            allow_nonuniform_solver=True,
        )
        params.loading.V_L = 1e-9
        half_rate = 0.5e-9
        params.bc.left.ux.set_fixed()
        params.bc.left.uy.set_velocity(-half_rate)
        params.bc.right.ux.set_fixed()
        params.bc.right.uy.set_velocity(half_rate)
        params.bc.top.set_traction_free()
        params.bc.bottom.ux.set_fixed()
        params.bc.bottom.uy.set_velocity(half_rate)

        grid = Grid(params)
        builder = MatrixBuilder(params, grid)
        slip_rate = 1e-9 * (1.0 + 0.3 * np.sin(2.0 * np.pi * grid.y / params.W_f))
        slip_rate[grid.y >= params.W_f] = params.loading.V_L
        matrix = builder.build_LH()
        rhs = builder.build_RH(0.0, slip_rate)
        solution = spsolve(matrix, rhs)

        vpx = np.reshape(solution[0::2], (params.Nx + 1, params.Ny + 1)).T
        vpy = np.reshape(solution[1::2], (params.Nx + 1, params.Ny + 1)).T
        vx = vpx[:, :params.Nx]
        vy = vpy[:params.Ny, :]
        tau, sigma = StressCalUtil(prefer_numba=False).compute_stress_fields(
            vy, vx, grid.dx, grid.dy, params.lam, params.G,
            grid.cosa, grid.sina, params.Ny, params.Nx,
            x=grid.x, y=grid.y, xp=grid.xp, yp=grid.yp,
        )

        mid = params.Nx // 2
        residual = matrix @ solution - rhs
        interface_rows = []
        for iy in range(params.Ny - 1):
            _, shear_row = builder._dofs(mid + 1, iy, params.Ny)
            interface_rows.append(shear_row)
        for iy in range(1, params.Ny):
            normal_row, _ = builder._dofs(mid, iy, params.Ny)
            interface_rows.append(normal_row)

        # A small algebraic residual only proves that the rows assembled into
        # the matrix were solved.  The physical invariant is that those rows
        # impose continuity on the very same discrete tractions subsequently
        # used by the friction law.
        assert np.max(np.abs(residual[interface_rows])) < 1e-18

        shear_left = tau[:, mid - 1]
        shear_right = tau[:, mid + 1]
        np.testing.assert_allclose(
            shear_left[:-1], shear_right[:-1], rtol=1e-8, atol=1e-11
        )

        recovered_left, recovered_right = StressCalUtil(
            prefer_numba=False
        ).recover_fault_normal_stress(
            sigma,
            grid.x,
            grid.y,
            grid.xp,
            grid.yp,
            left_column=mid - 1,
            right_column=mid,
        )
        np.testing.assert_allclose(
            recovered_left, recovered_right, rtol=1e-8, atol=1e-11
        )
