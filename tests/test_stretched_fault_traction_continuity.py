import numpy as np
from scipy.sparse.linalg import spsolve

from fastslippy.pre_processing.grid import Grid
from fastslippy.pre_processing.model_parameters import ModelParameters
from fastslippy.solver.matrix_builder import MatrixBuilder
from fastslippy.utilities.stress_cal_util import StressCalUtil


def test_stretched_fault_has_continuous_tractions_for_all_bp3_dips():
    """The matrix interface rows and stress recovery must use one traction stencil."""
    for alpha in (30.0, 60.0, 90.0):
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
        solution = spsolve(builder.build_LH(), builder.build_RH(0.0, slip_rate))

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
        # Exclude the one-sided y stencils at the two external boundaries.
        assert np.max(np.abs(tau[3:-3, mid - 1] - tau[3:-3, mid + 1])) < 1e-10
        assert np.max(np.abs(sigma[3:-3, mid - 1] - sigma[3:-3, mid])) < 1e-10
