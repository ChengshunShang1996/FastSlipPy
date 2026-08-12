import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RegularGridInterpolator
from scipy import sparse

from fastslippy.pre_processing.model_parameters import ModelParameters
from fastslippy.pre_processing.grid import Grid
from fastslippy.solver.matrix_builder import MatrixBuilder
from fastslippy.utilities.stress_cal_util import StressCalUtil


def solve_and_analyze(params: ModelParameters):
    # Build
    grid = Grid(params)
    builder = MatrixBuilder(params, grid)
    LH = builder.build_LH()
    # symmetry norm
    LH_dense = LH.todense()
    sym_norm = float(np.max(np.abs(LH_dense - LH_dense.T)))
    # condition number (dense)
    try:
        cond = float(np.linalg.cond(LH_dense))
    except Exception:
        cond = float('inf')

    # Build RH with small fault velocity
    v_fault = np.full(params.Ny, params.Vi if params.Vi>0 else 1e-12, dtype=float)
    RH = builder.build_RH(dPdt=0.0, V=v_fault)
    # Solve
    sol = spsolve(LH, RH)
    # reshape
    Nx, Ny = params.Nx, params.Ny
    vpx = np.reshape(sol[0::2], (params.Nx + 1, params.Ny + 1), order="C").T
    vpy = np.reshape(sol[1::2], (params.Nx + 1, params.Ny + 1), order="C").T
    vpx = vpx[:, :Nx]
    vpy = vpy[:Ny, :]

    # compute stress fields using StressCalUtil (nonuniform path if needed)
    scu = StressCalUtil(prefer_numba=False)
    if grid.is_nonuniform:
        tauqs, sigmaqs = scu.compute_stress_fields(
            vpy, vpx, grid.dx, grid.dy,
            params.lam, params.G, grid.cosa, grid.sina, Ny, Nx,
            x=grid.x, y=grid.y, xp=grid.xp, yp=grid.yp)
    else:
        tauqs, sigmaqs = scu.compute_stress_fields(
            vpy, vpx, grid.dx, grid.dy,
            params.lam, params.G, grid.cosa, grid.sina, Ny, Nx)

    return {
        'grid': grid,
        'LH': LH,
        'sym_norm': sym_norm,
        'cond': cond,
        'vpx': vpx,
        'vpy': vpy,
        'tauqs': tauqs,
        'sigmaqs': sigmaqs,
    }


def interp_to_uniform(src_grid, field, dst_y, dst_x):
    # src_grid: tuple (y_coords, x_coords)
    src_y, src_x = src_grid
    interpolator = RegularGridInterpolator((src_y, src_x), field, bounds_error=False, fill_value=None)
    yy, xx = np.meshgrid(dst_y, dst_x, indexing='ij')
    pts = np.column_stack([yy.ravel(), xx.ravel()])
    out = interpolator(pts).reshape(yy.shape)
    return out


def relative_l2(a, b):
    den = max(float(np.linalg.norm(a)), 1e-30)
    return float(np.linalg.norm(a - b) / den)


if __name__ == '__main__':
    base_kwargs = dict(
        case_type='lab',
        xsize=1.0,
        ysize=1.0,
        Nx=41,
        Ny=41,
        Vi=1e-40,
        # use direct solver for determinism
        linear_solver='direct'
    )

    # uniform
    p_uniform = ModelParameters(**base_kwargs)
    print('Running uniform...')
    u = solve_and_analyze(p_uniform)

    # x-only stretch
    p_x_only = ModelParameters(**base_kwargs,
                               x_stretch_enabled=True,
                               y_stretch_enabled=False,
                               x_stretch_inner_size=0.45,
                               x_stretch_inner_points=21,
                               x_stretch_power=3,
                               allow_nonuniform_solver=True)
    print('Running x-only stretch...')
    xonly = solve_and_analyze(p_x_only)

    # x+y stretch
    p_xy = ModelParameters(**base_kwargs,
                           x_stretch_enabled=True,
                           y_stretch_enabled=True,
                           x_stretch_inner_size=0.45,
                           x_stretch_inner_points=21,
                           x_stretch_power=3,
                           y_stretch_inner_size=0.45,
                           y_stretch_inner_points=21,
                           y_stretch_power=3,
                           allow_nonuniform_solver=True)
    print('Running xy stretch...')
    xy = solve_and_analyze(p_xy)

    # compare metrics
    print('\n=== Matrix symmetry norms ===')
    print(f"uniform sym_norm = {u['sym_norm']:.3e}")
    print(f"x-only sym_norm = {xonly['sym_norm']:.3e}")
    print(f"xy sym_norm = {xy['sym_norm']:.3e}")

    print('\n=== Condition numbers (dense cond) ===')
    print(f"uniform cond = {u['cond']:.3e}")
    print(f"x-only cond = {xonly['cond']:.3e}")
    print(f"xy cond = {xy['cond']:.3e}")

    # interpolate fields to uniform grid for comparison
    # target coords from uniform grid
    ug = u['grid']
    tgt_y_vpy = ug.y[1:-1]
    tgt_x_vpy = ug.xp[1:-1]
    tgt_y_vpx = ug.yp[1:-1]
    tgt_x_vpx = ug.x[1:-1]

    # vpx: source grids
    vpx_xonly_interp = interp_to_uniform((xonly['grid'].yp[1:-1], xonly['grid'].x[1:-1]), xonly['vpx'][1:-1,1:-1], tgt_y_vpx, tgt_x_vpx)
    vpx_xy_interp    = interp_to_uniform((xy['grid'].yp[1:-1], xy['grid'].x[1:-1]), xy['vpx'][1:-1,1:-1], tgt_y_vpx, tgt_x_vpx)
    vpx_uniform_ref  = u['vpx'][1:-1,1:-1]

    vpy_xonly_interp = interp_to_uniform((xonly['grid'].y[1:-1], xonly['grid'].xp[1:-1]), xonly['vpy'][1:-1,1:-1], tgt_y_vpy, tgt_x_vpy)
    vpy_xy_interp    = interp_to_uniform((xy['grid'].y[1:-1], xy['grid'].xp[1:-1]), xy['vpy'][1:-1,1:-1], tgt_y_vpy, tgt_x_vpy)
    vpy_uniform_ref  = u['vpy'][1:-1,1:-1]

    print('\n=== Relative L2 errors vs uniform reference (velocity fields) ===')
    print(f"x-only vpx rel L2 = {relative_l2(vpx_uniform_ref, vpx_xonly_interp):.3e}")
    print(f"xy    vpx rel L2 = {relative_l2(vpx_uniform_ref, vpx_xy_interp):.3e}")
    print(f"x-only vpy rel L2 = {relative_l2(vpy_uniform_ref, vpy_xonly_interp):.3e}")
    print(f"xy    vpy rel L2 = {relative_l2(vpy_uniform_ref, vpy_xy_interp):.3e}")

    # tauqs
    tau_xonly_interp = interp_to_uniform((xonly['grid'].y[1:-1], xonly['grid'].x[1:-1]), xonly['tauqs'][1:-1,1:-1], ug.y[1:-1], ug.x[1:-1])
    tau_xy_interp    = interp_to_uniform((xy['grid'].y[1:-1], xy['grid'].x[1:-1]), xy['tauqs'][1:-1,1:-1], ug.y[1:-1], ug.x[1:-1])
    tau_uniform_ref  = u['tauqs'][1:-1,1:-1]

    print('\n=== Relative L2 errors vs uniform reference (tauqs) ===')
    print(f"x-only tau rel L2 = {relative_l2(tau_uniform_ref, tau_xonly_interp):.3e}")
    print(f"xy    tau rel L2 = {relative_l2(tau_uniform_ref, tau_xy_interp):.3e}")

    # show max absolute differences in metric arrays if present
    print('\n=== Grid spacings (dx edges / dy edges) max ratios ===')
    print(f"uniform dx max/min = {np.max(np.diff(ug.x))/np.min(np.diff(ug.x)):.3f}")
    print(f"x-only dx max/min = {np.max(np.diff(xonly['grid'].x))/np.min(np.diff(xonly['grid'].x)):.3f}")
    print(f"xy dx max/min = {np.max(np.diff(xy['grid'].x))/np.min(np.diff(xy['grid'].x)):.3f}")
    print(f"uniform dy max/min = {np.max(np.diff(ug.y))/np.min(np.diff(ug.y)):.3f}")
    print(f"x-only dy max/min = {np.max(np.diff(xonly['grid'].y))/np.min(np.diff(xonly['grid'].y)):.3f}")
    print(f"xy dy max/min = {np.max(np.diff(xy['grid'].y))/np.min(np.diff(xy['grid'].y)):.3f}")

    print('\nDiagnostic complete.')
