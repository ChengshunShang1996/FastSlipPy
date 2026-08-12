import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import spsolve

from fastslippy.pre_processing.model_parameters import ModelParameters
from fastslippy.pre_processing.grid import Grid
from fastslippy.solver.matrix_builder import MatrixBuilder
from fastslippy.utilities.stress_cal_util import StressCalUtil


def first_derivative_nonuniform(arr, coords, axis):
    arr = np.asarray(arr)
    coords = np.asarray(coords, dtype=float)
    n = coords.size
    arr_moved = np.moveaxis(arr, axis, -1)
    out = np.empty_like(arr_moved, dtype=float)
    # endpoints
    h_r = coords[1] - coords[0]
    out[..., 0] = (arr_moved[..., 1] - arr_moved[..., 0]) / h_r
    h_l = coords[-1] - coords[-2]
    out[..., -1] = (arr_moved[..., -1] - arr_moved[..., -2]) / h_l
    for j in range(1, n - 1):
        h_l = coords[j] - coords[j - 1]
        h_r = coords[j + 1] - coords[j]
        df_b = (arr_moved[..., j] - arr_moved[..., j - 1]) / h_l
        df_f = (arr_moved[..., j + 1] - arr_moved[..., j]) / h_r
        out[..., j] = (h_r * df_b + h_l * df_f) / (h_l + h_r)
    return np.moveaxis(out, -1, axis)


def interp_to_uniform(src_coords, field, dst_y, dst_x):
    src_y, src_x = src_coords
    interpolator = RegularGridInterpolator((src_y, src_x), field, bounds_error=False, fill_value=None)
    yy, xx = np.meshgrid(dst_y, dst_x, indexing='ij')
    pts = np.column_stack([yy.ravel(), xx.ravel()])
    out = interpolator(pts).reshape(yy.shape)
    return out


def relative_l2(a, b):
    den = max(float(np.linalg.norm(a)), 1e-30)
    return float(np.linalg.norm(a - b) / den)


base_kwargs = dict(
    case_type='lab',
    xsize=1.0,
    ysize=1.0,
    Nx=41,
    Ny=41,
    Vi=1e-40,
    linear_solver='direct'
)

# reference
p_uniform = ModelParameters(**base_kwargs)
# uniform solution from previous script or recompute
print('Building uniform reference...')
G = p_uniform.G
sina = np.sin(np.deg2rad(p_uniform.alpha))
cosa = np.cos(np.deg2rad(p_uniform.alpha))

# helper to build solution
def build_solution(p):
    grid = Grid(p)
    builder = MatrixBuilder(p, grid)
    LH = builder.build_LH()
    v_fault = np.full(p.Ny, p.Vi if p.Vi>0 else 1e-12, dtype=float)
    RH = builder.build_RH(dPdt=0.0, V=v_fault)
    sol = spsolve(LH, RH)
    vpx = np.reshape(sol[0::2], (p.Nx + 1, p.Ny + 1), order='C').T
    vpy = np.reshape(sol[1::2], (p.Nx + 1, p.Ny + 1), order='C').T
    vpx = vpx[:, :p.Nx]
    vpy = vpy[:p.Ny, :]
    return grid, vpx, vpy

ug, u_vpx, u_vpy = build_solution(p_uniform)
print('Uniform built')

# x+y stretch case
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

grid_xy, vpx_xy, vpy_xy = build_solution(p_xy)
print('XY built')

# compute StressCalUtil baseline tau (current implementation)
scu = StressCalUtil(prefer_numba=False)
# use nonuniform path
tau_xy, sigma_xy = scu.compute_stress_fields(vpy_xy, vpx_xy, grid_xy.dx, grid_xy.dy,
                                              p_xy.lam, p_xy.G, grid_xy.cosa, grid_xy.sina,
                                              p_xy.Ny, p_xy.Nx,
                                              x=grid_xy.x, y=grid_xy.y, xp=grid_xy.xp, yp=grid_xy.yp)

# Now recompute intermediates manually to extract mm_duxdx and mm_duydy and local sx
Nx, Ny = p_xy.Nx, p_xy.Ny
# coords
x = grid_xy.x
xp = grid_xy.xp
y = grid_xy.y
yp = grid_xy.yp

# term1, term2
dx_uy = np.diff(xp)
dy_ux = np.diff(yp)
dx_ux = np.diff(x)
dy_uy = np.diff(y)
term1 = np.diff(vpy_xy, axis=1) / dx_uy[None, :]
term2 = (1 - 2 * (grid_xy.cosa ** 2)) * np.diff(vpx_xy, axis=0) / dy_ux[:, None]

# duxdx and mm_duxdx
duxdx = first_derivative_nonuniform(vpx_xy, x, axis=1)   # (Ny+1, Nx)
mm_duxdx = 0.5 * (duxdx[:-1, :] + duxdx[1:, :])            # (Ny, Nx)

# duydy and mm_duydy
duydy = first_derivative_nonuniform(vpy_xy, y, axis=0)    # (Ny, Nx+1)
mm_duydy = 0.5 * (duydy[:, :-1] + duydy[:, 1:])           # (Ny, Nx)

# baseline tau (should equal tau_xy above)
tau_manual = p_xy.G / grid_xy.sina * (term1 + term2 + grid_xy.cosa * (mm_duxdx - mm_duydy))

# get local sx/dx_loc from MatrixBuilder._second_derivative_weights
mb = MatrixBuilder(p_xy, grid_xy)
# for ux columns j in 0..Nx-1 use g.x coords
sx_x = np.empty(Nx)
for j in range(Nx):
    # _second_derivative_weights expects 1 <= idx <= coords.size-2 for interior
    # For endpoints use one-sided spacing as an approximation consistent with _local_step
    coords = grid_xy.x
    if j <= 0:
        sx_x[j] = coords[1] - coords[0]
    elif j >= coords.size - 1:
        sx_x[j] = coords[-1] - coords[-2]
    else:
        _, _, _, hmsq = mb._second_derivative_weights(coords, j)
        sx_x[j] = np.sqrt(hmsq)
# for y-based second derivative used for duydy we use grid_xy.y and indices 0..Ny-1
sx_y = np.empty(Ny)
for i in range(Ny):
    coords = grid_xy.y
    if i <= 0:
        sx_y[i] = coords[1] - coords[0]
    elif i >= coords.size - 1:
        sx_y[i] = coords[-1] - coords[-2]
    else:
        _, _, _, hmsq = mb._second_derivative_weights(coords, i)
        sx_y[i] = np.sqrt(hmsq)

# Try candidate scalings
candidates = []
# no scaling
candidates.append(('none', 1.0, 1.0))
# multiply mm_duxdx by dx_loc
candidates.append(('mm_dx', 'mul_dx', 'none'))
# multiply mm_duydy by dy_loc
candidates.append(('mm_dy', 'none', 'mul_dy'))
# multiply both by respective loc
candidates.append(('both_mul', 'mul_dx', 'mul_dy'))
# divide both
candidates.append(('both_div', 'div_dx', 'div_dy'))
# multiply both by sqrt
candidates.append(('both_sqrt', 'sqrt', 'sqrt'))

results = {}
for name, op_x, op_y in candidates:
    # apply operations
    mmx = mm_duxdx.copy()
    mmy = mm_duydy.copy()
    if op_x == 'mul_dx':
        mmx = mmx * sx_x[None, :]
    elif op_x == 'div_dx':
        mmx = mmx / sx_x[None, :]
    elif op_x == 'sqrt':
        mmx = mmx * np.sqrt(sx_x)[None, :]
    if op_y == 'mul_dy':
        mmy = mmy * sx_y[:, None]
    elif op_y == 'div_dy':
        mmy = mmy / sx_y[:, None]
    elif op_y == 'sqrt':
        mmy = mmy * np.sqrt(sx_y)[:, None]

    tau_cand = p_xy.G / grid_xy.sina * (term1 + term2 + grid_xy.cosa * (mmx - mmy))
    # interpolate to uniform reference grid
    tgt_y = ug.y[1:-1]
    tgt_x = ug.x[1:-1]
    tau_cand_interp = interp_to_uniform((grid_xy.y[1:-1], grid_xy.x[1:-1]), tau_cand[1:-1,1:-1], tgt_y, tgt_x)
    tau_ref = tau_manual[1:-1,1:-1]
    err = relative_l2(tau_ref, tau_cand_interp)
    results[name] = err

print('\nCandidate scaling relative L2 errors (lower is better):')
for k,v in results.items():
    print(f"{k:12s}: {v:.6e}")

print('\nAlso report max abs differences between mm_duxdx and scaled variants at central region:')
center_region = (slice(mm_duxdx.shape[0]//4, mm_duxdx.shape[0]*3//4), slice(mm_duxdx.shape[1]//4, mm_duxdx.shape[1]*3//4))
print('mm_duxdx mean abs:', np.mean(np.abs(mm_duxdx[center_region])))
print('mm_duydy mean abs:', np.mean(np.abs(mm_duydy[center_region])))
print('sx_x mean:', np.mean(sx_x))
print('sx_y mean:', np.mean(sx_y))

print('\nDiagnostic scaling script complete.')
