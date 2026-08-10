#/////////////////////////////////////////////////
__author__      = "Chengshun Shang (Utrecht University)"
__copyright__   = "Copyright (C) 2026-present by Chengshun Shang"
__version__     = "0.0.1"
__maintainer__  = "Chengshun Shang"
__email__       = "c.shang@uu.nl"
__status__      = "development"
__date__        = "Aug 9, 2026"
__license__     = "MIT License"
#/////////////////////////////////////////////////

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import spsolve

from fastslippy.pre_processing.grid import Grid
from fastslippy.pre_processing.model_parameters import ModelParameters
from fastslippy.solver.matrix_builder import MatrixBuilder


def _solve_displacement_fields(params: ModelParameters):
    params.bc.left.ux.set_fixed()
    params.bc.left.uy.set_fixed()
    params.bc.right.ux.set_fixed()
    params.bc.right.uy.set_velocity(1e-5)
    params.bc.bottom.ux.set_fixed()
    params.bc.bottom.uy.set_velocity(1e-5)
    params.bc.top.ux.set_fixed()
    params.bc.top.uy.set_velocity(1e-5)

    grid = Grid(params)
    builder = MatrixBuilder(params, grid)
    lh = builder.build_LH()
    v_fault = np.full(params.Ny, params.Vi, dtype=float)
    rh = builder.build_RH(dPdt=0.0, V=v_fault)
    sol = spsolve(lh, rh)

    vpx = np.reshape(sol[0::2], (params.Nx + 1, params.Ny + 1), order="C").T
    vpy = np.reshape(sol[1::2], (params.Nx + 1, params.Ny + 1), order="C").T
    return grid, vpx, vpy


def _interp_to_uniform_grid(src_coords, field, dst_coords):
    src_y, src_x = src_coords
    dst_y, dst_x = dst_coords
    interpolator = RegularGridInterpolator((src_y, src_x), field, bounds_error=False, fill_value=None)
    yy, xx = np.meshgrid(dst_y, dst_x, indexing="ij")
    points = np.column_stack([yy.ravel(), xx.ravel()])
    return interpolator(points).reshape(yy.shape)


def _relative_l2_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(reference)), 1e-30)
    return float(np.linalg.norm(candidate - reference) / denom)


def test_stretched_mesh_converges_toward_uniform_solution_when_stretch_is_relaxed():
    base_kwargs = dict(
        case_type="lab",
        xsize=1.0,
        ysize=1.0,
        Nx=41,
        Ny=41,
        Vi=1e-40,
    )

    uniform_grid, uniform_vpx, uniform_vpy = _solve_displacement_fields(
        ModelParameters(**base_kwargs)
    )
    uniform_vpx = uniform_vpx[:, : base_kwargs["Nx"]]
    uniform_vpy = uniform_vpy[: base_kwargs["Ny"], :]

    configs = [
        dict(
            x_stretch_enabled=True,
            y_stretch_enabled=True,
            x_stretch_inner_size=0.20,
            x_stretch_inner_points=11,
            x_stretch_power=4,
            y_stretch_inner_size=0.20,
            y_stretch_inner_points=11,
            y_stretch_power=4,
            allow_nonuniform_solver=True,
        ),
        dict(
            x_stretch_enabled=True,
            y_stretch_enabled=True,
            x_stretch_inner_size=0.45,
            x_stretch_inner_points=21,
            x_stretch_power=3,
            y_stretch_inner_size=0.45,
            y_stretch_inner_points=21,
            y_stretch_power=3,
            allow_nonuniform_solver=True,
        ),
        dict(
            x_stretch_enabled=True,
            y_stretch_enabled=True,
            x_stretch_inner_size=0.75,
            x_stretch_inner_points=31,
            x_stretch_power=2,
            y_stretch_inner_size=0.75,
            y_stretch_inner_points=31,
            y_stretch_power=2,
            allow_nonuniform_solver=True,
        ),
    ]

    errors = []
    for stretch_kwargs in configs:
        grid, vpx, vpy = _solve_displacement_fields(ModelParameters(**base_kwargs, **stretch_kwargs))
        vpx = vpx[:, : base_kwargs["Nx"]]
        vpy = vpy[: base_kwargs["Ny"], :]

        uniform_vpx_ref = uniform_vpx[1:-1, 1:-1]
        stretched_vpx = _interp_to_uniform_grid(
            (grid.yp[1:-1], grid.x[1:-1]),
            vpx[1:-1, 1:-1],
            (uniform_grid.yp[1:-1], uniform_grid.x[1:-1]),
        )

        uniform_vpy_ref = uniform_vpy[1:-1, 1:-1]
        stretched_vpy = _interp_to_uniform_grid(
            (grid.y[1:-1], grid.xp[1:-1]),
            vpy[1:-1, 1:-1],
            (uniform_grid.y[1:-1], uniform_grid.xp[1:-1]),
        )

        errors.append(
            max(
                _relative_l2_error(uniform_vpx_ref, stretched_vpx),
                _relative_l2_error(uniform_vpy_ref, stretched_vpy),
            )
        )

    assert errors[2] < errors[1] < errors[0]
