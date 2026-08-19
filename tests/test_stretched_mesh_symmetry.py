import numpy as np
from scipy.sparse.linalg import spsolve

from fastslippy.pre_processing.grid import Grid
from fastslippy.pre_processing.model_parameters import ModelParameters
from fastslippy.solver.matrix_builder import MatrixBuilder


def _solve_ux(params: ModelParameters) -> np.ndarray:
    """Solve a left/right mirror-symmetric loading configuration."""
    params.bc.left.ux.set_fixed()
    params.bc.left.uy.set_velocity(-1e-5)
    params.bc.right.ux.set_fixed()
    params.bc.right.uy.set_velocity(1e-5)
    params.bc.bottom.ux.set_fixed()
    params.bc.bottom.uy.set_fixed()
    params.bc.top.ux.set_fixed()
    params.bc.top.uy.set_fixed()

    grid = Grid(params)
    builder = MatrixBuilder(params, grid)
    lh = builder.build_LH()
    v_fault = np.full(params.Ny, params.Vi, dtype=float)
    rh = builder.build_RH(dPdt=0.0, V=v_fault)
    sol = spsolve(lh, rh)
    vpx = np.reshape(sol[0::2], (params.Nx + 1, params.Ny + 1), order="C").T
    return vpx[:, :params.Nx]


def _relative_mirror_error_x(field: np.ndarray) -> float:
    mirrored = field[:, ::-1]
    denom = max(float(np.max(np.abs(field))), 1e-30)
    return float(np.max(np.abs(field - mirrored)) / denom)


def test_stretched_mesh_preserves_left_right_symmetry():
    params = ModelParameters(
        case_type="lab",
        xsize=1.0,
        ysize=1.0,
        Nx=41,
        Ny=41,
        Vi=1e-40,
        x_stretch_enabled=True,
        y_stretch_enabled=False,
        x_stretch_inner_size=0.5,
        x_stretch_inner_points=31,
        x_stretch_power=2,
        allow_nonuniform_solver=True,
    )
    ux = _solve_ux(params)
    assert _relative_mirror_error_x(ux) < 1e-10
