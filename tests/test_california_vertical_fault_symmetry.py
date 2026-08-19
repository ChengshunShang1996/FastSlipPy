import numpy as np
from scipy.sparse.linalg import spsolve

from fastslippy.pre_processing.grid import Grid
from fastslippy.pre_processing.model_parameters import ModelParameters
from fastslippy.solver.matrix_builder import MatrixBuilder


def test_vertical_fault_far_field_loading_is_mirror_symmetric():
    """A vertical, homogeneous BP3 setup must retain left/right symmetry."""
    params = ModelParameters(
        case_type="california",
        alpha=90.0,
        Nx=41,
        Ny=41,
        xsize=80_000.0,
        ysize=80_000.0,
        W_f=1e99,
    )
    half_rate = 0.5e-9
    params.bc.top.set_traction_free()
    params.bc.bottom.ux.set_fixed()
    params.bc.bottom.uy.set_velocity(half_rate)

    grid = Grid(params)
    builder = MatrixBuilder(params, grid)
    rhs = builder.build_RH(0.0, np.zeros(params.Ny))
    solution = spsolve(builder.build_LH().tocsc(), rhs)

    ux = solution[0::2].reshape(params.Nx + 1, params.Ny + 1).T[:, :params.Nx]
    uy = solution[1::2].reshape(params.Nx + 1, params.Ny + 1).T[:params.Ny, :]
    mid = params.Nx // 2

    # uy has two fault-face nodes (mid and mid+1); omit them when mirroring.
    np.testing.assert_allclose(uy[:, :mid], -uy[:, mid + 2:][:, ::-1], atol=1e-18)
    np.testing.assert_allclose(ux[:, :mid], ux[:, mid + 1:][:, ::-1], atol=1e-18)
