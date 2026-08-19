#/////////////////////////////////////////////////
__author__      = "Chengshun Shang (Utrecht University)"
__copyright__   = "Copyright (C) 2026-present by Chengshun Shang"
__version__     = "0.0.1"
__maintainer__  = "Chengshun Shang"
__email__       = "c.shang@uu.nl"
__status__      = "development"
__date__        = "May 22, 2026"
__license__     = "MIT License"
#/////////////////////////////////////////////////

import numpy as np

from fastslippy.pre_processing.model_parameters import ModelParameters
from fastslippy.pre_processing.grid import Grid
from fastslippy.solver.matrix_builder import MatrixBuilder
from fastslippy.utilities.stress_cal_util import StressCalUtil

def test_rigid_rotation():

    """
    Benchmark 2:
    rigid body rotation

    ux = -omega * y
    uy =  omega * x

    Expected:
        tauqs   = 0
        sigmaqs = 0
    """

    # --------------------------------------------------
    # 1. Build model/grid
    # --------------------------------------------------

    params = ModelParameters(
        alpha = 90.0,
        Nx=51,
        Ny=51
    )

    grid = Grid(params)

    omega = 1e-6

    # ux nodes: shape (Ny+1, Nx)
    Xux = grid.Xux
    Yux = grid.Yux

    # uy nodes: shape (Ny, Nx+1)
    Xuy = grid.Xuy
    Yuy = grid.Yuy

    ux = -omega * Yux
    uy =  omega * Xuy

    tauqs, sigmaqs = StressCalUtil().compute_stress_fields(
        uy=uy,
        ux=ux,
        dx=grid.dx,
        dy=grid.dy,
        lam=params.lam,
        G=params.G,
        cosa=grid.cosa,
        sina=grid.sina,
        Ny=params.Ny,
        Nx=params.Nx
    )

    max_tau = np.max(np.abs(tauqs))
    max_sigma = np.max(np.abs(sigmaqs))
    tol = 1e-10

    assert max_tau < tol
    assert max_sigma < tol


def test_bottom_traction_free_annuls_rigid_rotation_for_inclined_faults():
    """The inclined free-surface rows must not react to a rigid rotation."""
    omega = 1e-6

    for alpha in (90.0, 60.0, 45.0, 30.0):
        params = ModelParameters(Nx=51, Ny=51, alpha=alpha)
        params.bc.top.ux.set_traction_free()
        params.bc.top.uy.set_traction_free()
        grid = Grid(params)
        builder = MatrixBuilder(params, grid)

        # Physical rigid rotation U_X=-omega*Z, U_Z=omega*X expressed in
        # U = ux*(1, 0) + uy*(cos(alpha), sin(alpha)).
        ux = -omega * grid.Yux - grid.cosa * omega * grid.Xux / grid.sina
        uy = omega * grid.Xuy / grid.sina

        U = np.zeros(grid.N)
        for ix in range(params.Nx + 1):
            for iy in range(params.Ny + 1):
                kux, kuy = builder._dofs(ix, iy, params.Ny)
                if ix < params.Nx:
                    U[kux] = ux[iy, ix]
                if iy < params.Ny:
                    U[kuy] = uy[iy, ix]

        residual = builder.build_LH() @ U
        rows = []
        for ix in range(1, params.Nx):
            _, kuy = builder._dofs(ix, 0, params.Ny)
            rows.append(kuy)
        for ix in range(1, params.Nx - 1):
            kux, _ = builder._dofs(ix, 0, params.Ny)
            rows.append(kux)

        assert np.max(np.abs(residual[rows])) < 1e-18
