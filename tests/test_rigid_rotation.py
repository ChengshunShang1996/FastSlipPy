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