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

def test_pure_shear():

    """
    Benchmark 5:
    pure shear test

    ux = gamma * y
    uy = 0

    Expected:
        tauqs   = constant
        sigmaqs = 0
    """

    # --------------------------------------------------
    # 1. Build model/grid
    # --------------------------------------------------

    params = ModelParameters(
        Nx=51,
        Ny=51,
        alpha=90.0
    )

    grid = Grid(params)

    gamma = 1e-6

    # --------------------------------------------------
    # 2. Coordinates
    # --------------------------------------------------

    Yux = grid.Yux
    Xuy = grid.Xuy

    # --------------------------------------------------
    # 3. Displacement field
    # --------------------------------------------------

    # ux shape = (Ny+1, Nx)
    ux = gamma * Yux

    # uy shape = (Ny, Nx+1)
    uy = np.zeros_like(Xuy)

    # --------------------------------------------------
    # 4. Compute stresses
    # --------------------------------------------------

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

    # --------------------------------------------------
    # 5. Diagnostics
    # --------------------------------------------------

    tau_std  = np.std(tauqs)

    max_sigma = np.max(np.abs(sigmaqs))

    tau_tol = 1e-10
    sigma_tol = 1e-10

    assert tau_std < tau_tol
    assert max_sigma < sigma_tol