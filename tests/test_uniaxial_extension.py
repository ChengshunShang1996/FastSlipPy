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

def test_uniaxial_extension():

    """
    Benchmark 3:
    uniaxial extension

    ux = a * x
    uy = 0

    Expected:
        tauqs ≈ 0
        sigmaqs = constant
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

    # prescribed strain
    a = 1e-6

    # --------------------------------------------------
    # 2. Coordinates
    # --------------------------------------------------

    Xux = grid.Xux
    Xuy = grid.Xuy

    # --------------------------------------------------
    # 3. Displacement field
    # --------------------------------------------------

    # ux shape = (Ny+1, Nx)
    ux = a * Xux

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

    max_tau = np.max(np.abs(tauqs))

    sigma_mean = np.mean(sigmaqs)
    sigma_std = np.std(sigmaqs)

    tau_tol = 1e-10
    sigma_tol = 1e-10

    assert max_tau < tau_tol
    assert sigma_std < sigma_tol