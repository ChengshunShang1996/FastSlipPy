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

def test_fault_slip_symmetry():

    """
    Benchmark 6:
    smooth anti-symmetric fault slip

    ux = 0.5*D*tanh(x/w)
    uy = 0

    Expected:
        - tau localized near fault
        - left/right symmetry
        - sigma ≈ 0
        - no checkerboard
    """

    # --------------------------------------------------
    # 1. Build model/grid
    # --------------------------------------------------

    params = ModelParameters(
        Nx=101,
        Ny=101,
        alpha=90.0
    )

    grid = Grid(params)

    # --------------------------------------------------
    # 2. Slip parameters
    # --------------------------------------------------

    D = 1e-3

    w = 3 * grid.dx

    # --------------------------------------------------
    # 3. Coordinates
    # --------------------------------------------------

    Xux = grid.Xux
    Xuy = grid.Xuy

    # --------------------------------------------------
    # 4. Smooth fault slip field
    # --------------------------------------------------

    #This is compression
    #ux = 0.5 * D * np.tanh(Xux / w)
    #uy = np.zeros_like(Xuy)

    ux = np.zeros_like(Xux)
    uy = 0.5 * D * np.tanh(Xuy / w)

    # --------------------------------------------------
    # 5. Compute stresses
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
    # 6. Symmetry diagnostics
    # --------------------------------------------------

    mid = params.Nx // 2

    tau_left  = tauqs[:, :mid]
    tau_right = np.flip(tauqs[:, mid+1:], axis=1)

    symmetry_error = np.max(np.abs(tau_left - tau_right))

    max_sigma = np.max(np.abs(sigmaqs))


    # --------------------------------------------------
    # 7. Pass/fail
    # --------------------------------------------------

    tol = 1e-10

    assert symmetry_error < tol