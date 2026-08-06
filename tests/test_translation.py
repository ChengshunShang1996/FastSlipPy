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

def test_rigid_translation():

    params = ModelParameters(
        Nx=51,
        Ny=51
    )

    grid = Grid(params)

    ux = np.ones((params.Ny + 1, params.Nx)) * 1.234

    uy = np.ones((params.Ny, params.Nx + 1)) * 2.345

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

    assert np.max(np.abs(tauqs)) < 1e-12

    assert np.max(np.abs(sigmaqs)) < 1e-12


def test_rigid_translation_nonuniform_grid():

    params = ModelParameters(
        xsize=1200.0,
        ysize=1000.0,
        Nx=51,
        Ny=51,
        x_stretch_enabled=True,
        x_stretch_inner_size=300.0,
        x_stretch_inner_points=21,
        y_stretch_enabled=True,
        y_stretch_inner_size=200.0,
        y_stretch_inner_points=13,
    )

    grid = Grid(params)

    ux = np.ones((params.Ny + 1, params.Nx)) * 1.234
    uy = np.ones((params.Ny, params.Nx + 1)) * 2.345

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
        Nx=params.Nx,
        x=grid.x,
        y=grid.y,
        xp=grid.xp,
        yp=grid.yp,
    )

    assert np.max(np.abs(tauqs)) < 1e-12
    assert np.max(np.abs(sigmaqs)) < 1e-12