import numpy as np

from main import (
    ModelParameters,
    Grid,
    compute_stress_fields
)

def test_rigid_translation():

    params = ModelParameters(
        Nx=51,
        Ny=51
    )

    grid = Grid(params)

    ux = np.ones((params.Ny + 1, params.Nx)) * 1.234

    uy = np.ones((params.Ny, params.Nx + 1)) * 2.345

    tauqs, sigmaqs = compute_stress_fields(
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