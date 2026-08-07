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

def test_constant_strain_interior_equilibrium():

    """
    Benchmark 4:
    Constant strain interior equilibrium test

    ux = a * x
    uy = 0

    Since stress is constant:

        div(sigma) = 0

    therefore:

        LH @ U = 0

    should hold to machine precision.
    """

    # --------------------------------------------------
    # 1. Build model/grid/matrix
    # --------------------------------------------------

    params = ModelParameters(
        Nx=51,
        Ny=51,
        alpha=90.0
    )

    grid = Grid(params)

    builder = MatrixBuilder(params, grid)

    LH = builder.build_LH()

    # --------------------------------------------------
    # 2. Prescribed displacement field
    # --------------------------------------------------

    a = 1e-6

    # staggered fields
    ux = a * grid.Xux
    uy = np.zeros_like(grid.Xuy)

    # --------------------------------------------------
    # 3. Pack into global vector U
    # --------------------------------------------------

    U = np.zeros(grid.N)

    for ix in range(params.Nx + 1):
        for iy in range(params.Ny + 1):

            kux, kuy = builder._dofs(ix, iy, params.Ny)

            # ux nodes exist for ix < Nx
            if ix < params.Nx:
                U[kux] = ux[iy, ix]

            # uy nodes exist for iy < Ny
            if iy < params.Ny:
                U[kuy] = uy[iy, ix]

    # --------------------------------------------------
    # 4. RHS = 0
    # --------------------------------------------------

    RH = np.zeros(grid.N)

    # --------------------------------------------------
    # 5. Residual
    # --------------------------------------------------

    residual = LH @ U

    interior_dofs = []

    for ix in range(1, params.Nx-1):
        for iy in range(1, params.Ny-1):

            kux, kuy = builder._dofs(ix, iy, params.Ny)

            # ux equation exists for ix < Nx
            if ix < params.Nx:
                interior_dofs.append(kux)

            # uy equation exists for iy < Ny
            if iy < params.Ny:
                interior_dofs.append(kuy)

    interior_dofs = np.array(interior_dofs)

    max_residual = np.max(np.abs(residual[interior_dofs]))

    assert max_residual < 1e-12


def test_constant_strain_interior_equilibrium_nonuniform_mesh():

    params = ModelParameters(
        Nx=51,
        Ny=51,
        alpha=90.0,
        xsize=1200.0,
        ysize=1000.0,
        x_stretch_enabled=True,
        x_stretch_inner_size=300.0,
        x_stretch_inner_points=21,
        y_stretch_enabled=True,
        y_stretch_inner_size=200.0,
        y_stretch_inner_points=13,
        allow_nonuniform_solver=True,
    )

    grid = Grid(params)
    builder = MatrixBuilder(params, grid)
    LH = builder.build_LH()

    a = 1e-6
    ux = a * grid.Xux
    uy = np.zeros_like(grid.Xuy)

    U = np.zeros(grid.N)
    for ix in range(params.Nx + 1):
        for iy in range(params.Ny + 1):
            kux, kuy = builder._dofs(ix, iy, params.Ny)
            if ix < params.Nx:
                U[kux] = ux[iy, ix]
            if iy < params.Ny:
                U[kuy] = uy[iy, ix]

    residual = LH @ U
    interior_dofs = []
    for ix in range(1, params.Nx - 1):
        for iy in range(1, params.Ny - 1):
            kux, kuy = builder._dofs(ix, iy, params.Ny)
            if ix < params.Nx:
                interior_dofs.append(kux)
            if iy < params.Ny:
                interior_dofs.append(kuy)

    interior_dofs = np.array(interior_dofs)
    max_residual = np.max(np.abs(residual[interior_dofs]))

    assert max_residual < 1e-12