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


def test_top_traction_free_annuls_rigid_rotation_for_inclined_faults():
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


def test_bottom_traction_free_annuls_rigid_rotation_for_inclined_faults():
    """The deep zero-traction rows are the bottom mirror of the free surface."""
    omega = 1e-6

    for alpha in (90.0, 60.0, 45.0, 30.0):
        params = ModelParameters(
            case_type="california",
            Nx=31,
            Ny=29,
            xsize=80e3,
            ysize=60e3,
            alpha=alpha,
            x_stretch_enabled=True,
            y_stretch_enabled=True,
            x_stretch_inner_size=20e3,
            y_stretch_inner_size=20e3,
            x_stretch_inner_points=15,
            y_stretch_inner_points=11,
            allow_nonuniform_solver=True,
        )
        params.bc.top.set_traction_free()
        params.bc.bottom.set_traction_free()
        grid = Grid(params)
        builder = MatrixBuilder(params, grid)

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
        mid = params.Nx // 2
        rows = []
        for ix in range(1, params.Nx):
            if ix != mid:
                _, kuy = builder._dofs(ix, params.Ny - 1, params.Ny)
                rows.append(kuy)
        for ix in range(params.Nx):
            kux, _ = builder._dofs(ix, params.Ny, params.Ny)
            rows.append(kux)

        assert np.max(np.abs(residual[rows])) < 1e-14


def test_bottom_traction_free_accepts_nonzero_affine_free_traction():
    """Both bottom traction components vanish for a strained affine field."""
    for case_type, alpha in (
        ("california", 90.0),
        ("california", 60.0),
        ("california", 30.0),
        ("groningen", 60.0),
    ):
        params = ModelParameters(
            case_type=case_type,
            Nx=31,
            Ny=29,
            xsize=80e3,
            ysize=60e3,
            alpha=alpha,
        )
        params.bc.top.set_traction_free()
        params.bc.bottom.set_traction_free()
        grid = Grid(params)
        builder = MatrixBuilder(params, grid)

        exx = 2e-6
        ux_z = 3e-6
        uz_x = -ux_z
        uzz = -params.lam * exx / (params.lam + 2.0 * params.G)
        physical_z_ux = uz_x * grid.Xux + uzz * grid.Yux
        physical_z_uy = uz_x * grid.Xuy + uzz * grid.Yuy
        ux = (
            exx * grid.Xux
            + ux_z * grid.Yux
            - grid.cosa * physical_z_ux / grid.sina
        )
        uy = physical_z_uy / grid.sina

        U = np.zeros(grid.N)
        for ix in range(params.Nx + 1):
            for iy in range(params.Ny + 1):
                kux, kuy = builder._dofs(ix, iy, params.Ny)
                if ix < params.Nx:
                    U[kux] = ux[iy, ix]
                if iy < params.Ny:
                    U[kuy] = uy[iy, ix]

        residual = builder.build_LH() @ U
        mid = params.Nx // 2
        rows = []
        for ix in range(1, params.Nx):
            if case_type != "california" or ix != mid:
                _, kuy = builder._dofs(ix, params.Ny - 1, params.Ny)
                rows.append(kuy)
        for ix in range(params.Nx):
            kux, _ = builder._dofs(ix, params.Ny, params.Ny)
            rows.append(kux)

        assert np.max(np.abs(residual[rows])) < 1e-14
