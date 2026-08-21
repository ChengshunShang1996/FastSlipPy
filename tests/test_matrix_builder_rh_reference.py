#/////////////////////////////////////////////////
__author__      = "Chengshun Shang (Utrecht University)"
__copyright__   = "Copyright (C) 2026-present by Chengshun Shang"
__version__     = "0.0.1"
__maintainer__  = "Chengshun Shang"
__email__       = "c.shang@uu.nl"
__status__      = "development"
__date__        = "Aug 6, 2026"
__license__     = "MIT License"
#/////////////////////////////////////////////////

import numpy as np
import pytest

from fastslippy.pre_processing.grid import Grid
from fastslippy.pre_processing.model_parameters import (
    BoundaryProfile,
    FaultBottomTreatment,
    FaultSurfaceTreatment,
    ModelParameters,
)
from fastslippy.solver.matrix_builder import MatrixBuilder


def _reference_build_rh(p: ModelParameters, g: Grid, builder: MatrixBuilder,
                        dPdt: float, V: np.ndarray) -> np.ndarray:
    Nx, Ny, N = p.Nx, p.Ny, g.N
    G = p.G
    sina, cosa = g.sina, g.cosa
    y = g.y
    mid = Nx // 2
    dx_xuy = builder._dx_xuy
    dy_yuy = builder._dy_yuy
    dx_xux = builder._dx_xux
    dy_yux = builder._dy_yux
    uniform = builder._is_uniform
    dx_uniform = float(g.dx)
    dy_uniform = float(g.dy)
    surface_fault_intersection = (
        p.fault_surface_treatment
        == FaultSurfaceTreatment.FREE_SURFACE_FAULT_INTERSECTION
    )
    bottom_fault_intersection = (
        p.fault_bottom_treatment
        == FaultBottomTreatment.DEEP_BOUNDARY_FAULT_INTERSECTION
    )

    RH = np.zeros(N)

    for ix in range(Nx + 1):
        for iy in range(Ny + 1):
            kux, kuy = MatrixBuilder._dofs(ix, iy, Ny)

            if iy < Ny:
                if uniform:
                    dx_loc = dx_uniform
                    dy_loc = dy_uniform
                else:
                    dx_loc = dx_xuy[ix]
                    dy_loc = dy_yuy[iy]

                if ix == 0:
                    if p.bc.left.uy.type.name == "VELOCITY":
                        RH[kuy] = 2.0 * p.bc.left.uy.value
                elif ix == Nx:
                    if p.bc.right.uy.type.name == "VELOCITY":
                        RH[kuy] = 2.0 * p.bc.right.uy.value
                elif iy == 0:
                    is_reserved = surface_fault_intersection and ix in (mid, mid + 1)
                    if p.bc.top.uy.type.name == "VELOCITY" and not is_reserved:
                        profile = p.bc.top.uy.profile
                        if profile == BoundaryProfile.FULL:
                            RH[kuy] = p.bc.top.uy.value
                        elif profile == BoundaryProfile.POSITIVE_FAULT_BLOCK and ix > mid:
                            RH[kuy] = p.bc.top.uy.value
                        elif profile == BoundaryProfile.NEGATIVE_FAULT_BLOCK and ix <= mid:
                            RH[kuy] = p.bc.top.uy.value
                        elif profile == BoundaryProfile.ANTISYMMETRIC_ABOUT_FAULT:
                            RH[kuy] = (-1.0 if ix <= mid else 1.0) * p.bc.top.uy.value
                elif iy == Ny - 1:
                    if p.bc.bottom.uy.type.name == "VELOCITY":
                        profile = p.bc.bottom.uy.profile
                        if profile == BoundaryProfile.FULL:
                            RH[kuy] = p.bc.bottom.uy.value
                        elif profile == BoundaryProfile.POSITIVE_FAULT_BLOCK and ix > mid:
                            RH[kuy] = p.bc.bottom.uy.value
                        elif profile == BoundaryProfile.NEGATIVE_FAULT_BLOCK and ix <= mid:
                            RH[kuy] = p.bc.bottom.uy.value
                        elif profile == BoundaryProfile.ANTISYMMETRIC_ABOUT_FAULT:
                            RH[kuy] = (-1.0 if ix <= mid else 1.0) * p.bc.bottom.uy.value
                elif ix == mid:
                    RH[kuy] = V[iy]
                elif ix == mid + 1:
                    pass
                else:
                    if p.case_type == "groningen":
                        yv = y[iy]
                        if yv == 850 and ix >= mid + 1:
                            RH[kuy] = dPdt / dy_loc * dx_loc * dx_loc / G * sina
                        if yv == 1050 and ix >= mid + 1:
                            RH[kuy] = -dPdt / dy_loc * dx_loc * dx_loc / G * sina
                        if yv == 800 and ix <= mid:
                            RH[kuy] = dPdt / dy_loc * dx_loc * dx_loc / G * sina
                        if yv == 1000 and ix <= mid:
                            RH[kuy] = -dPdt / dy_loc * dx_loc * dx_loc / G * sina

            if ix < Nx:
                if uniform:
                    dx_loc = dx_uniform
                    dy_loc = dy_uniform
                else:
                    dx_loc = dx_xux[ix]
                    dy_loc = dy_yux[iy]

                if iy == 0:
                    if p.bc.top.ux.type.name == "VELOCITY":
                        RH[kux] = 2.0 * p.bc.top.ux.value
                elif iy == Ny:
                    if p.bc.bottom.ux.type.name == "VELOCITY":
                        RH[kux] = 2.0 * p.bc.bottom.ux.value
                elif ix == 0:
                    if p.bc.left.ux.type.name == "VELOCITY":
                        RH[kux] = p.bc.left.ux.value
                elif ix == Nx - 1:
                    if p.bc.right.ux.type.name == "VELOCITY":
                        RH[kux] = p.bc.right.ux.value
                elif ix == mid:
                    if p.case_type == "groningen":
                        yv = y[iy]
                        if 800 < yv <= 850:
                            RH[kux] = -dPdt * dx_loc / G
                        if 1000 < yv <= 1050:
                            RH[kux] = dPdt * dx_loc / G
                else:
                    if p.case_type == "groningen":
                        yv = y[iy]
                        if yv == 1050 and ix > mid + 1:
                            RH[kux] = dPdt / dy_loc * dx_loc * dx_loc / G * sina * cosa
                        if yv == 1000 and ix < mid + 1:
                            RH[kux] = dPdt / dy_loc * dx_loc * dx_loc / G * sina * cosa
                        if yv == 850 and ix > mid + 1:
                            RH[kux] = -dPdt / dy_loc * dx_loc * dx_loc / G * sina * cosa
                        if yv == 800 and ix < mid + 1:
                            RH[kux] = -dPdt / dy_loc * dx_loc * dx_loc / G * sina * cosa

    if surface_fault_intersection:
        _, kuy = MatrixBuilder._dofs(mid, 0, Ny)
        RH[kuy] = V[0]
    if bottom_fault_intersection:
        _, kuy = MatrixBuilder._dofs(mid, Ny - 1, Ny)
        RH[kuy] = V[Ny - 1]

    return RH


@pytest.mark.parametrize(
    "params,dPdt",
    [
        (
            ModelParameters(case_type="lab", Nx=31, Ny=31, xsize=1.0, ysize=1.0),
            0.0,
        ),
        (
            ModelParameters(case_type="california", Nx=31, Ny=31, xsize=1200.0, ysize=1000.0, W_f=500.0),
            0.0,
        ),
        (
            ModelParameters(case_type="groningen", Nx=51, Ny=201, xsize=2000.0, ysize=2000.0),
            -0.0127,
        ),
        (
            ModelParameters(
                case_type="groningen",
                Nx=51,
                Ny=201,
                xsize=2000.0,
                ysize=2000.0,
                x_stretch_enabled=True,
                x_stretch_inner_size=600.0,
                x_stretch_inner_points=21,
                y_stretch_enabled=True,
                y_stretch_inner_size=300.0,
                y_stretch_inner_points=41,
                allow_nonuniform_solver=True,
            ),
            -0.0127,
        ),
    ],
)
def test_build_rh_matches_reference_logic(params: ModelParameters, dPdt: float):
    params.bc.left.ux.set_velocity(1e-5)
    params.bc.left.uy.set_velocity(2e-5)
    params.bc.right.ux.set_velocity(3e-5)
    params.bc.right.uy.set_velocity(4e-5)
    params.bc.top.ux.set_velocity(5e-5)
    params.bc.top.uy.set_velocity(6e-5)
    params.bc.bottom.ux.set_velocity(7e-5)
    params.bc.bottom.uy.set_velocity(8e-5)
    params.loading.V_L = 9e-5

    grid = Grid(params)
    builder = MatrixBuilder(params, grid)
    V = np.linspace(1e-9, 2e-9, params.Ny)

    rh_actual = builder.build_RH(dPdt, V)
    rh_ref = _reference_build_rh(params, grid, builder, dPdt, V)

    np.testing.assert_allclose(rh_actual, rh_ref, rtol=0.0, atol=0.0)
