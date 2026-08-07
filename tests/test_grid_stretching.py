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
from fastslippy.pre_processing.model_parameters import ModelParameters
from fastslippy.solver.matrix_builder import MatrixBuilder


def test_uniform_grid_remains_uniform():
    params = ModelParameters(xsize=1200.0, ysize=1000.0, Nx=21, Ny=11)
    grid = Grid(params)

    assert not grid.is_nonuniform
    assert np.allclose(grid.dx_edges, grid.dx_edges[0])
    assert np.allclose(grid.dy_edges, grid.dy_edges[0])


def test_stretched_x_and_y_generate_nonuniform_mesh():
    params = ModelParameters(
        xsize=1200.0,
        ysize=1000.0,
        Nx=21,
        Ny=11,
        x_stretch_enabled=True,
        x_stretch_inner_size=300.0,
        x_stretch_inner_points=7,
        x_stretch_power=2,
        y_stretch_enabled=True,
        y_stretch_inner_size=200.0,
        y_stretch_inner_points=4,
        y_stretch_power=2,
    )
    grid = Grid(params)

    assert grid.is_nonuniform
    assert np.all(np.diff(grid.x) > 0)
    assert np.all(np.diff(grid.y) > 0)
    assert np.isclose(grid.x[params.Nx // 2], 0.0)

    # Finer spacing around fault core in x.
    mid = params.Nx // 2
    assert (grid.x[mid + 1] - grid.x[mid]) < (grid.x[-1] - grid.x[-2])

    # Finer spacing near y=0 than at depth.
    assert (grid.y[1] - grid.y[0]) < (grid.y[-1] - grid.y[-2])


def test_invalid_stretch_inner_zone_is_rejected():
    with pytest.raises(ValueError):
        ModelParameters(
            xsize=1200.0,
            Nx=21,
            x_stretch_enabled=True,
            x_stretch_inner_size=700.0,
            x_stretch_inner_points=7,
        )


def test_nonuniform_matrix_builder_requires_explicit_opt_in():
    params = ModelParameters(
        xsize=1200.0,
        ysize=1000.0,
        Nx=21,
        Ny=11,
        x_stretch_enabled=True,
        x_stretch_inner_size=300.0,
        x_stretch_inner_points=7,
    )
    grid = Grid(params)
    with pytest.raises(ValueError, match="allow_nonuniform_solver=True"):
        MatrixBuilder(params, grid)
