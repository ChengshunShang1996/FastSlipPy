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


def _manual_piecewise_metric(length, n_points, inner_length, inner_points, power, index_positions):
    n_total = n_points - 1
    n_inner = inner_points - 1
    sb = n_inner / n_total
    b = n_total * inner_length / n_inner
    s = np.clip(np.asarray(index_positions, dtype=float) / n_total, 0.0, 1.0)
    metric = np.full_like(s, b / n_total, dtype=float)
    outer = s > sb
    if np.any(outer):
        chi = (s[outer] - sb) / (1.0 - sb)
        metric[outer] = (b + (length - b) * power * np.power(chi, power - 1) / (1.0 - sb)) / n_total
    return metric


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


def test_stretched_grid_analytic_metrics_match_closed_form():
    params = ModelParameters(
        xsize=1200.0,
        ysize=1000.0,
        Nx=21,
        Ny=11,
        x_stretch_enabled=True,
        x_stretch_inner_size=300.0,
        x_stretch_inner_points=7,
        x_stretch_power=3,
        y_stretch_enabled=True,
        y_stretch_inner_size=200.0,
        y_stretch_inner_points=4,
        y_stretch_power=2,
    )
    grid = Grid(params)

    x_center = (params.Nx - 1) / 2
    x_nodes = np.abs(np.arange(params.Nx, dtype=float) - x_center)
    x_staggered = np.abs(np.arange(params.Nx + 1, dtype=float) - 0.5 - x_center)
    x_expected = _manual_piecewise_metric(
        length=params.xsize / 2.0,
        n_points=(params.Nx + 1) // 2,
        inner_length=params.x_stretch_inner_size / 2.0,
        inner_points=(params.x_stretch_inner_points + 1) // 2,
        power=params.x_stretch_power,
        index_positions=x_nodes,
    )
    xp_expected = _manual_piecewise_metric(
        length=params.xsize / 2.0,
        n_points=(params.Nx + 1) // 2,
        inner_length=params.x_stretch_inner_size / 2.0,
        inner_points=(params.x_stretch_inner_points + 1) // 2,
        power=params.x_stretch_power,
        index_positions=x_staggered,
    )
    y_expected = _manual_piecewise_metric(
        length=params.ysize,
        n_points=params.Ny,
        inner_length=params.y_stretch_inner_size,
        inner_points=params.y_stretch_inner_points,
        power=params.y_stretch_power,
        index_positions=np.arange(params.Ny, dtype=float),
    )
    yp_expected = _manual_piecewise_metric(
        length=params.ysize,
        n_points=params.Ny,
        inner_length=params.y_stretch_inner_size,
        inner_points=params.y_stretch_inner_points,
        power=params.y_stretch_power,
        index_positions=np.arange(params.Ny + 1, dtype=float) - 0.5,
    )

    np.testing.assert_allclose(grid.metric_x, x_expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(grid.metric_xp, xp_expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(grid.metric_y, y_expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(grid.metric_yp, yp_expected, rtol=0.0, atol=1e-12)


def test_nonuniform_matrix_builder_uses_analytic_metrics():
    params = ModelParameters(
        xsize=1200.0,
        ysize=1000.0,
        Nx=21,
        Ny=11,
        x_stretch_enabled=True,
        x_stretch_inner_size=300.0,
        x_stretch_inner_points=7,
        y_stretch_enabled=True,
        y_stretch_inner_size=200.0,
        y_stretch_inner_points=4,
        allow_nonuniform_solver=True,
    )
    grid = Grid(params)
    builder = MatrixBuilder(params, grid)

    np.testing.assert_allclose(builder._dx_xuy, grid.metric_xp, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(builder._dy_yuy, grid.metric_y, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(builder._dx_xux, grid.metric_x, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(builder._dy_yux, grid.metric_yp, rtol=0.0, atol=0.0)


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


def test_stretch_max_cell_size_caps_are_enforced():
    base_params = ModelParameters(
        xsize=1200.0,
        ysize=1000.0,
        Nx=21,
        Ny=11,
        x_stretch_enabled=True,
        x_stretch_inner_size=300.0,
        x_stretch_inner_points=7,
        y_stretch_enabled=True,
        y_stretch_inner_size=200.0,
        y_stretch_inner_points=4,
    )
    base_grid = Grid(base_params)
    max_dx = float(np.max(base_grid.dx_edges))
    max_dy = float(np.max(base_grid.dy_edges))

    capped_params_ok = ModelParameters(
        xsize=1200.0,
        ysize=1000.0,
        Nx=21,
        Ny=11,
        x_stretch_enabled=True,
        x_stretch_inner_size=300.0,
        x_stretch_inner_points=7,
        y_stretch_enabled=True,
        y_stretch_inner_size=200.0,
        y_stretch_inner_points=4,
        x_stretch_max_cell_size=max_dx * 1.001,
        y_stretch_max_cell_size=max_dy * 1.001,
    )
    Grid(capped_params_ok)

    capped_params_fail = ModelParameters(
        xsize=1200.0,
        ysize=1000.0,
        Nx=21,
        Ny=11,
        x_stretch_enabled=True,
        x_stretch_inner_size=300.0,
        x_stretch_inner_points=7,
        y_stretch_enabled=True,
        y_stretch_inner_size=200.0,
        y_stretch_inner_points=4,
        x_stretch_max_cell_size=max_dx * 0.9,
    )
    with pytest.raises(ValueError, match="x-stretch max cell size exceeded"):
        Grid(capped_params_fail)


def test_max_cell_size_error_reports_suggested_mesh_size():
    base_params = ModelParameters(
        xsize=1200.0,
        ysize=1000.0,
        Nx=21,
        Ny=11,
        x_stretch_enabled=True,
        x_stretch_inner_size=300.0,
        x_stretch_inner_points=7,
    )
    base_grid = Grid(base_params)
    too_small_cap = float(np.max(base_grid.dx_edges)) * 0.9
    params = ModelParameters(
        xsize=1200.0,
        ysize=1000.0,
        Nx=21,
        Ny=11,
        x_stretch_enabled=True,
        x_stretch_inner_size=300.0,
        x_stretch_inner_points=7,
        x_stretch_max_cell_size=too_small_cap,
    )
    with pytest.raises(ValueError, match=r"Suggested Nx=\d+"):
        Grid(params)
