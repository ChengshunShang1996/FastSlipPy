"""Static diagnostics for the BP3 elastic operator and its deep boundary."""

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from fastslippy.pre_processing.grid import Grid
from fastslippy.pre_processing.model_parameters import ModelParameters
from fastslippy.solver.matrix_builder import MatrixBuilder
from fastslippy.utilities.stress_cal_util import StressCalUtil


def _bp3_params(alpha: float, *, stretched: bool) -> ModelParameters:
    kwargs = {}
    if stretched:
        kwargs.update(
            x_stretch_enabled=True,
            y_stretch_enabled=True,
            x_stretch_inner_size=20e3,
            y_stretch_inner_size=20e3,
            x_stretch_inner_points=21,
            y_stretch_inner_points=21,
            x_stretch_power=2,
            y_stretch_power=2,
            allow_nonuniform_solver=True,
        )
    params = ModelParameters(
        case_type="california",
        alpha=alpha,
        xsize=80e3,
        ysize=60e3,
        Nx=41,
        Ny=41,
        **kwargs,
    )
    params.bc.left.ux.set_fixed()
    params.bc.left.uy.set_fixed()
    params.bc.right.ux.set_fixed()
    params.bc.right.uy.set_fixed()
    params.bc.top.set_traction_free()
    params.bc.bottom.set_traction_free()
    return params


def _pack(builder: MatrixBuilder, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    params = builder.p
    result = np.zeros(builder.grid.N)
    for ix in range(params.Nx + 1):
        for iy in range(params.Ny + 1):
            kux, kuy = builder._dofs(ix, iy, params.Ny)
            if ix < params.Nx:
                result[kux] = ux[iy, ix]
            if iy < params.Ny:
                result[kuy] = uy[iy, ix]
    return result


def _solve_static_bp3(alpha: float, n: int, fault_rate=None):
    """Solve one dimensionless BP3 loading/slip-rate elasticity problem."""

    params = ModelParameters(
        case_type="california",
        alpha=alpha,
        xsize=80e3,
        ysize=60e3,
        Nx=n,
        Ny=n,
        W_f=40e3,
        x_stretch_enabled=True,
        y_stretch_enabled=True,
        x_stretch_inner_size=20e3,
        y_stretch_inner_size=20e3,
        x_stretch_inner_points=(n + 1) // 2,
        y_stretch_inner_points=(n + 1) // 2,
        x_stretch_power=2,
        y_stretch_power=2,
        allow_nonuniform_solver=True,
    )
    params.bc.left.ux.set_fixed()
    params.bc.right.ux.set_fixed()
    params.bc.top.set_traction_free()
    params.bc.bottom.set_traction_free()
    if fault_rate is None:
        params.bc.left.uy.set_velocity(-0.5)
        params.bc.right.uy.set_velocity(0.5)
    else:
        params.bc.left.uy.set_fixed()
        params.bc.right.uy.set_fixed()

    grid = Grid(params)
    builder = MatrixBuilder(params, grid)
    if fault_rate is None:
        fault_rate = np.zeros(params.Ny)
    elif callable(fault_rate):
        fault_rate = fault_rate(grid.y)
    solution = spsolve(
        builder.build_LH().tocsc(), builder.build_RH(0.0, fault_rate)
    )
    ux = solution[0::2].reshape(params.Nx + 1, params.Ny + 1).T[:, :params.Nx]
    uy = solution[1::2].reshape(params.Nx + 1, params.Ny + 1).T[:params.Ny, :]
    stress_util = StressCalUtil(prefer_numba=False)
    tau, sigma = stress_util.compute_stress_fields(
        uy, ux, grid.dx, grid.dy, params.lam, params.G,
        grid.cosa, grid.sina, params.Ny, params.Nx,
        x=grid.x, y=grid.y, xp=grid.xp, yp=grid.yp,
    )
    mid = params.Nx // 2
    normal_left, normal_right = stress_util.recover_fault_normal_stress(
        sigma, grid.x, grid.y, grid.xp, grid.yp,
        left_column=mid - 1, right_column=mid,
    )
    return grid, tau, normal_left, normal_right


@pytest.mark.parametrize("stretched", [False, True])
@pytest.mark.parametrize("alpha", [30.0, 60.0, 90.0])
def test_horizontal_traction_rows_match_nonzero_affine_solution(alpha, stretched):
    """Top and bottom rows must return the analytic, nonzero traction.

    Checking a nonzero value prevents a sign or scale error from passing merely
    because every term vanishes for rigid motion or terms cancel for one
    specially selected traction-free affine field.
    """

    params = _bp3_params(alpha, stretched=stretched)
    grid = Grid(params)
    builder = MatrixBuilder(params, grid)

    # Components in the oblique basis used by the solver:
    # ux = A*x + B*y, uy = C*x + D*y.
    A, B, C, D = 1.7e-6, -2.3e-6, 0.9e-6, 3.1e-6
    ux = A * grid.x[None, :] + B * grid.yp[:, None]
    uy = C * grid.xp[None, :] + D * grid.y[:, None]
    residual = builder.build_LH() @ _pack(builder, ux, uy)

    lam, shear = params.lam, params.G
    c, s = grid.cosa, grid.sina
    normal_traction = (
        lam * A + (lam + 2.0 * shear) * D - 2.0 * shear * c * C
    )
    shear_traction = shear / s * (
        B + (1.0 - 2.0 * c * c) * C - c * A + c * D
    )

    mid = params.Nx // 2
    # Exclude side corners and the duplicated fault rows, whose equations are
    # selected by a different branch of MatrixBuilder.
    x_indices = [ix for ix in range(2, params.Nx - 2) if ix not in (mid, mid + 1)]
    for ix in x_indices:
        top_ux, top_uy = builder._dofs(ix, 0, params.Ny)
        bottom_ux, _ = builder._dofs(ix, params.Ny, params.Ny)
        _, bottom_uy = builder._dofs(ix, params.Ny - 1, params.Ny)

        expected_ux = builder._dx_xux[ix] / shear * shear_traction
        expected_uy = builder._dx_xuy[ix] / shear * normal_traction
        np.testing.assert_allclose(
            residual[top_ux], expected_ux, rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(
            residual[bottom_ux], expected_ux, rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(
            residual[top_uy], expected_uy, rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(
            residual[bottom_uy], expected_uy, rtol=1e-10, atol=1e-12
        )


@pytest.mark.parametrize("stretched", [False, True])
@pytest.mark.parametrize("alpha", [30.0, 60.0, 90.0])
def test_horizontal_traction_rows_match_quadratic_solution(alpha, stretched):
    """Boundary traction recovery must remain quadratic-exact when stretched."""

    params = _bp3_params(alpha, stretched=stretched)
    grid = Grid(params)
    builder = MatrixBuilder(params, grid)

    # Coefficients are deliberately generic so every derivative entering both
    # horizontal traction components is nonzero and varies along the boundary.
    axx, axy, ayy, ax, ay = 2.1e-11, -1.3e-11, 0.7e-11, 1.7e-6, -2.3e-6
    bxx, bxy, byy, bx, by = -0.8e-11, 1.9e-11, -1.1e-11, 0.9e-6, 3.1e-6

    def ux_field(x, y):
        return axx * x*x + axy * x*y + ayy * y*y + ax * x + ay * y

    def uy_field(x, y):
        return bxx * x*x + bxy * x*y + byy * y*y + bx * x + by * y

    ux = ux_field(grid.x[None, :], grid.yp[:, None])
    uy = uy_field(grid.xp[None, :], grid.y[:, None])
    residual = builder.build_LH() @ _pack(builder, ux, uy)

    lam, shear = params.lam, params.G
    c, s = grid.cosa, grid.sina

    def tractions(x, y):
        ux_x = 2.0 * axx * x + axy * y + ax
        ux_y = axy * x + 2.0 * ayy * y + ay
        uy_x = 2.0 * bxx * x + bxy * y + bx
        uy_y = bxy * x + 2.0 * byy * y + by
        normal = lam * ux_x + (lam + 2.0 * shear) * uy_y - 2.0 * shear * c * uy_x
        tangent = shear / s * (
            ux_y + (1.0 - 2.0 * c*c) * uy_x - c * ux_x + c * uy_y
        )
        return tangent, normal

    mid = params.Nx // 2
    x_indices = [ix for ix in range(2, params.Nx - 2) if ix not in (mid, mid + 1)]
    for ix in x_indices:
        top_ux, top_uy = builder._dofs(ix, 0, params.Ny)
        bottom_ux, _ = builder._dofs(ix, params.Ny, params.Ny)
        _, bottom_uy = builder._dofs(ix, params.Ny - 1, params.Ny)
        top_tangent, top_normal = tractions(grid.x[ix], grid.y[0])
        bottom_tangent, bottom_normal = tractions(grid.x[ix], grid.y[-1])
        # Normal traction rows live at xp[ix], rather than x[ix].
        _, top_normal = tractions(grid.xp[ix], grid.y[0])
        _, bottom_normal = tractions(grid.xp[ix], grid.y[-1])

        np.testing.assert_allclose(
            residual[top_ux], builder._dx_xux[ix] / shear * top_tangent,
            rtol=2e-10, atol=2e-12,
        )
        np.testing.assert_allclose(
            residual[bottom_ux], builder._dx_xux[ix] / shear * bottom_tangent,
            rtol=2e-10, atol=2e-12,
        )
        np.testing.assert_allclose(
            residual[top_uy], builder._dx_xuy[ix] / shear * top_normal,
            rtol=2e-10, atol=2e-12,
        )
        np.testing.assert_allclose(
            residual[bottom_uy], builder._dx_xuy[ix] / shear * bottom_normal,
            rtol=2e-10, atol=2e-12,
        )


@pytest.mark.parametrize("alpha", [30.0, 60.0, 90.0])
def test_stretched_bulk_annuls_quadratic_harmonic_equilibrium(alpha):
    """A quadratic displacement in exact Navier equilibrium has zero bulk load."""

    params = _bp3_params(alpha, stretched=True)
    grid = Grid(params)
    builder = MatrixBuilder(params, grid)

    # grad(X**3 - 3*X*Z**2), expressed in the oblique displacement basis.
    ux_global = 3.0 * grid.Xux**2 - 3.0 * grid.Yux**2
    uz_global = -6.0 * grid.Xux * grid.Yux
    ux = ux_global - grid.cosa * uz_global / grid.sina
    uy = -6.0 * grid.Xuy * grid.Yuy / grid.sina
    residual = builder.build_LH() @ _pack(builder, ux, uy)

    mid = params.Nx // 2
    rows = []
    for ix in range(2, params.Nx - 1):
        if ix in (mid, mid + 1):
            continue
        for iy in range(2, params.Ny - 2):
            kux, kuy = builder._dofs(ix, iy, params.Ny)
            rows.extend((kux, kuy))

    # All coordinate-aware derivative formulas used here are polynomial-exact
    # for a quadratic field, so this should be a roundoff-level identity rather
    # than merely a small relative residual.
    scale = np.max(np.abs(_pack(builder, ux, uy)))
    assert np.max(np.abs(residual[rows])) / scale < 1e-11


@pytest.mark.parametrize("alpha", [30.0, 60.0, 90.0])
def test_stretched_bulk_matches_nonzero_quadratic_body_force(alpha):
    """Every transformed Navier term has the right sign and row scaling."""

    params = _bp3_params(alpha, stretched=True)
    grid = Grid(params)
    builder = MatrixBuilder(params, grid)
    axx, axy, ayy = 2.1e-6, -1.3e-6, 0.7e-6
    bxx, bxy, byy = -0.8e-6, 1.9e-6, -1.1e-6
    ux = (
        axx * grid.x[None, :] ** 2
        + axy * grid.x[None, :] * grid.yp[:, None]
        + ayy * grid.yp[:, None] ** 2
    )
    uy = (
        bxx * grid.xp[None, :] ** 2
        + bxy * grid.xp[None, :] * grid.y[:, None]
        + byy * grid.y[:, None] ** 2
    )
    residual = builder.build_LH() @ _pack(builder, ux, uy)

    c = grid.cosa
    ratio = (params.lam + 2.0 * params.G) / params.G
    coupling = (params.lam + params.G) / params.G
    rotated = c * (params.lam + 3.0 * params.G) / params.G
    expected_ux = (
        ratio * 2.0 * axx
        + 2.0 * ayy
        - rotated * axy
        + coupling * bxy
        - c * coupling * 2.0 * byy
    )
    expected_uy = (
        2.0 * bxx
        + ratio * 2.0 * byy
        - rotated * bxy
        + coupling * axy
        - c * coupling * 2.0 * axx
    )

    mid = params.Nx // 2
    for ix in range(2, params.Nx - 1):
        if ix in (mid, mid + 1):
            continue
        _, _, _, ux_scale = builder._second_derivative_weights(grid.x, ix)
        _, _, _, uy_scale = builder._second_derivative_weights(grid.xp, ix)
        for iy in range(2, params.Ny - 2):
            kux, kuy = builder._dofs(ix, iy, params.Ny)
            np.testing.assert_allclose(
                residual[kux], ux_scale * expected_ux, rtol=2e-10, atol=2e-5
            )
            np.testing.assert_allclose(
                residual[kuy], uy_scale * expected_uy, rtol=2e-10, atol=2e-5
            )


def test_side_displacement_owns_horizontal_traction_corners():
    """The x=+/-Lx displacement condition includes top and bottom corners."""

    params = _bp3_params(60.0, stretched=True)
    grid = Grid(params)
    builder = MatrixBuilder(params, grid)
    matrix = builder.build_LH().tocsr()

    for ix in (0, params.Nx - 1):
        for iy in (0, params.Ny):
            kux, _ = builder._dofs(ix, iy, params.Ny)
            row = matrix.getrow(kux)
            assert row.nnz == 1
            assert row.indices[0] == kux
            assert row.data[0] == pytest.approx(1.0)


def test_static_locked_fault_loading_kernel_converges_with_resolution():
    """Fault tractions converge without integrating an earthquake cycle."""

    values = []
    for n in (41, 61, 81):
        grid, tau, normal_left, normal_right = _solve_static_bp3(60.0, n)
        mid = n // 2
        depth_index = int(np.argmin(np.abs(grid.y - 10e3)))
        assert grid.y[depth_index] == pytest.approx(10e3)
        values.append(
            np.array(
                [
                    tau[depth_index, mid],
                    0.5
                    * (normal_left[depth_index] + normal_right[depth_index]),
                ]
            )
        )

    coarse_to_fine = np.abs(values[0] - values[2])
    medium_to_fine = np.abs(values[1] - values[2])
    assert np.all(medium_to_fine < coarse_to_fine)


def test_vertical_unit_slip_keeps_normal_coupling_small():
    """The special vertical BP3 rows preserve the mode-II decoupling."""

    n = 61
    _, tau, normal_left, normal_right = _solve_static_bp3(
        90.0,
        n,
        fault_rate=lambda y: np.exp(-((y - 12e3) / 3e3) ** 2),
    )
    mid = n // 2
    shear_scale = np.max(np.abs(tau[:, mid - 1]))
    normal_scale = np.max(np.abs(0.5 * (normal_left + normal_right)))
    assert normal_scale / shear_scale < 2e-3
