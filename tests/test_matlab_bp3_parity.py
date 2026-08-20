import numpy as np
from math import factorial

from examples.run_case_BP3 import build_bp3_parameters
from fastslippy import FastSlipPy
from fastslippy.pre_processing.frictional_zones import FrictionalZones
from fastslippy.pre_processing.grid import Grid
from fastslippy.pre_processing.model_parameters import ModelParameters
from fastslippy.solver.fault_state import FaultState
from fastslippy.solver.matrix_builder import MatrixBuilder
from fastslippy.solver.stress_state import StressState
from fastslippy.utilities.grid_operators import (
    build_recovery_operators,
    finite_difference_weights,
)


def _bp3_parameters(**overrides):
    values = dict(
        case_type="california",
        alpha=60.0,
        Nx=11,
        Ny=10,
        xsize=1000.0,
        ysize=900.0,
        W_f=700.0,
        H=300.0,
        h=100.0,
        rho=2670.0,
        cs=3464.0,
        nu=0.25,
        mu0=0.6,
        V0=1e-6,
        a0=0.01,
        a_max=0.025,
        b0=0.015,
        L=0.008,
        Vi=1e-9,
    )
    values.update(overrides)
    return ModelParameters(**values)


def test_bp3_example_grid_matches_matlab_stretch_parameters():
    params = build_bp3_parameters()
    grid = Grid(params)

    element_size = 200.0
    x_core = 10e3
    y_core = 20e3
    nx_stretch = 40
    ny_stretch = 36
    expected_nx = 2 * (round(x_core / element_size) + nx_stretch) + 1
    expected_ny = round(y_core / element_size) + ny_stretch + 1

    assert (params.Nx, params.Ny) == (expected_nx, expected_ny)
    assert grid.is_nonuniform
    np.testing.assert_allclose(
        grid.x[params.Nx // 2 - 50:params.Nx // 2 + 51],
        np.linspace(-x_core, x_core, 101),
    )
    np.testing.assert_allclose(grid.y[:101], np.linspace(0.0, y_core, 101))
    np.testing.assert_allclose([grid.x[0], grid.x[-1]], [-40e3, 40e3])
    np.testing.assert_allclose([grid.y[0], grid.y[-1]], [0.0, 45e3])


def test_fdweights_are_polynomial_exact_off_node():
    points = np.array([-3.0, -0.5, 1.25, 4.0])
    target = 0.4
    for derivative in (0, 1, 2):
        weights = finite_difference_weights(target, points, derivative)
        for power in range(points.size):
            values = points**power
            if power < derivative:
                expected = 0.0
            else:
                coefficient = factorial(power) / factorial(
                    power - derivative
                )
                expected = coefficient * target ** (power - derivative)
            np.testing.assert_allclose(weights @ values, expected, atol=2e-13)


def test_recovery_operators_fix_boundaries_and_graded_interpolation():
    x = np.array([-4.0, -1.0, 0.5, 3.0, 8.0])
    y = np.array([0.0, 1.0, 2.5, 6.0, 11.0])

    def midpoints(values):
        result = np.empty(values.size + 1)
        result[1:-1] = 0.5 * (values[:-1] + values[1:])
        result[0] = values[0] - 0.5 * (values[1] - values[0])
        result[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
        return result

    xp = midpoints(x)
    yp = midpoints(y)
    recovery = build_recovery_operators(x, y, xp, yp)

    np.testing.assert_allclose(
        recovery.derivative_x @ (x**2 + 3*x - 2), 2*x + 3, atol=2e-13
    )
    np.testing.assert_allclose(
        recovery.derivative_y @ (y**2 - 4*y), 2*y - 4, atol=2e-13
    )
    np.testing.assert_allclose(
        recovery.midpoint_y_to_node @ (2*yp + 7), 2*y + 7, atol=2e-13
    )
    np.testing.assert_allclose(
        recovery.sigma_centres_to_nodes @ (3*yp[1:-1] - 5),
        3*y - 5,
        atol=2e-13,
    )


def test_bp3_surface_bottom_and_side_rows_match_matlab_layout():
    params = _bp3_parameters()
    params.bc.left.uy.set_velocity(-0.5e-9)
    params.bc.right.uy.set_velocity(0.5e-9)
    params.bc.top.set_traction_free()
    params.bc.bottom.set_fixed()
    grid = Grid(params)
    builder = MatrixBuilder(params, grid)
    matrix = builder.build_LH()
    mid = params.Nx // 2

    for iy in (0, params.Ny - 1):
        _, row = builder._dofs(mid, iy, params.Ny)
        left = row
        right = row + (params.Ny + 1) * 2
        actual = dict(zip(matrix.getrow(row).indices, matrix.getrow(row).data))
        assert actual == {left: -1.0, right: 1.0}

    fault_ghost, _ = builder._dofs(mid, 0, params.Ny)
    actual = dict(
        zip(
            matrix.getrow(fault_ghost).indices,
            matrix.getrow(fault_ghost).data,
        )
    )
    assert actual == {
        fault_ghost: 1.0,
        fault_ghost + 2: -2.0,
        fault_ghost + 4: 1.0,
    }

    _, left_row = builder._dofs(0, 4, params.Ny)
    actual = dict(zip(matrix.getrow(left_row).indices, matrix.getrow(left_row).data))
    assert actual == {
        left_row: 1.0,
        left_row + (params.Ny + 1) * 2: 1.0,
    }

    velocity = np.linspace(1e-9, 2e-9, params.Ny)
    rhs = builder.build_RH(0.0, velocity)
    np.testing.assert_allclose(rhs[builder._kuy[:, mid]], velocity)
    np.testing.assert_allclose(
        rhs[builder._kuy[:, 0]], 2.0 * params.bc.left.uy.value
    )
    np.testing.assert_allclose(
        rhs[builder._kuy[:, params.Nx]], 2.0 * params.bc.right.uy.value
    )


def test_bp3_half_node_second_derivatives_are_cubic_exact():
    params = _bp3_parameters(
        Nx=21,
        Ny=16,
        xsize=1200.0,
        ysize=1000.0,
        x_stretch_enabled=True,
        y_stretch_enabled=True,
        x_stretch_inner_size=300.0,
        y_stretch_inner_size=200.0,
        x_stretch_inner_points=7,
        y_stretch_inner_points=4,
        allow_nonuniform_solver=True,
    )
    grid = Grid(params)
    builder = MatrixBuilder(params, grid)
    matrix = builder.build_LH()
    coupling = (params.lam + params.G) / params.G

    ux_only = np.zeros(grid.N)
    uy_only = np.zeros(grid.N)
    for ix in range(params.Nx + 1):
        for iy in range(params.Ny + 1):
            kux, kuy = builder._dofs(ix, iy, params.Ny)
            if ix < params.Nx:
                ux_only[kux] = grid.x[ix] ** 3
            if iy < params.Ny:
                uy_only[kuy] = grid.y[iy] ** 3

    ix, iy = 3, 5
    _, uy_row = builder._dofs(ix, iy, params.Ny)
    expected = (
        -grid.cosa * coupling * builder._dx_xuy[ix] ** 2
        * 6.0 * grid.xp[ix]
    )
    np.testing.assert_allclose(matrix.getrow(uy_row) @ ux_only, expected, rtol=2e-14)

    ix, iy = 4, 10
    ux_row, _ = builder._dofs(ix, iy, params.Ny)
    expected = (
        -grid.cosa * coupling * builder._dx_xux[ix] ** 2
        * 6.0 * grid.yp[iy]
    )
    np.testing.assert_allclose(matrix.getrow(ux_row) @ uy_only, expected, rtol=2e-14)


def test_signed_bp3_friction_recovers_initial_steady_velocity():
    for sign in (1.0, -1.0):
        params = _bp3_parameters(Vi=sign * 1e-9, W_f=1e9)
        params.loading.V_p = sign * 1e-9
        params.loading.V_L = sign * 1e-9
        grid = Grid(params)
        friction = FrictionalZones(params, grid.y)
        stress = StressState(params, grid.y)
        fault = FaultState(params, stress, friction, fault_y=grid.y)

        expected_tau0 = (
            params.sigma0 * params.a_max
            * np.arcsinh(
                params.Vi / (2 * params.V0)
                * np.exp(
                    (
                        params.mu0
                        + params.b0 * np.log(params.V0 / abs(params.Vi))
                    ) / params.a_max
                )
            )
            + params.eta * params.Vi
        )
        np.testing.assert_allclose(stress.tau0, expected_tau0)

        fault.solve_slip_rate_matlab(np.zeros(params.Ny), stress, friction)
        np.testing.assert_allclose(fault.V, params.Vi, rtol=0.0, atol=0.0)
        fault.solve_slip_rate_newton_v2(np.zeros(params.Ny), stress, friction)
        np.testing.assert_allclose(fault.V, params.Vi, rtol=0.0, atol=0.0)


def test_newton_v2_solves_both_velocity_branches_after_large_stress_step():
    for sign in (1.0, -1.0):
        params = _bp3_parameters(Vi=sign * 1e-9, W_f=1e9)
        params.loading.V_p = sign * 1e-9
        params.loading.V_L = sign * 1e-9
        grid = Grid(params)
        friction = FrictionalZones(params, grid.y)
        stress = StressState(params, grid.y)
        fault = FaultState(params, stress, friction, fault_y=grid.y)

        # Put the root far outside the old ``2 * previous V`` bracket.
        fault.V[:] = sign * 1e-30
        stress_step = np.full(params.Ny, sign * 5.0e6)
        fault.solve_slip_rate_newton_v2(stress_step, stress, friction)

        assert np.all(np.signbit(fault.V) == (sign < 0.0))
        exponent = (
            params.mu0
            + friction.b * np.log(params.V0 * fault.theta / params.L)
        ) / friction.a
        residual = (
            fault.sigma * friction.a
            * np.arcsinh(
                fault.V / (2.0 * params.V0) * np.exp(exponent)
            )
            + params.eta * fault.V
            - (stress.tau0 + stress_step)
        )
        assert np.max(np.abs(residual)) <= params.friction_tolerance


def test_short_bp3_run_advances_to_exact_final_time(tmp_path):
    params = _bp3_parameters(
        Nx=11,
        Ny=10,
        xsize=1000.0,
        ysize=900.0,
        x_stretch_enabled=True,
        y_stretch_enabled=True,
        x_stretch_inner_size=200.0,
        y_stretch_inner_size=150.0,
        x_stretch_inner_points=5,
        y_stretch_inner_points=4,
        x_stretch_power=2,
        y_stretch_power=2,
        allow_nonuniform_solver=True,
        Nt=3,
        dt_init=1.0,
        dt_max=1.0,
        tfinal=3.0,
        output_interval=1,
        checkpoint_interval=100,
        output_vtk_option=False,
    )
    params.loading.V_p = 1e-9
    params.loading.V_L = 1e-9
    params.bc.left.ux.set_fixed()
    params.bc.left.uy.set_velocity(-0.5e-9)
    params.bc.right.ux.set_fixed()
    params.bc.right.uy.set_velocity(0.5e-9)
    params.bc.top.set_traction_free()
    params.bc.bottom.set_fixed()

    model = FastSlipPy(params=params, output_dir=str(tmp_path))
    assert model.grid.is_nonuniform
    model.figure_creator.plot_results = lambda *args, **kwargs: None
    newton_v2 = model.fault.solve_slip_rate_newton_v2
    calls = []

    def tracked_newton_v2(*args, **kwargs):
        calls.append(True)
        return newton_v2(*args, **kwargs)

    model.fault.solve_slip_rate_newton_v2 = tracked_newton_v2
    model.fault.solve_slip_rate_matlab = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("BP3 should use Newton v2, not MATLAB bisection")
    )
    model.run()

    np.testing.assert_allclose(model.output.tm, [1.0, 2.0, 3.0])
    assert len(calls) == 3
    assert np.all(np.isfinite(model.fault.V))
    assert np.all(np.isfinite(model.fault.sigma))
