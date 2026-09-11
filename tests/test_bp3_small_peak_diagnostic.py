"""Fast nonlinear small-peak checks driven by the actual BP3 elastic operator."""

import numpy as np

from examples.run_bp3_small_peak_diagnostic import build_small_bp3_parameters
from fastslippy.pre_processing.grid import Grid
from fastslippy.utilities.bp3_small_peak import (
    FaultModeResponse,
    build_fault_traction_response,
    critical_stiffness,
    diagnose_fault_reciprocity,
    diagnose_nucleation_stiffness,
    gaussian_fault_mode,
    localized_gaussian_basis,
    project_fault_history_onto_modes,
    rate_state_friction_coefficient,
    reduced_rate_state_jacobian,
    simulate_modal_pulse,
    signed_rate_state_friction_derivatives_profile,
    signed_rate_state_friction_coefficient_profile,
    solve_fault_mode_response,
    solve_fault_loading_response,
    solve_fault_traction_response,
)


def test_fault_reciprocity_uses_pure_shear_work_matrix():
    y = np.array([0.0, 1.0, 3.0, 6.0])
    modes = np.eye(y.size)
    spacing = np.diff(y)
    weights = np.empty_like(y)
    weights[0] = 0.5 * spacing[0]
    weights[-1] = 0.5 * spacing[-1]
    weights[1:-1] = 0.5 * (spacing[:-1] + spacing[1:])
    symmetric_work = np.array([
        [4.0, -1.0, 0.5, 0.0],
        [-1.0, 3.0, 0.25, 0.1],
        [0.5, 0.25, 2.0, -0.2],
        [0.0, 0.1, -0.2, 1.0],
    ])
    response = FaultModeResponse(
        y=y,
        modes=modes,
        tau=symmetric_work / weights[:, None],
        # Deliberately nonsymmetric normal coupling must not contaminate the
        # elastic reciprocity metric.
        sigma_effective=np.triu(np.ones_like(symmetric_work)),
    )
    result = diagnose_fault_reciprocity(response)

    np.testing.assert_allclose(result.work_matrix, symmetric_work)
    assert result.antisymmetric_fraction < 1e-15
    assert result.maximum_pairwise_fraction < 1e-15


def test_modal_history_projection_recovers_coefficients_and_residual():
    y = np.linspace(0.0, 10.0, 21)
    modes = np.column_stack((
        np.sin(np.pi * y / 10.0),
        np.sin(2.0 * np.pi * y / 10.0),
    ))
    coefficients = np.array([[0.0, 2.0, -1.0], [0.0, -0.5, 3.0]])
    history = 7.0 + modes @ coefficients
    result = project_fault_history_onto_modes(
        y, modes, history, reference_index=0,
    )

    np.testing.assert_allclose(result.coefficients, coefficients, atol=2e-15)
    np.testing.assert_allclose(result.captured_fraction[1:], 1.0, atol=2e-15)
    assert result.captured_fraction[0] == 0.0


def test_rate_state_friction_derivatives_match_finite_differences():
    velocity = np.array([1e-12, 1e-9, 1e-5])
    theta = np.array([3e7, 8e6, 2e3])
    a = np.full(3, 0.01)
    b = np.full(3, 0.015)
    friction, derivative_velocity, derivative_theta = (
        signed_rate_state_friction_derivatives_profile(
            velocity, theta, a=a, b=b, mu0=0.6, V0=1e-6, L=0.008
        )
    )
    velocity_step = velocity * 1e-6
    theta_step = theta * 1e-6
    friction_velocity_plus = signed_rate_state_friction_coefficient_profile(
        velocity + velocity_step, theta, a=a, b=b,
        mu0=0.6, V0=1e-6, L=0.008,
    )
    friction_velocity_minus = signed_rate_state_friction_coefficient_profile(
        velocity - velocity_step, theta, a=a, b=b,
        mu0=0.6, V0=1e-6, L=0.008,
    )
    friction_theta_plus = signed_rate_state_friction_coefficient_profile(
        velocity, theta + theta_step, a=a, b=b,
        mu0=0.6, V0=1e-6, L=0.008,
    )
    friction_theta_minus = signed_rate_state_friction_coefficient_profile(
        velocity, theta - theta_step, a=a, b=b,
        mu0=0.6, V0=1e-6, L=0.008,
    )

    assert np.all(np.isfinite(friction))
    np.testing.assert_allclose(
        derivative_velocity,
        (friction_velocity_plus - friction_velocity_minus) / (2 * velocity_step),
        rtol=2e-8,
    )
    np.testing.assert_allclose(
        derivative_theta,
        (friction_theta_plus - friction_theta_minus) / (2 * theta_step),
        rtol=2e-8,
    )


def test_reduced_rate_state_jacobian_has_expected_shape_and_is_finite():
    y = np.linspace(0.0, 10.0, 21)
    modes = np.column_stack((
        np.sin(np.pi * y / 10.0),
        np.sin(2.0 * np.pi * y / 10.0),
    ))
    tau_responses = -3e7 * modes
    sigma_responses = 1e5 * modes
    velocity = np.full(y.shape, 1e-9)
    theta = np.full(y.shape, 0.008 / 1e-9)
    sigma = np.full(y.shape, 50e6)
    a = np.full(y.shape, 0.01)
    b = np.full(y.shape, 0.015)
    jacobian = reduced_rate_state_jacobian(
        y, modes, tau_responses, sigma_responses,
        velocity=velocity, theta=theta, sigma_effective=sigma,
        a=a, b=b, mu0=0.6, V0=1e-6, L=0.008,
        eta=4.6e6, metric_profile=np.full(y.shape, 1.0),
    )

    assert jacobian.shape == (4, 4)
    assert np.all(np.isfinite(jacobian))


def test_signed_friction_profile_matches_scalar_rate_state_law():
    velocity = np.array([-1e-3, -1e-9, 1e-40, 1e-9, 1e-3])
    theta = np.full(velocity.shape, 0.008 / 1e-9)
    a = np.full(velocity.shape, 0.01)
    b = np.full(velocity.shape, 0.015)
    actual = signed_rate_state_friction_coefficient_profile(
        velocity, theta, a=a, b=b, mu0=0.6, V0=1e-6, L=0.008
    )
    expected = np.array([
        np.sign(value) * rate_state_friction_coefficient(
            abs(value), theta[index], a=a[index], b=b[index],
            mu0=0.6, V0=1e-6, L=0.008,
        )
        for index, value in enumerate(velocity)
    ])
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=0.0)


def test_low_rank_nucleation_modes_match_full_fault_operator():
    params = build_small_bp3_parameters()
    full = build_fault_traction_response(params)
    basis = localized_gaussian_basis(
        full.y, top=2e3, bottom=7e3, spacing=2e3, width=1e3
    )
    low_rank = solve_fault_mode_response(params, basis)

    np.testing.assert_allclose(
        low_rank.tau, full.tau @ basis, rtol=2e-9, atol=2e-3
    )
    np.testing.assert_allclose(
        low_rank.sigma_effective,
        full.sigma_effective @ basis,
        rtol=2e-9,
        atol=2e-3,
    )


def test_reduced_nucleation_stiffness_modes_satisfy_eigenproblem():
    params = build_small_bp3_parameters()
    y = Grid(params).y
    basis = localized_gaussian_basis(
        y, top=2e3, bottom=7e3, spacing=1e3, width=0.75e3
    )
    response = solve_fault_mode_response(params, basis)
    mu = np.full(params.Ny, params.mu0)
    kc = np.full(
        params.Ny,
        critical_stiffness(
            sigma0=params.sigma0,
            a=params.a0,
            b=params.b0,
            L=params.L,
        ),
    )
    result = diagnose_nucleation_stiffness(
        response,
        friction_coefficient=mu,
        critical_stiffness_profile=kc,
    )

    assert np.all(np.diff(result.stiffness_ratios) >= 0.0)
    assert np.all(np.isfinite(result.stiffness_ratios))
    assert np.all(result.spatial_modes[y < 2e3] == 0.0)
    assert np.all(result.spatial_modes[y >= 7e3] == 0.0)
    for index, ratio in enumerate(result.stiffness_ratios):
        coefficients = result.coefficient_modes[:, index]
        residual = (
            result.elastic_stiffness_matrix @ coefficients
            - ratio * result.critical_stiffness_matrix @ coefficients
        )
        scale = max(
            np.linalg.norm(
                result.elastic_stiffness_matrix @ coefficients
            ),
            1.0,
        )
        assert np.linalg.norm(residual) / scale < 2e-10


def test_condensed_fault_response_matches_full_2d_solve():
    """The fast model must preserve the current interface and boundary rows."""

    params = build_small_bp3_parameters()
    response = build_fault_traction_response(params)
    rng = np.random.default_rng(7351)
    fault_rate = rng.normal(size=params.Ny)
    # Avoid testing endpoint trace conventions in this bulk operator identity.
    fault_rate[[0, -1]] = 0.0

    predicted_tau, predicted_sigma = response.apply(fault_rate)
    direct_tau, direct_sigma = solve_fault_traction_response(params, fault_rate)
    np.testing.assert_allclose(predicted_tau, direct_tau, rtol=2e-9, atol=2e-3)
    np.testing.assert_allclose(
        predicted_sigma, direct_sigma, rtol=2e-9, atol=2e-3
    )


def test_complete_creep_loading_has_zero_static_traction_rate():
    """Matched plate/fault rates must remain a stress-free rigid mode."""

    params = build_small_bp3_parameters()
    params.bc.left.uy.set_velocity(-0.5 * params.loading.V_p)
    params.bc.right.uy.set_velocity(0.5 * params.loading.V_p)
    response = solve_fault_loading_response(
        params, np.full(params.Ny, params.loading.V_p)
    )

    scale = params.G * params.loading.V_p / np.min(np.diff(response.y))
    assert np.max(np.abs(response.tau_rate)) / scale < 2e-8
    assert np.max(np.abs(response.sigma_effective_rate)) / scale < 2e-8


def test_near_critical_operator_can_decay_a_small_peak_and_run_away():
    """Modes on opposite sides of critical stiffness take opposite branches."""

    params = build_small_bp3_parameters()
    response = build_fault_traction_response(params)
    plate_rate = params.loading.V_p
    mu_ss = rate_state_friction_coefficient(
        plate_rate,
        params.L / plate_rate,
        a=params.a0,
        b=params.b0,
        mu0=params.mu0,
        V0=params.V0,
        L=params.L,
    )
    kc = critical_stiffness(
        sigma0=params.sigma0, a=params.a0, b=params.b0, L=params.L
    )

    candidates = []
    for width in np.geomspace(0.05e3, 8.0e3, 72):
        mode = gaussian_fault_mode(response.y, centre=4.0e3, width=width)
        modal = response.project_mode(mode, mu_ss)
        candidates.append((modal.effective_stiffness / kc, modal))

    stable_ratio, stable_mode = min(
        candidates, key=lambda item: abs(item[0] - 1.2)
    )
    unstable_ratio, unstable_mode = min(
        candidates, key=lambda item: abs(item[0] - 0.8)
    )
    assert 1.1 < stable_ratio < 1.3
    assert 0.7 < unstable_ratio < 0.9

    stable = simulate_modal_pulse(
        stable_mode, params, initial_velocity_factor=10.0
    )
    unstable = simulate_modal_pulse(
        unstable_mode, params, initial_velocity_factor=10.0
    )

    assert stable.peak_velocity >= 5.0 * plate_rate
    assert stable.decayed
    assert unstable.became_event
