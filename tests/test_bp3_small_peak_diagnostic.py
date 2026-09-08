"""Fast nonlinear small-peak checks driven by the actual BP3 elastic operator."""

import numpy as np

from examples.run_bp3_small_peak_diagnostic import build_small_bp3_parameters
from fastslippy.utilities.bp3_small_peak import (
    build_fault_traction_response,
    critical_stiffness,
    gaussian_fault_mode,
    rate_state_friction_coefficient,
    simulate_modal_pulse,
    solve_fault_traction_response,
)


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
