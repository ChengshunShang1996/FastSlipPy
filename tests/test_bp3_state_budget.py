import numpy as np
import pytest

from fastslippy.utilities.bp3_state_budget import (
    event_windows,
    frozen_rate_state_acceleration,
    rate_state_log_velocity_budget,
)


def test_event_windows_support_station_and_fault_histories():
    station = np.array([0.0, 2.0, 3.0, 0.0, 0.0, 4.0, 0.0])
    expected = [(1, 2, 2), (5, 5, 5)]
    assert [
        (event.start, event.peak, event.end)
        for event in event_windows(station, 1.0)
    ] == expected
    fault = np.vstack((0.5 * station, station))
    assert [
        (event.start, event.peak, event.end)
        for event in event_windows(fault, 1.0)
    ] == expected


def test_rate_state_budget_closes_for_an_exact_synthetic_balance():
    time = np.linspace(0.0, 20.0, 2001)
    growth = 0.08
    velocity = 1e-9 * np.exp(growth * time)
    theta = np.full(time.shape, 2e6)
    sigma = np.full(time.shape, 50e6)
    a, b, mu0, V0, L, eta = 0.01, 0.015, 0.6, 1e-6, 0.008, 4.6e6
    friction = mu0 + a * np.log(velocity / V0) + b * np.log(V0 * theta / L)
    traction = sigma * friction + eta * velocity

    result = rate_state_log_velocity_budget(
        time, velocity, traction, sigma, theta,
        a=a, b=b, mu0=mu0, V0=V0, L=L, eta=eta,
    )

    interior = slice(2, -2)
    np.testing.assert_allclose(result.observed[interior], growth, rtol=1e-9)
    np.testing.assert_allclose(result.predicted[interior], growth, rtol=2e-7)
    np.testing.assert_allclose(result.normal_stress[interior], 0.0, atol=2e-12)
    np.testing.assert_allclose(result.state_evolution[interior], 0.0, atol=2e-12)


def test_rate_state_budget_rejects_nonmonotonic_time():
    values = np.ones(3)
    with pytest.raises(ValueError, match="strictly increasing"):
        rate_state_log_velocity_budget(
            np.array([0.0, 1.0, 1.0]), values, values, values, values,
            a=0.01, b=0.015, mu0=0.6, V0=1e-6, L=0.008, eta=1.0,
        )


def test_frozen_acceleration_uses_aging_law_and_elastic_rates():
    velocity = np.array([1e-10, 2e-9, 1e-7])
    theta = np.array([4e7, 3e6, 2e4])
    sigma = np.full(3, 50e6)
    tau_rate = np.array([-1e-3, 2e-3, -4e-3])
    sigma_rate = np.array([1e-5, -2e-5, 3e-5])
    a = np.full(3, 0.01)
    b = np.full(3, 0.015)

    result = frozen_rate_state_acceleration(
        velocity, theta, sigma, tau_rate, sigma_rate,
        a=a, b=b, mu0=0.6, V0=1e-6, L=0.008, eta=4.6e6,
    )

    np.testing.assert_allclose(
        result.predicted,
        result.shear_loading + result.normal_stress + result.state_evolution,
    )
    assert np.all(result.denominator_pa > 0.0)
    assert np.all(result.direct_effect_coefficient > 0.0)
    assert np.all(np.isfinite(result.predicted))
