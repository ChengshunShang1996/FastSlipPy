"""History-only state and driving-rate diagnostics for SEAS BP3-QD.

The quasidynamic rate-and-state balance, written with positive shear-traction
magnitude ``T`` and effective normal stress ``sigma``, is

    T = sigma * f(V, theta) + eta * V.

Differentiating this identity separates logarithmic velocity acceleration into
shear-loading, normal-stress, and state-evolution contributions.  This is a
useful diagnostic near a small-peak bifurcation because it requires only saved
fault histories and makes no assumption about a steady interseismic path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EventWindow:
    """Inclusive indices for one threshold-defined event."""

    start: int
    peak: int
    end: int


@dataclass(frozen=True)
class LogVelocityBudget:
    """Terms in the differentiated rate-and-state force balance, in 1/s."""

    friction_coefficient: np.ndarray
    denominator_pa: np.ndarray
    observed: np.ndarray
    shear_loading: np.ndarray
    normal_stress: np.ndarray
    state_evolution: np.ndarray
    predicted: np.ndarray
    closure_residual: np.ndarray


@dataclass(frozen=True)
class FrozenAccelerationBudget:
    """Instantaneous acceleration terms for one frozen fault state."""

    friction_coefficient: np.ndarray
    direct_effect_coefficient: np.ndarray
    denominator_pa: np.ndarray
    shear_loading: np.ndarray
    normal_stress: np.ndarray
    state_evolution: np.ndarray
    predicted: np.ndarray


def event_windows(velocity: np.ndarray, threshold: float) -> list[EventWindow]:
    """Return contiguous windows where the maximum absolute rate is active.

    ``velocity`` may be a single station history ``(nt,)`` or a fault history
    ``(ny, nt)``.  Windows use inclusive start/end indices.
    """

    rate = np.asarray(velocity, dtype=float)
    if rate.ndim == 1:
        activity = np.abs(rate)
    elif rate.ndim == 2:
        activity = np.max(np.abs(rate), axis=0)
    else:
        raise ValueError("velocity must be one- or two-dimensional.")
    if activity.size == 0 or np.any(~np.isfinite(activity)):
        raise ValueError("velocity must contain a finite, non-empty history.")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold must be finite and positive.")

    active = activity >= threshold
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    ends = np.flatnonzero(active & ~np.r_[active[1:], False])
    return [
        EventWindow(
            start=int(start),
            peak=int(start + np.argmax(activity[start : end + 1])),
            end=int(end),
        )
        for start, end in zip(starts, ends)
    ]


def rate_state_log_velocity_budget(
    time: np.ndarray,
    velocity: np.ndarray,
    shear_traction: np.ndarray,
    normal_stress: np.ndarray,
    theta: np.ndarray,
    *,
    a: float,
    b: float,
    mu0: float,
    V0: float,
    L: float,
    eta: float,
) -> LogVelocityBudget:
    r"""Decompose ``d(log|V|)/dt`` using the quasidynamic force balance.

    The returned terms satisfy

    .. math::

       \frac{d\ln |V|}{dt} =
       \frac{\dot T - f\dot\sigma
       - b\sigma\,d\ln\theta/dt}{a\sigma + \eta |V|}.

    ``shear_traction`` must be the positive traction magnitude.  Thus, for the
    current FastSlipPy NPZ convention it is ``taum``; for a BP3/DFRA ASCII
    file it is minus column 4.  Normal stress is compression-positive.
    """

    arrays = [
        np.asarray(value, dtype=float)
        for value in (time, velocity, shear_traction, normal_stress, theta)
    ]
    time_array, rate, traction, sigma, state = arrays
    if any(value.ndim != 1 for value in arrays):
        raise ValueError("all histories must be one-dimensional.")
    if any(value.shape != time_array.shape for value in arrays[1:]):
        raise ValueError("all histories must have the same shape.")
    if time_array.size < 3:
        raise ValueError("at least three snapshots are required.")
    if np.any(~np.isfinite(np.concatenate(arrays))):
        raise ValueError("histories must contain only finite values.")
    if np.any(np.diff(time_array) <= 0.0):
        raise ValueError("time must be strictly increasing.")
    if np.any(sigma <= 0.0) or np.any(state <= 0.0):
        raise ValueError("normal_stress and theta must be positive.")
    constants = (a, b, V0, L, eta)
    if any(not np.isfinite(value) or value <= 0.0 for value in constants):
        raise ValueError("a, b, V0, L, and eta must be finite and positive.")
    if not np.isfinite(mu0):
        raise ValueError("mu0 must be finite.")

    speed = np.maximum(np.abs(rate), np.finfo(float).tiny)
    log_speed = np.log(speed)
    log_state = np.log(state)
    friction = mu0 + a * np.log(speed / V0) + b * np.log(V0 * state / L)
    denominator = a * sigma + eta * speed

    edge_order = 2
    observed = np.gradient(log_speed, time_array, edge_order=edge_order)
    traction_rate = np.gradient(traction, time_array, edge_order=edge_order)
    sigma_rate = np.gradient(sigma, time_array, edge_order=edge_order)
    log_state_rate = np.gradient(log_state, time_array, edge_order=edge_order)
    shear_term = traction_rate / denominator
    normal_term = -friction * sigma_rate / denominator
    state_term = -b * sigma * log_state_rate / denominator
    predicted = shear_term + normal_term + state_term
    return LogVelocityBudget(
        friction_coefficient=friction,
        denominator_pa=denominator,
        observed=observed,
        shear_loading=shear_term,
        normal_stress=normal_term,
        state_evolution=state_term,
        predicted=predicted,
        closure_residual=observed - predicted,
    )


def frozen_rate_state_acceleration(
    velocity: np.ndarray,
    theta: np.ndarray,
    normal_stress: np.ndarray,
    shear_traction_rate: np.ndarray,
    normal_stress_rate: np.ndarray,
    *,
    a: np.ndarray,
    b: np.ndarray,
    mu0: float,
    V0: float,
    L: float,
    eta: float,
) -> FrozenAccelerationBudget:
    r"""Return the exact instantaneous BP3 ``d(log|V|)/dt`` budget.

    This version uses the regularized-asinh friction law implemented by
    :class:`FaultState` and the aging law ``theta_dot = 1-|V|theta/L``.  The
    supplied traction rates may therefore be replaced by those from another
    spatial discretization without changing the frozen fault state.
    """

    values = [
        np.asarray(value, dtype=float)
        for value in (
            velocity, theta, normal_stress, shear_traction_rate,
            normal_stress_rate, a, b,
        )
    ]
    shape = values[0].shape
    if any(value.ndim != 1 or value.shape != shape for value in values):
        raise ValueError("all frozen-state arrays must be one-dimensional and match.")
    if shape[0] == 0 or np.any(~np.isfinite(np.concatenate(values))):
        raise ValueError("frozen-state arrays must be finite and non-empty.")
    rate, state, sigma, tau_rate, sigma_rate, direct, evolution = values
    if np.any(state <= 0.0) or np.any(sigma <= 0.0):
        raise ValueError("theta and normal_stress must be positive.")
    if np.any(direct <= 0.0) or np.any(evolution <= 0.0):
        raise ValueError("a and b must be positive.")
    constants = (V0, L, eta)
    if any(not np.isfinite(value) or value <= 0.0 for value in constants):
        raise ValueError("V0, L, and eta must be finite and positive.")
    if not np.isfinite(mu0):
        raise ValueError("mu0 must be finite.")

    speed = np.maximum(np.abs(rate), np.finfo(float).tiny)
    log_q = (
        np.log(speed / (2.0 * V0))
        + (mu0 + evolution * np.log(V0 * state / L)) / direct
    )
    large = log_q > 20.0
    asinh_q = np.empty_like(log_q)
    q_over_hypot = np.empty_like(log_q)
    asinh_q[large] = log_q[large] + np.log1p(
        np.sqrt(1.0 + np.exp(-2.0 * log_q[large]))
    )
    q_over_hypot[large] = 1.0 / np.sqrt(
        1.0 + np.exp(-2.0 * log_q[large])
    )
    q = np.exp(np.clip(log_q[~large], -745.0, 20.0))
    asinh_q[~large] = np.arcsinh(q)
    q_over_hypot[~large] = q / np.hypot(1.0, q)

    friction = direct * asinh_q
    direct_log_derivative = direct * q_over_hypot
    state_log_derivative = evolution * q_over_hypot
    denominator = sigma * direct_log_derivative + eta * speed
    theta_log_rate = 1.0 / state - speed / L
    shear_term = tau_rate / denominator
    normal_term = -friction * sigma_rate / denominator
    state_term = -sigma * state_log_derivative * theta_log_rate / denominator
    return FrozenAccelerationBudget(
        friction_coefficient=friction,
        direct_effect_coefficient=direct_log_derivative,
        denominator_pa=denominator,
        shear_loading=shear_term,
        normal_stress=normal_term,
        state_evolution=state_term,
        predicted=shear_term + normal_term + state_term,
    )
