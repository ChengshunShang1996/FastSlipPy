"""Fast, mechanism-preserving diagnostics for BP3-QD small transients.

The two-dimensional elastic problem is linear.  Its effect on the nonlinear
fault can therefore be condensed into two matrices that map fault slip rate to
quasi-static shear-traction rate and effective-normal-stress rate.  Constructing
the matrices costs one sparse factorisation and one backsolve per fault node;
subsequent fault-only experiments are inexpensive.

Unlike a generic spring-slider model, these matrices retain the dip angle,
fault-interface rows, free surfaces, finite boundaries, and mesh stretching of
the supplied :class:`~fastslippy.pre_processing.model_parameters.ModelParameters`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.sparse.linalg import factorized

from fastslippy.pre_processing.grid import Grid
from fastslippy.pre_processing.model_parameters import CaseType, ModelParameters
from fastslippy.solver.matrix_builder import MatrixBuilder
from fastslippy.utilities.stress_cal_util import StressCalUtil


@dataclass(frozen=True)
class FaultTractionResponse:
    """Fault traction-rate response to an arbitrary nodal slip-rate vector.

    ``tau`` and ``sigma_effective`` have shape ``(Ny, Ny)``.  Column ``j`` is
    the response to a unit slip rate at fault node ``j`` with all non-fault
    boundary data held fixed.  Units are Pa/m because a velocity in m/s maps
    to a stress rate in Pa/s.
    """

    y: np.ndarray
    tau: np.ndarray
    sigma_effective: np.ndarray

    def apply(self, fault_rate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return shear and effective-normal-stress rates on the fault."""

        rate = np.asarray(fault_rate, dtype=float)
        if rate.shape != self.y.shape:
            raise ValueError(
                f"fault_rate must have shape {self.y.shape}, got {rate.shape}."
            )
        return self.tau @ rate, self.sigma_effective @ rate

    def project_mode(self, mode: np.ndarray, friction_coefficient: float):
        """Project the response onto one slip mode using depth quadrature."""

        phi = np.asarray(mode, dtype=float)
        if phi.shape != self.y.shape:
            raise ValueError(f"mode must have shape {self.y.shape}, got {phi.shape}.")
        if not np.any(phi):
            raise ValueError("mode must not be identically zero.")

        weights = _trapezoidal_node_weights(self.y)
        denominator = float(np.dot(weights, phi * phi))
        tau_coefficient = float(np.dot(weights * phi, self.tau @ phi) / denominator)
        sigma_coefficient = float(
            np.dot(weights * phi, self.sigma_effective @ phi) / denominator
        )
        effective_stiffness = -(
            tau_coefficient - friction_coefficient * sigma_coefficient
        )
        return ModalTractionResponse(
            tau_coefficient=tau_coefficient,
            sigma_coefficient=sigma_coefficient,
            effective_stiffness=effective_stiffness,
        )


@dataclass(frozen=True)
class FaultLoadingResponse:
    """Instantaneous fault traction rates for one complete loading state.

    Unlike :class:`FaultTractionResponse`, these arrays retain the response to
    the configured side-boundary velocities.  They are therefore the relevant
    quantities for comparing finite-domain and outer-mesh effects at a fixed
    fault slip-rate profile.
    """

    y: np.ndarray
    tau_rate: np.ndarray
    sigma_effective_rate: np.ndarray


@dataclass(frozen=True)
class ModalTractionResponse:
    """Scalar coefficients for one projected fault-slip mode."""

    tau_coefficient: float
    sigma_coefficient: float
    effective_stiffness: float


@dataclass(frozen=True)
class PulseResult:
    """Time history and classification of a near-critical modal pulse."""

    time: np.ndarray
    velocity: np.ndarray
    theta: np.ndarray
    relative_slip: np.ndarray
    peak_velocity: float
    final_velocity: float
    event_threshold: float

    @property
    def became_event(self) -> bool:
        return self.peak_velocity >= self.event_threshold

    @property
    def decayed(self) -> bool:
        return (
            not self.became_event
            and self.final_velocity <= 0.1 * self.peak_velocity
        )


def _trapezoidal_node_weights(coordinates: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 1 or coordinates.size < 2:
        raise ValueError("coordinates must be a one-dimensional array of length >= 2.")
    spacing = np.diff(coordinates)
    if np.any(spacing <= 0.0):
        raise ValueError("coordinates must be strictly increasing.")
    weights = np.empty_like(coordinates)
    weights[0] = 0.5 * spacing[0]
    weights[-1] = 0.5 * spacing[-1]
    weights[1:-1] = 0.5 * (spacing[:-1] + spacing[1:])
    return weights


def _unpack_velocity(solution: np.ndarray, params: ModelParameters):
    vpx = np.reshape(
        solution[0::2], (params.Nx + 1, params.Ny + 1), order="C"
    ).T
    vpy = np.reshape(
        solution[1::2], (params.Nx + 1, params.Ny + 1), order="C"
    ).T
    return vpx[:, : params.Nx], vpy[: params.Ny, :]


def _fault_tractions(
    params: ModelParameters,
    grid: Grid,
    stress_util: StressCalUtil,
    solution: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    vx, vy = _unpack_velocity(solution, params)
    tau, sigma = stress_util.compute_stress_fields(
        vy,
        vx,
        grid.dx,
        grid.dy,
        params.lam,
        params.G,
        grid.cosa,
        grid.sina,
        params.Ny,
        params.Nx,
        x=grid.x,
        y=grid.y,
        xp=grid.xp,
        yp=grid.yp,
    )
    mid = params.Nx // 2
    left, right = stress_util.recover_fault_normal_stress(
        sigma,
        grid.x,
        grid.y,
        grid.xp,
        grid.yp,
        left_column=mid - 1,
        right_column=mid,
    )
    # FastSlipPy defines effective normal stress as sigma0 - sigma_qs.
    return tau[:, mid].copy(), -0.5 * (left + right)


def build_fault_traction_response(
    params: ModelParameters,
) -> FaultTractionResponse:
    """Condense the full 2-D elastic solve into fault traction matrices.

    Existing nonzero velocity boundary conditions are allowed.  Their response
    is removed column by column, so the returned matrices contain only the
    incremental response caused by fault slip rate.
    """

    if params.case_type != CaseType.CALIFORNIA:
        raise ValueError("The BP3 fault response requires case_type='california'.")

    grid = Grid(params)
    builder = MatrixBuilder(params, grid)
    solve = factorized(builder.build_LH().tocsc())
    stress_util = StressCalUtil(prefer_numba=False)

    zero_rate = np.zeros(params.Ny)
    base_solution = solve(builder.build_RH(0.0, zero_rate).copy())
    base_tau, base_sigma = _fault_tractions(
        params, grid, stress_util, base_solution
    )

    tau_response = np.empty((params.Ny, params.Ny), dtype=float)
    sigma_response = np.empty_like(tau_response)
    unit_rate = np.zeros(params.Ny)
    for node in range(params.Ny):
        unit_rate[node] = 1.0
        solution = solve(builder.build_RH(0.0, unit_rate).copy())
        tau, sigma = _fault_tractions(params, grid, stress_util, solution)
        tau_response[:, node] = tau - base_tau
        sigma_response[:, node] = sigma - base_sigma
        unit_rate[node] = 0.0

    return FaultTractionResponse(
        y=grid.y.copy(),
        tau=tau_response,
        sigma_effective=sigma_response,
    )


def solve_fault_traction_response(
    params: ModelParameters,
    fault_rate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Direct 2-D response used to validate the condensed matrices."""

    rate = np.asarray(fault_rate, dtype=float)
    if rate.shape != (params.Ny,):
        raise ValueError(f"fault_rate must have shape ({params.Ny},), got {rate.shape}.")
    grid = Grid(params)
    builder = MatrixBuilder(params, grid)
    solve = factorized(builder.build_LH().tocsc())
    stress_util = StressCalUtil(prefer_numba=False)
    base = solve(builder.build_RH(0.0, np.zeros(params.Ny)).copy())
    solution = solve(builder.build_RH(0.0, rate).copy())
    base_tau, base_sigma = _fault_tractions(params, grid, stress_util, base)
    tau, sigma = _fault_tractions(params, grid, stress_util, solution)
    return tau - base_tau, sigma - base_sigma


def solve_fault_loading_response(
    params: ModelParameters,
    fault_rate: np.ndarray,
) -> FaultLoadingResponse:
    """Solve one full BP3 loading state and return on-fault traction rates.

    The side-boundary contribution is deliberately retained.  This makes the
    result suitable for A/B comparisons in which the physical domain or the
    stretched outer mesh changes while the imposed plate and fault rates stay
    fixed.  Only one sparse factorisation/backsolve is required per mesh.
    """

    if params.case_type != CaseType.CALIFORNIA:
        raise ValueError("The BP3 loading response requires case_type='california'.")

    rate = np.asarray(fault_rate, dtype=float)
    if rate.shape != (params.Ny,):
        raise ValueError(f"fault_rate must have shape ({params.Ny},), got {rate.shape}.")

    grid = Grid(params)
    builder = MatrixBuilder(params, grid)
    solution = factorized(builder.build_LH().tocsc())(
        builder.build_RH(0.0, rate).copy()
    )
    tau_rate, sigma_effective_rate = _fault_tractions(
        params,
        grid,
        StressCalUtil(prefer_numba=False),
        solution,
    )
    return FaultLoadingResponse(
        y=grid.y.copy(),
        tau_rate=tau_rate,
        sigma_effective_rate=sigma_effective_rate,
    )


def gaussian_fault_mode(
    y: np.ndarray,
    *,
    centre: float,
    width: float,
) -> np.ndarray:
    """Return a smooth unit-amplitude slip mode away from endpoint artefacts."""

    if width <= 0.0:
        raise ValueError("width must be positive.")
    y = np.asarray(y, dtype=float)
    mode = np.exp(-0.5 * ((y - centre) / width) ** 2)
    # Surface/bottom values are boundary traces rather than independent modes.
    mode[[0, -1]] = 0.0
    return mode


def _asinh_exponential(log_value: float) -> float:
    if log_value > 20.0:
        return log_value + np.log1p(np.sqrt(1.0 + np.exp(-2.0 * log_value)))
    return float(np.arcsinh(np.exp(log_value)))


def rate_state_friction_coefficient(
    velocity: float,
    theta: float,
    *,
    a: float,
    b: float,
    mu0: float,
    V0: float,
    L: float,
) -> float:
    """Regularised BP3 rate-and-state coefficient for positive velocity."""

    if velocity <= 0.0 or theta <= 0.0:
        raise ValueError("velocity and theta must be positive.")
    exponent = (mu0 + b * np.log(V0 * theta / L)) / a
    log_argument = np.log(velocity) - np.log(2.0 * V0) + exponent
    return a * _asinh_exponential(float(log_argument))


def critical_stiffness(*, sigma0: float, a: float, b: float, L: float) -> float:
    """Aging-law critical stiffness ``sigma0 * (b-a) / L``."""

    return sigma0 * (b - a) / L


def simulate_modal_pulse(
    modal: ModalTractionResponse,
    params: ModelParameters,
    *,
    initial_velocity_factor: float = 100.0,
    duration_state_times: float = 12.0,
    event_threshold: float = 1.0e-3,
    step_fraction: float = 0.03,
) -> PulseResult:
    """Perturb steady sliding and test whether the transient decays or runs away.

    The modal relative slip ``x`` is measured from steady plate motion.  Hence
    ``x_dot = V - Vp`` and the steady loading contribution is removed exactly.
    The scalar elastic feedback retains both projected shear and normal stress.
    """

    Vp = abs(float(params.loading.V_p or params.Vi))
    if Vp <= 0.0:
        raise ValueError("A positive plate or initial slip-rate magnitude is required.")
    if initial_velocity_factor <= 1.0:
        raise ValueError("initial_velocity_factor must be greater than one.")
    if duration_state_times <= 0.0 or step_fraction <= 0.0:
        raise ValueError("duration_state_times and step_fraction must be positive.")

    a = float(params.a0)
    b = float(params.b0)
    theta_ss = params.L / Vp
    mu_ss = rate_state_friction_coefficient(
        Vp, theta_ss, a=a, b=b, mu0=params.mu0, V0=params.V0, L=params.L
    )
    tau_ss = params.sigma0 * mu_ss + params.eta * Vp

    target_velocity = initial_velocity_factor * Vp
    mu_target = rate_state_friction_coefficient(
        target_velocity,
        theta_ss,
        a=a,
        b=b,
        mu0=params.mu0,
        V0=params.V0,
        L=params.L,
    )
    target_tau = params.sigma0 * mu_target + params.eta * target_velocity
    denominator = modal.tau_coefficient - mu_target * modal.sigma_coefficient
    if denominator == 0.0:
        raise ValueError("The modal elastic feedback is singular.")
    relative_slip = (target_tau - tau_ss) / denominator
    if params.sigma0 + modal.sigma_coefficient * relative_slip <= 0.0:
        raise ValueError("The requested pulse would make effective normal stress non-positive.")

    def solve_velocity(x: float, theta: float) -> float:
        sigma = params.sigma0 + modal.sigma_coefficient * x
        tau = tau_ss + modal.tau_coefficient * x
        if sigma <= 0.0 or tau <= 0.0:
            raise RuntimeError("Modal pulse left the positive stress branch.")

        def residual(log_velocity: float) -> float:
            velocity = np.exp(log_velocity)
            mu = rate_state_friction_coefficient(
                velocity,
                theta,
                a=a,
                b=b,
                mu0=params.mu0,
                V0=params.V0,
                L=params.L,
            )
            return sigma * mu + params.eta * velocity - tau

        # Keep the log-space bracket far below all BP3 velocities without
        # relying on a subnormal round-trip through exp(), which can become
        # exactly zero on some platforms.
        lower = np.log(1.0e-300)
        upper_velocity = max(tau / params.eta, Vp)
        upper = np.log(upper_velocity)
        return float(np.exp(brentq(residual, lower, upper, xtol=1e-12, rtol=1e-12)))

    state_time = params.L / Vp
    final_time = duration_state_times * state_time
    time = 0.0
    theta = theta_ss
    dt = step_fraction * state_time / initial_velocity_factor

    times = [time]
    velocities = [solve_velocity(relative_slip, theta)]
    thetas = [theta]
    slips = [relative_slip]

    while time < final_time and velocities[-1] < event_threshold:
        velocity = velocities[-1]
        dt = min(
            1.2 * dt,
            step_fraction * params.L / max(velocity, Vp),
            0.02 * state_time,
            final_time - time,
        )
        relative_slip += (velocity - Vp) * dt

        state_exponent = velocity * dt / params.L
        theta = (
            params.L / velocity * (1.0 - np.exp(-state_exponent))
            + theta * np.exp(-state_exponent)
        )
        time += dt
        next_velocity = solve_velocity(relative_slip, theta)

        times.append(time)
        velocities.append(next_velocity)
        thetas.append(theta)
        slips.append(relative_slip)

        if len(times) > 200_000:
            raise RuntimeError("Modal pulse exceeded the integration-step safety limit.")

    velocity_array = np.asarray(velocities)
    return PulseResult(
        time=np.asarray(times),
        velocity=velocity_array,
        theta=np.asarray(thetas),
        relative_slip=np.asarray(slips),
        peak_velocity=float(np.max(velocity_array)),
        final_velocity=float(velocity_array[-1]),
        event_threshold=float(event_threshold),
    )
