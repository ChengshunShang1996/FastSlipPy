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
from scipy.linalg import eigh
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
class FaultModeResponse:
    """Elastic traction response to a small set of fault-slip modes.

    The mode and response arrays have shape ``(Ny, number_of_modes)``.  This
    low-rank form avoids constructing the full ``Ny x Ny`` fault operator on a
    production mesh.
    """

    y: np.ndarray
    modes: np.ndarray
    tau: np.ndarray
    sigma_effective: np.ndarray


@dataclass(frozen=True)
class NucleationStiffnessResult:
    """Reduced spatial nucleation modes and their stiffness ratios."""

    y: np.ndarray
    basis: np.ndarray
    mass_matrix: np.ndarray
    elastic_stiffness_matrix: np.ndarray
    critical_stiffness_matrix: np.ndarray
    stiffness_ratios: np.ndarray
    coefficient_modes: np.ndarray
    spatial_modes: np.ndarray
    tau_responses: np.ndarray
    sigma_effective_responses: np.ndarray
    coulomb_responses: np.ndarray
    friction_coefficient: np.ndarray
    critical_stiffness: np.ndarray
    antisymmetric_fraction: float


@dataclass(frozen=True)
class FaultReciprocityResult:
    """Weighted reciprocity metrics for a reduced fault-traction operator.

    Elastic reciprocity applies to the shear traction that is energetically
    conjugate to tangential fault slip.  It does not, in general, require the
    Coulomb combination ``tau - mu * sigma_effective`` to be symmetric.
    ``work_matrix[i, j]`` is the work of test mode ``i`` against the shear
    traction generated by trial mode ``j``.
    """

    work_matrix: np.ndarray
    symmetric_matrix: np.ndarray
    antisymmetric_matrix: np.ndarray
    antisymmetric_fraction: float
    maximum_pairwise_fraction: float


@dataclass(frozen=True)
class ModalHistoryProjection:
    """Projection of a fault-history perturbation onto spatial modes.

    ``coefficients`` has shape ``(number_of_modes, number_of_times)`` and
    retains the units of the supplied history.  ``captured_fraction`` is the
    fraction of the weighted squared norm represented by the modal subspace;
    it is zero at the reference snapshot, where the perturbation vanishes.
    """

    coefficients: np.ndarray
    captured_fraction: np.ndarray
    reference_index: int


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


def diagnose_fault_reciprocity(
    response: FaultModeResponse,
) -> FaultReciprocityResult:
    """Measure Betti reciprocity of a reduced shear-traction response.

    For trial modes ``Phi`` and their shear-traction responses ``T Phi``, a
    work-consistent elastic discretisation should satisfy

    ``Phi.T @ H @ (T Phi) == (Phi.T @ H @ (T Phi)).T``.

    The reported fractions use the symmetric work as their scale.  A small
    value is evidence for consistency only on the span of the supplied modes;
    localized corner modes must be included to test corner closures.
    """

    modes = np.asarray(response.modes, dtype=float)
    traction = np.asarray(response.tau, dtype=float)
    y = np.asarray(response.y, dtype=float)
    if modes.ndim != 2 or modes.shape[0] != y.size:
        raise ValueError("response modes must have shape (len(y), n_modes).")
    if traction.shape != modes.shape:
        raise ValueError("response tau must have the same shape as modes.")
    if np.any(~np.isfinite(modes)) or np.any(~np.isfinite(traction)):
        raise ValueError("response modes and tractions must be finite.")

    weights = _trapezoidal_node_weights(y)
    work = modes.T @ (weights[:, None] * traction)
    symmetric = 0.5 * (work + work.T)
    antisymmetric = 0.5 * (work - work.T)
    tiny = np.finfo(float).tiny
    antisymmetric_fraction = float(
        np.linalg.norm(antisymmetric)
        / max(np.linalg.norm(symmetric), tiny)
    )
    maximum_pairwise_fraction = float(
        np.max(np.abs(work - work.T))
        / max(np.max(np.abs(work)), tiny)
    )
    return FaultReciprocityResult(
        work_matrix=work,
        symmetric_matrix=symmetric,
        antisymmetric_matrix=antisymmetric,
        antisymmetric_fraction=antisymmetric_fraction,
        maximum_pairwise_fraction=maximum_pairwise_fraction,
    )


def project_fault_history_onto_modes(
    y: np.ndarray,
    modes: np.ndarray,
    history: np.ndarray,
    *,
    reference_index: int,
    metric_profile: np.ndarray | None = None,
) -> ModalHistoryProjection:
    """Project changes in an on-fault history onto a fixed modal subspace.

    The projection is a weighted least-squares fit of
    ``history[:, t] - history[:, reference_index]``.  A positive
    ``metric_profile`` can be supplied to use, for example, the critical
    weakening-energy metric associated with the stiffness modes.
    """

    coordinates = np.asarray(y, dtype=float)
    spatial_modes = np.asarray(modes, dtype=float)
    values = np.asarray(history, dtype=float)
    if spatial_modes.ndim != 2 or spatial_modes.shape[0] != coordinates.size:
        raise ValueError("modes must have shape (len(y), number_of_modes).")
    if values.ndim != 2 or values.shape[0] != coordinates.size:
        raise ValueError("history must have shape (len(y), number_of_times).")
    if not 0 <= reference_index < values.shape[1]:
        raise ValueError("reference_index is outside the history.")
    if np.any(~np.isfinite(spatial_modes)) or np.any(~np.isfinite(values)):
        raise ValueError("modes and history must be finite.")

    weights = _trapezoidal_node_weights(coordinates)
    if metric_profile is not None:
        metric = np.asarray(metric_profile, dtype=float)
        if metric.shape != coordinates.shape or np.any(~np.isfinite(metric)):
            raise ValueError("metric_profile must be finite and match y.")
        support = np.any(spatial_modes != 0.0, axis=1)
        if np.any(metric[support] <= 0.0):
            raise ValueError("metric_profile must be positive on modal support.")
        weights = weights * np.where(support, metric, 0.0)

    gram = spatial_modes.T @ (weights[:, None] * spatial_modes)
    gram_eigenvalues = np.linalg.eigvalsh(0.5 * (gram + gram.T))
    tolerance = max(float(np.max(np.abs(gram_eigenvalues))), 1.0) * 1e-12
    if gram_eigenvalues[0] <= tolerance:
        raise ValueError("The supplied modes are linearly dependent in the metric.")

    perturbation = values - values[:, [reference_index]]
    coefficients = np.linalg.solve(
        gram,
        spatial_modes.T @ (weights[:, None] * perturbation),
    )
    reconstruction = spatial_modes @ coefficients
    total_norm_squared = np.sum(weights[:, None] * perturbation**2, axis=0)
    residual_norm_squared = np.sum(
        weights[:, None] * (perturbation - reconstruction) ** 2,
        axis=0,
    )
    captured = np.zeros(values.shape[1], dtype=float)
    nonzero = total_norm_squared > np.finfo(float).tiny
    captured[nonzero] = 1.0 - (
        residual_norm_squared[nonzero] / total_norm_squared[nonzero]
    )
    captured = np.clip(captured, 0.0, 1.0)
    return ModalHistoryProjection(
        coefficients=coefficients,
        captured_fraction=captured,
        reference_index=int(reference_index),
    )


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


def solve_fault_mode_response(
    params: ModelParameters,
    modes: np.ndarray,
) -> FaultModeResponse:
    """Apply the production elastic operator to a small modal basis.

    Nonzero external velocity boundaries are removed with one zero-slip solve,
    so every returned column is the incremental traction caused by that fault
    mode alone.  A factorization plus ``number_of_modes + 1`` backsolves is
    required.
    """
    if params.case_type != CaseType.CALIFORNIA:
        raise ValueError("The BP3 mode response requires case_type='california'.")
    basis = np.asarray(modes, dtype=float)
    if basis.ndim == 1:
        basis = basis[:, None]
    if basis.ndim != 2 or basis.shape[0] != params.Ny:
        raise ValueError(
            f"modes must have shape ({params.Ny}, n_modes), got {basis.shape}."
        )
    if basis.shape[1] == 0 or np.any(~np.isfinite(basis)):
        raise ValueError("modes must contain at least one finite column.")
    if np.any(np.linalg.norm(basis, axis=0) == 0.0):
        raise ValueError("Every mode must be nonzero.")
    if np.any(basis[[0, -1], :] != 0.0):
        raise ValueError("Fault endpoint trace values must be zero in every mode.")

    grid = Grid(params)
    builder = MatrixBuilder(params, grid)
    solve = factorized(builder.build_LH().tocsc())
    stress_util = StressCalUtil(prefer_numba=False)
    base_solution = solve(
        builder.build_RH(0.0, np.zeros(params.Ny)).copy()
    )
    base_tau, base_sigma = _fault_tractions(
        params, grid, stress_util, base_solution
    )

    tau_response = np.empty_like(basis)
    sigma_response = np.empty_like(basis)
    for column in range(basis.shape[1]):
        solution = solve(builder.build_RH(0.0, basis[:, column]).copy())
        tau, sigma = _fault_tractions(
            params, grid, stress_util, solution
        )
        tau_response[:, column] = tau - base_tau
        sigma_response[:, column] = sigma - base_sigma

    return FaultModeResponse(
        y=grid.y.copy(),
        modes=basis.copy(),
        tau=tau_response,
        sigma_effective=sigma_response,
    )


def localized_gaussian_basis(
    y: np.ndarray,
    *,
    top: float,
    bottom: float,
    spacing: float,
    width: float,
) -> np.ndarray:
    """Build truncated Gaussian basis functions inside a nucleation band."""
    coordinates = np.asarray(y, dtype=float)
    if not 0.0 <= top < bottom <= coordinates[-1]:
        raise ValueError("Require 0 <= top < bottom <= max(y).")
    if spacing <= 0.0 or width <= 0.0:
        raise ValueError("spacing and width must be positive.")
    centers = np.arange(top, bottom, spacing, dtype=float)
    if centers.size == 0:
        raise ValueError("The requested band contains no basis centres.")
    basis = np.exp(
        -0.5 * ((coordinates[:, None] - centers[None, :]) / width) ** 2
    )
    basis[(coordinates < top) | (coordinates >= bottom), :] = 0.0
    basis[[0, -1], :] = 0.0
    norms = np.max(np.abs(basis), axis=0)
    if np.any(norms == 0.0):
        raise ValueError("Grid resolution leaves an empty localized basis mode.")
    return basis / norms


def diagnose_nucleation_stiffness(
    response: FaultModeResponse,
    *,
    friction_coefficient: np.ndarray,
    critical_stiffness_profile: np.ndarray,
) -> NucleationStiffnessResult:
    """Solve the reduced spatial stiffness/weakening eigenproblem.

    For a displacement mode ``phi``, the incremental failure stress is
    ``delta_tau - mu * delta_sigma_effective``.  Its negative is the elastic
    restoring stiffness.  The generalized eigenvalue compares that stiffness
    with the local aging-law weakening rate ``sigma * (b-a) / L``.  Ratios
    below one identify spatial modes that are softer than this quasistatic
    nucleation criterion; they are a diagnostic, not by themselves proof of a
    dynamic instability.
    """
    y = np.asarray(response.y, dtype=float)
    basis = np.asarray(response.modes, dtype=float)
    mu = np.asarray(friction_coefficient, dtype=float)
    kc = np.asarray(critical_stiffness_profile, dtype=float)
    if mu.shape != y.shape or kc.shape != y.shape:
        raise ValueError("friction_coefficient and critical_stiffness must match y.")
    if np.any(~np.isfinite(mu)) or np.any(~np.isfinite(kc)):
        raise ValueError("Friction and critical-stiffness profiles must be finite.")

    weights = _trapezoidal_node_weights(y)
    weighted_basis = weights[:, None] * basis
    mass = basis.T @ weighted_basis
    coulomb = response.tau - mu[:, None] * response.sigma_effective
    stiffness_unsymmetric = -(basis.T @ (weights[:, None] * coulomb))
    stiffness = 0.5 * (stiffness_unsymmetric + stiffness_unsymmetric.T)
    critical = basis.T @ (
        weights[:, None] * kc[:, None] * basis
    )

    critical_eigenvalues = np.linalg.eigvalsh(critical)
    tolerance = max(
        np.max(np.abs(critical_eigenvalues)), 1.0
    ) * 1e-12
    if critical_eigenvalues[0] <= tolerance:
        raise ValueError(
            "The projected critical-stiffness matrix is not positive definite; "
            "restrict the basis to the velocity-weakening region."
        )
    ratios, coefficient_modes = eigh(stiffness, critical)
    spatial_modes = basis @ coefficient_modes
    scales = np.max(np.abs(spatial_modes), axis=0)
    spatial_modes /= scales
    coefficient_modes /= scales
    for column in range(spatial_modes.shape[1]):
        peak = int(np.argmax(np.abs(spatial_modes[:, column])))
        if spatial_modes[peak, column] < 0.0:
            spatial_modes[:, column] *= -1.0
            coefficient_modes[:, column] *= -1.0

    tau_modes = response.tau @ coefficient_modes
    sigma_modes = response.sigma_effective @ coefficient_modes
    coulomb_modes = tau_modes - mu[:, None] * sigma_modes
    antisymmetric = 0.5 * (
        stiffness_unsymmetric - stiffness_unsymmetric.T
    )
    antisymmetric_fraction = float(
        np.linalg.norm(antisymmetric)
        / max(np.linalg.norm(stiffness), np.finfo(float).tiny)
    )
    return NucleationStiffnessResult(
        y=y.copy(),
        basis=basis.copy(),
        mass_matrix=mass,
        elastic_stiffness_matrix=stiffness,
        critical_stiffness_matrix=critical,
        stiffness_ratios=ratios,
        coefficient_modes=coefficient_modes,
        spatial_modes=spatial_modes,
        tau_responses=tau_modes,
        sigma_effective_responses=sigma_modes,
        coulomb_responses=coulomb_modes,
        friction_coefficient=mu.copy(),
        critical_stiffness=kc.copy(),
        antisymmetric_fraction=antisymmetric_fraction,
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


def signed_rate_state_friction_coefficient_profile(
    velocity: np.ndarray,
    theta: np.ndarray,
    *,
    a: np.ndarray,
    b: np.ndarray,
    mu0: float,
    V0: float,
    L: float,
) -> np.ndarray:
    """Vectorized signed BP3 friction coefficient without overflow."""
    velocity = np.asarray(velocity, dtype=float)
    theta = np.asarray(theta, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if not (velocity.shape == theta.shape == a.shape == b.shape):
        raise ValueError("velocity, theta, a, and b must have identical shapes.")
    if np.any(theta <= 0.0) or np.any(a <= 0.0):
        raise ValueError("theta and a must be positive.")
    speed = np.maximum(np.abs(velocity), np.finfo(float).tiny)
    exponent = (mu0 + b * np.log(V0 * theta / L)) / a
    log_argument = np.log(speed / (2.0 * V0)) + exponent
    asinh_argument = np.empty_like(log_argument)
    large = log_argument > 20.0
    asinh_argument[large] = (
        log_argument[large]
        + np.log1p(np.sqrt(1.0 + np.exp(-2.0 * log_argument[large])))
    )
    asinh_argument[~large] = np.arcsinh(np.exp(log_argument[~large]))
    return np.sign(velocity) * a * asinh_argument


def signed_rate_state_friction_derivatives_profile(
    velocity: np.ndarray,
    theta: np.ndarray,
    *,
    a: np.ndarray,
    b: np.ndarray,
    mu0: float,
    V0: float,
    L: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``f``, ``df/dV`` and ``df/dtheta`` for the BP3 friction law."""

    velocity = np.asarray(velocity, dtype=float)
    theta = np.asarray(theta, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if not (velocity.shape == theta.shape == a.shape == b.shape):
        raise ValueError("velocity, theta, a, and b must have identical shapes.")
    if np.any(velocity == 0.0):
        raise ValueError("The friction derivatives require nonzero velocity.")
    if np.any(theta <= 0.0) or np.any(a <= 0.0) or V0 <= 0.0 or L <= 0.0:
        raise ValueError("theta, a, V0, and L must be positive.")

    speed = np.abs(velocity)
    log_q_abs = (
        np.log(speed / (2.0 * V0))
        + (mu0 + b * np.log(V0 * theta / L)) / a
    )
    q_over_hypot = np.empty_like(log_q_abs)
    nonnegative = log_q_abs >= 0.0
    q_over_hypot[nonnegative] = 1.0 / np.sqrt(
        1.0 + np.exp(-2.0 * log_q_abs[nonnegative])
    )
    q_over_hypot[~nonnegative] = np.exp(log_q_abs[~nonnegative]) / np.sqrt(
        1.0 + np.exp(2.0 * log_q_abs[~nonnegative])
    )
    q_over_hypot *= np.sign(velocity)
    friction = signed_rate_state_friction_coefficient_profile(
        velocity,
        theta,
        a=a,
        b=b,
        mu0=mu0,
        V0=V0,
        L=L,
    )
    derivative_velocity = a * q_over_hypot / velocity
    derivative_theta = b * q_over_hypot / theta
    return friction, derivative_velocity, derivative_theta


def reduced_rate_state_jacobian(
    y: np.ndarray,
    modes: np.ndarray,
    tau_responses: np.ndarray,
    sigma_effective_responses: np.ndarray,
    *,
    velocity: np.ndarray,
    theta: np.ndarray,
    sigma_effective: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    mu0: float,
    V0: float,
    L: float,
    eta: float,
    metric_profile: np.ndarray | None = None,
) -> np.ndarray:
    r"""Linearize the coupled aging-law dynamics in a spatial modal basis.

    The state is ``[delta slip coefficients, delta theta coefficients]``.
    The algebraic radiation-damped friction equation is differentiated first,
    then the resulting velocity perturbation is inserted into
    ``delta U dot = delta V`` and
    ``theta dot = 1 - abs(V) theta / L``.

    This is an instantaneous reduced Jacobian along a non-steady trajectory.
    Its eigenvalues diagnose local growth but do not replace finite-time
    integration of the time-dependent tangent system.
    """

    coordinates = np.asarray(y, dtype=float)
    spatial_modes = np.asarray(modes, dtype=float)
    tau_modes = np.asarray(tau_responses, dtype=float)
    sigma_modes = np.asarray(sigma_effective_responses, dtype=float)
    expected = spatial_modes.shape
    if spatial_modes.ndim != 2 or spatial_modes.shape[0] != coordinates.size:
        raise ValueError("modes must have shape (len(y), number_of_modes).")
    if tau_modes.shape != expected or sigma_modes.shape != expected:
        raise ValueError("traction responses must have the same shape as modes.")

    profiles = [velocity, theta, sigma_effective, a, b]
    velocity, theta, sigma_effective, a, b = (
        np.asarray(profile, dtype=float) for profile in profiles
    )
    if any(profile.shape != coordinates.shape for profile in (
        velocity, theta, sigma_effective, a, b
    )):
        raise ValueError("All state and friction profiles must match y.")
    if np.any(sigma_effective <= 0.0) or eta < 0.0:
        raise ValueError("Effective normal stress must be positive and eta nonnegative.")

    weights = _trapezoidal_node_weights(coordinates)
    if metric_profile is not None:
        metric = np.asarray(metric_profile, dtype=float)
        if metric.shape != coordinates.shape:
            raise ValueError("metric_profile must match y.")
        support = np.any(spatial_modes != 0.0, axis=1)
        if np.any(~np.isfinite(metric)) or np.any(metric[support] <= 0.0):
            raise ValueError("metric_profile must be finite and positive on support.")
        weights *= np.where(support, metric, 0.0)
    gram = spatial_modes.T @ (weights[:, None] * spatial_modes)
    projector = np.linalg.solve(gram, spatial_modes.T * weights)

    friction, friction_velocity, friction_theta = (
        signed_rate_state_friction_derivatives_profile(
            velocity,
            theta,
            a=a,
            b=b,
            mu0=mu0,
            V0=V0,
            L=L,
        )
    )
    algebraic_denominator = sigma_effective * friction_velocity + eta
    if np.any(algebraic_denominator <= 0.0):
        raise ValueError("The linearized algebraic friction slope is not positive.")

    slip_to_velocity = (
        tau_modes - friction[:, None] * sigma_modes
    ) / algebraic_denominator[:, None]
    theta_to_velocity = -(
        sigma_effective * friction_theta / algebraic_denominator
    )[:, None] * spatial_modes

    slip_rate_from_slip = projector @ slip_to_velocity
    slip_rate_from_theta = projector @ theta_to_velocity
    aging_velocity_slope = -(theta / L) * np.sign(velocity)
    aging_theta_slope = -np.abs(velocity) / L
    theta_rate_from_slip = projector @ (
        aging_velocity_slope[:, None] * slip_to_velocity
    )
    theta_rate_from_theta = projector @ (
        aging_velocity_slope[:, None] * theta_to_velocity
        + aging_theta_slope[:, None] * spatial_modes
    )
    return np.block([
        [slip_rate_from_slip, slip_rate_from_theta],
        [theta_rate_from_slip, theta_rate_from_theta],
    ])


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
