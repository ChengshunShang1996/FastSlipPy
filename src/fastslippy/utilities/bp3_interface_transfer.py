"""Static BP3 diagnostics for fault endpoints and normal-traction recovery.

The production stress path first evaluates normal stress at cell centres and
then recovers it to fault nodes.  This module supplies an independent check
that differentiates the staggered displacement fields directly at the fault
trace, using separate one-sided stencils on the two fault faces.  Agreement is
expected under refinement, but neither recovery is declared an exact discrete
reaction traction by this diagnostic.

The same factorized elastic operator is also used to measure how perturbations
near the surface, the rate-state/creep junction, and the deep boundary load the
8--15 km nucleation region.  A separate A/B solve changes only whether the first
resolved node at or below ``W_f`` belongs to the prescribed creeping segment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import factorized

from fastslippy.pre_processing.grid import Grid
from fastslippy.pre_processing.model_parameters import CaseType, ModelParameters
from fastslippy.solver.matrix_builder import MatrixBuilder
from fastslippy.utilities.bp3_small_peak import _fault_tractions, _unpack_velocity
from fastslippy.utilities.grid_operators import finite_difference_weights
from fastslippy.utilities.stress_cal_util import StressCalUtil


@dataclass(frozen=True)
class NormalTractionRecovery:
    """Normal-stress perturbation on the two faces of the fault.

    The sign is the stress-component convention used by ``sigmaqs``; effective
    compressive normal-stress perturbation is minus the average of the faces.
    """

    left: np.ndarray
    right: np.ndarray

    @property
    def effective_average(self) -> np.ndarray:
        return -0.5 * (self.left + self.right)


@dataclass(frozen=True)
class FaultTransferResponse:
    """Low-rank endpoint-to-nucleation traction-transfer response."""

    y: np.ndarray
    source_labels: tuple[str, ...]
    source_modes: np.ndarray
    receiver_labels: tuple[str, ...]
    receiver_modes: np.ndarray
    tau: np.ndarray
    sigma_effective_current: np.ndarray
    sigma_effective_direct: np.ndarray
    projected_tau: np.ndarray
    projected_sigma_effective_current: np.ndarray
    projected_sigma_effective_direct: np.ndarray
    projected_coulomb_current: np.ndarray
    projected_coulomb_direct: np.ndarray


@dataclass(frozen=True)
class WfEndpointLoadingComparison:
    """Full loading response for two first-creep-node conventions.

    ``production_rate`` starts creep at ``searchsorted(y, W_f, left)``.  The
    alternative keeps that first node on the rate-state side and starts the
    prescribed segment one node deeper.  This remains well-defined when a
    stretched grid does not place a node exactly at ``W_f``.
    """

    production_rate: np.ndarray
    rs_endpoint_rate: np.ndarray
    tau_production: np.ndarray
    tau_rs_endpoint: np.ndarray
    sigma_effective_current_production: np.ndarray
    sigma_effective_current_rs_endpoint: np.ndarray
    sigma_effective_direct_production: np.ndarray
    sigma_effective_direct_rs_endpoint: np.ndarray

    @property
    def delta_tau(self) -> np.ndarray:
        return self.tau_rs_endpoint - self.tau_production

    @property
    def delta_sigma_effective_current(self) -> np.ndarray:
        return (
            self.sigma_effective_current_rs_endpoint
            - self.sigma_effective_current_production
        )

    @property
    def delta_sigma_effective_direct(self) -> np.ndarray:
        return (
            self.sigma_effective_direct_rs_endpoint
            - self.sigma_effective_direct_production
        )


@dataclass(frozen=True)
class BP3InterfaceTransferResult:
    """Combined independent-recovery, endpoint-transfer, and ``W_f`` A/B test."""

    transfer: FaultTransferResponse
    wf_loading: WfEndpointLoadingComparison


def _nearest_stencil(
    coordinates: np.ndarray,
    target: float,
    *,
    count: int = 3,
    allowed: np.ndarray | None = None,
) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=float)
    candidates = (
        np.arange(coords.size, dtype=int)
        if allowed is None
        else np.asarray(allowed, dtype=int)
    )
    if candidates.size < count:
        raise ValueError(f"need at least {count} points for a one-sided stencil")
    nearest = candidates[np.argsort(np.abs(coords[candidates] - target))[:count]]
    return np.sort(nearest)


def _evaluate_rows_at_targets(
    values: np.ndarray,
    source_coordinates: np.ndarray,
    targets: np.ndarray,
    *,
    derivative: int,
) -> np.ndarray:
    """Evaluate one 1-D sampled field at arbitrary targets with three points."""

    field = np.asarray(values, dtype=float)
    source = np.asarray(source_coordinates, dtype=float)
    target_values = np.asarray(targets, dtype=float)
    if field.shape != source.shape:
        raise ValueError("values and source_coordinates must have identical shape")
    result = np.empty(target_values.size, dtype=float)
    for index, target in enumerate(target_values):
        stencil = _nearest_stencil(source, float(target))
        weights = finite_difference_weights(
            float(target), source[stencil], derivative
        )
        result[index] = float(weights @ field[stencil])
    return result


def _differentiate_columns_at_nodes(
    values: np.ndarray, coordinates: np.ndarray
) -> np.ndarray:
    field = np.asarray(values, dtype=float)
    coords = np.asarray(coordinates, dtype=float)
    if field.shape[0] != coords.size:
        raise ValueError("the first field axis must match coordinates")
    derivative = np.empty_like(field)
    for index, target in enumerate(coords):
        stencil = _nearest_stencil(coords, float(target))
        weights = finite_difference_weights(
            float(target), coords[stencil], 1
        )
        derivative[index, :] = weights @ field[stencil, :]
    return derivative


def direct_fault_normal_stress(
    params: ModelParameters,
    grid: Grid,
    solution: np.ndarray,
) -> NormalTractionRecovery:
    """Recover both fault-face normal stresses directly at ``x=0``.

    Separate quadratic-exact one-sided x stencils are used on the two sides of
    the tangential displacement jump.  Derivatives in the along-fault
    direction are evaluated at the physical fault nodes, including one-sided
    endpoint stencils.  This is intentionally independent of the production
    cell-centre-to-node recovery.
    """

    ux, uy = _unpack_velocity(np.asarray(solution, dtype=float), params)
    mid = params.Nx // 2
    if mid < 2 or params.Nx - mid < 3 or params.Ny < 3:
        raise ValueError("the direct fault recovery requires at least 5 x nodes")

    x = np.asarray(grid.x, dtype=float)
    xp = np.asarray(grid.xp, dtype=float)
    y = np.asarray(grid.y, dtype=float)
    yp = np.asarray(grid.yp, dtype=float)
    x_fault = float(x[mid])

    left_x = _nearest_stencil(
        x, x_fault, allowed=np.arange(0, mid + 1, dtype=int)
    )
    right_x = _nearest_stencil(
        x, x_fault, allowed=np.arange(mid, params.Nx, dtype=int)
    )
    wx_left = finite_difference_weights(x_fault, x[left_x], 1)
    wx_right = finite_difference_weights(x_fault, x[right_x], 1)

    ux_x_left_yp = ux[:, left_x] @ wx_left
    ux_x_right_yp = ux[:, right_x] @ wx_right
    ux_x_left = _evaluate_rows_at_targets(
        ux_x_left_yp, yp, y, derivative=0
    )
    ux_x_right = _evaluate_rows_at_targets(
        ux_x_right_yp, yp, y, derivative=0
    )

    ux_y_fault = _evaluate_rows_at_targets(
        ux[:, mid], yp, y, derivative=1
    )

    uy_y = _differentiate_columns_at_nodes(uy, y)
    left_xp = _nearest_stencil(
        xp,
        x_fault,
        allowed=np.arange(0, mid + 1, dtype=int),
    )
    right_xp = _nearest_stencil(
        xp,
        x_fault,
        allowed=np.arange(mid + 1, params.Nx + 1, dtype=int),
    )
    wxp_left = finite_difference_weights(x_fault, xp[left_xp], 0)
    wxp_right = finite_difference_weights(x_fault, xp[right_xp], 0)
    uy_y_left = uy_y[:, left_xp] @ wxp_left
    uy_y_right = uy_y[:, right_xp] @ wxp_right

    common = -2.0 * params.G * grid.cosa * ux_y_fault
    left = (
        (params.lam + 2.0 * params.G) * ux_x_left
        + params.lam * uy_y_left
        + common
    )
    right = (
        (params.lam + 2.0 * params.G) * ux_x_right
        + params.lam * uy_y_right
        + common
    )
    return NormalTractionRecovery(left=left, right=right)


def _gaussian(y: np.ndarray, centre: float, width: float) -> np.ndarray:
    mode = np.exp(-0.5 * ((y - centre) / width) ** 2)
    scale = float(np.max(np.abs(mode)))
    if scale == 0.0:
        raise ValueError("Gaussian mode is not resolved by this fault grid")
    return mode / scale


def default_endpoint_source_modes(
    y: np.ndarray, w_f: float
) -> tuple[tuple[str, ...], np.ndarray]:
    """Return localized modes that separate the three interface endpoints."""

    coordinates = np.asarray(y, dtype=float)
    local_spacing = float(np.median(np.diff(coordinates)))
    narrow = max(0.75e3, 3.0 * local_spacing)
    bottom_width = max(2.0e3, 5.0 * local_spacing)

    surface = _gaussian(coordinates, 0.0, narrow)
    wf_rs = _gaussian(coordinates, w_f, narrow)
    wf_rs[coordinates > w_f] = 0.0
    wf_creep = _gaussian(coordinates, w_f, narrow)
    wf_creep[coordinates < w_f] = 0.0
    bottom = _gaussian(coordinates, coordinates[-1], bottom_width)
    rs_long_wave = np.sin(
        np.pi * np.minimum(coordinates, w_f) / w_f
    )
    rs_long_wave[coordinates >= w_f] = 0.0

    labels = (
        "surface",
        "wf_rs_side",
        "wf_creep_side",
        "bottom",
        "rs_long_wave",
    )
    modes = np.column_stack(
        (surface, wf_rs, wf_creep, bottom, rs_long_wave)
    )
    modes /= np.max(np.abs(modes), axis=0)
    return labels, modes


def default_nucleation_receiver_modes(
    y: np.ndarray,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Return positive averaging windows at the observed BP3 stations."""

    coordinates = np.asarray(y, dtype=float)
    width = max(0.75e3, 3.0 * float(np.median(np.diff(coordinates))))
    labels = ("depth_10km", "depth_11km", "depth_15km", "band_8_15km")
    band = np.zeros_like(coordinates)
    inside = (coordinates >= 8e3) & (coordinates <= 15e3)
    band[inside] = 1.0
    if not np.any(inside):
        raise ValueError("the grid does not resolve the 8--15 km receiver band")
    modes = np.column_stack(
        (
            _gaussian(coordinates, 10e3, width),
            _gaussian(coordinates, 11e3, width),
            _gaussian(coordinates, 15e3, width),
            band,
        )
    )
    return labels, modes


def _node_weights(y: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(y, dtype=float)
    weights = np.empty_like(coordinates)
    spacing = np.diff(coordinates)
    weights[0] = 0.5 * spacing[0]
    weights[-1] = 0.5 * spacing[-1]
    weights[1:-1] = 0.5 * (spacing[:-1] + spacing[1:])
    return weights


def deep_corner_source_modes(
    y: np.ndarray,
    w_f: float,
    *,
    fixed_deep_depth: float = 150e3,
    equivalent_width: float = 1e3,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Build work-comparable sources that isolate the deep fault corner.

    Every column is normalised so that ``integral(abs(mode), y)`` equals
    ``equivalent_width``.  Consequently, receiver projections can be compared
    directly without conflating source width with elastic transfer strength.

    The fixed-depth sources remain at the same physical depth when ``L_z`` is
    changed.  ``near_bottom_smooth`` instead follows the deep boundary while
    vanishing at the endpoint.  Comparing those modes separates source depth
    from proximity to the artificial traction-free boundary.
    """

    coordinates = np.asarray(y, dtype=float)
    if coordinates.ndim != 1 or coordinates.size < 5:
        raise ValueError("y must be a one-dimensional array with >= 5 nodes")
    if np.any(np.diff(coordinates) <= 0.0):
        raise ValueError("y must be strictly increasing")
    if not 0.0 < w_f < coordinates[-1]:
        raise ValueError("w_f must lie inside the fault grid")
    if equivalent_width <= 0.0 or not np.isfinite(equivalent_width):
        raise ValueError("equivalent_width must be finite and positive")

    def endpoint(index: int) -> np.ndarray:
        mode = np.zeros_like(coordinates)
        mode[index] = 1.0
        return mode

    def compact_cosine(centre: float, half_width: float) -> np.ndarray:
        distance = np.abs(coordinates - centre)
        mode = np.zeros_like(coordinates)
        inside = distance < half_width
        mode[inside] = 0.5 * (
            1.0 + np.cos(np.pi * distance[inside] / half_width)
        )
        return mode

    surface_width = min(2e3, 0.2 * coordinates[-1])
    surface_smooth = np.zeros_like(coordinates)
    surface_mask = coordinates < surface_width
    surface_smooth[surface_mask] = 0.5 * (
        1.0 + np.cos(np.pi * coordinates[surface_mask] / surface_width)
    )

    deep_spacing = max(
        coordinates[-1] - coordinates[-2],
        float(np.median(np.diff(coordinates))),
    )
    fixed_depth = min(
        float(fixed_deep_depth), coordinates[-1] - 3.0 * deep_spacing
    )
    if fixed_depth <= w_f:
        fixed_depth = 0.5 * (w_f + coordinates[-1])
    fixed_index = int(np.argmin(np.abs(coordinates - fixed_depth)))
    fixed_index = min(max(fixed_index, 1), coordinates.size - 2)

    fixed_node = endpoint(fixed_index)
    fixed_smooth = compact_cosine(float(coordinates[fixed_index]), 5e3)
    fixed_smooth[[0, -1]] = 0.0

    near_bottom_centre = coordinates[-1] - max(10e3, 5.0 * deep_spacing)
    near_bottom_half_width = coordinates[-1] - near_bottom_centre
    near_bottom_smooth = compact_cosine(
        float(near_bottom_centre), float(near_bottom_half_width)
    )
    near_bottom_smooth[[0, -1]] = 0.0

    wf_width = min(2e3, 0.2 * (coordinates[-1] - w_f))
    wf_rs = compact_cosine(w_f, wf_width)
    wf_rs[coordinates > w_f] = 0.0

    labels = (
        "nucleation_10km",
        "surface_endpoint",
        "surface_smooth",
        "wf_rs_side",
        "bottom_endpoint",
        "deep_fixed_node",
        "deep_fixed_smooth",
        "near_bottom_smooth",
    )
    modes = np.column_stack(
        (
            compact_cosine(10e3, 2e3),
            endpoint(0),
            surface_smooth,
            wf_rs,
            endpoint(-1),
            fixed_node,
            fixed_smooth,
            near_bottom_smooth,
        )
    )
    weights = _node_weights(coordinates)
    integrals = np.sum(weights[:, None] * np.abs(modes), axis=0)
    if np.any(integrals <= 0.0):
        missing = [labels[index] for index in np.flatnonzero(integrals <= 0.0)]
        raise ValueError(f"source modes are unresolved: {missing}")
    modes *= equivalent_width / integrals[None, :]
    return labels, modes


def _project_receivers(
    y: np.ndarray, receivers: np.ndarray, response: np.ndarray
) -> np.ndarray:
    weights = _node_weights(y)
    weighted_receivers = weights[:, None] * receivers
    denominator = np.sum(weighted_receivers, axis=0)
    if np.any(denominator <= 0.0):
        raise ValueError("receiver modes must have positive weighted integrals")
    return (weighted_receivers.T @ response) / denominator[:, None]


def current_fault_normal_stress(
    params: ModelParameters,
    grid: Grid,
    stress_util: StressCalUtil,
    solution: np.ndarray,
) -> NormalTractionRecovery:
    ux, uy = _unpack_velocity(np.asarray(solution, dtype=float), params)
    _, sigma = stress_util.compute_stress_fields(
        uy,
        ux,
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
    return NormalTractionRecovery(left=np.asarray(left), right=np.asarray(right))


def diagnose_bp3_interface_transfer(
    params: ModelParameters,
    *,
    friction_coefficient: float | np.ndarray = 0.6,
    source_labels: tuple[str, ...] | None = None,
    source_modes: np.ndarray | None = None,
    receiver_labels: tuple[str, ...] | None = None,
    receiver_modes: np.ndarray | None = None,
    creep_velocity: float | None = None,
) -> BP3InterfaceTransferResult:
    """Run all static endpoint and independent normal-recovery probes."""

    if params.case_type != CaseType.CALIFORNIA:
        raise ValueError("This diagnostic requires case_type='california'.")
    grid = Grid(params)
    y = grid.y
    if source_modes is None:
        default_labels, basis = default_endpoint_source_modes(y, params.W_f)
        source_labels = default_labels if source_labels is None else source_labels
    else:
        basis = np.asarray(source_modes, dtype=float)
    if receiver_modes is None:
        default_labels, receivers = default_nucleation_receiver_modes(y)
        receiver_labels = (
            default_labels if receiver_labels is None else receiver_labels
        )
    else:
        receivers = np.asarray(receiver_modes, dtype=float)

    if basis.ndim == 1:
        basis = basis[:, None]
    if receivers.ndim == 1:
        receivers = receivers[:, None]
    if basis.shape[0] != params.Ny or receivers.shape[0] != params.Ny:
        raise ValueError("source and receiver modes must use the fault grid")
    if source_labels is None or len(source_labels) != basis.shape[1]:
        raise ValueError("source_labels must match the source-mode columns")
    if receiver_labels is None or len(receiver_labels) != receivers.shape[1]:
        raise ValueError("receiver_labels must match the receiver-mode columns")

    mu = np.asarray(friction_coefficient, dtype=float)
    if mu.ndim == 0:
        mu = np.full(params.Ny, float(mu))
    if mu.shape != (params.Ny,) or np.any(~np.isfinite(mu)):
        raise ValueError("friction_coefficient must be finite on the fault grid")

    builder = MatrixBuilder(params, grid)
    solve = factorized(builder.build_LH().tocsc())
    stress_util = StressCalUtil(prefer_numba=False)
    zero = np.zeros(params.Ny, dtype=float)
    base_solution = solve(builder.build_RH(0.0, zero).copy())
    base_tau, _ = _fault_tractions(params, grid, stress_util, base_solution)
    base_current = current_fault_normal_stress(
        params, grid, stress_util, base_solution
    ).effective_average
    base_direct = direct_fault_normal_stress(
        params, grid, base_solution
    ).effective_average

    tau = np.empty_like(basis)
    sigma_current = np.empty_like(basis)
    sigma_direct = np.empty_like(basis)
    for column in range(basis.shape[1]):
        solution = solve(builder.build_RH(0.0, basis[:, column]).copy())
        tau_column, _ = _fault_tractions(params, grid, stress_util, solution)
        current = current_fault_normal_stress(
            params, grid, stress_util, solution
        ).effective_average
        direct = direct_fault_normal_stress(
            params, grid, solution
        ).effective_average
        tau[:, column] = tau_column - base_tau
        sigma_current[:, column] = current - base_current
        sigma_direct[:, column] = direct - base_direct

    projected_tau = _project_receivers(y, receivers, tau)
    projected_current = _project_receivers(y, receivers, sigma_current)
    projected_direct = _project_receivers(y, receivers, sigma_direct)
    coulomb_current = _project_receivers(
        y, receivers, tau - mu[:, None] * sigma_current
    )
    coulomb_direct = _project_receivers(
        y, receivers, tau - mu[:, None] * sigma_direct
    )

    transfer = FaultTransferResponse(
        y=y.copy(),
        source_labels=tuple(source_labels),
        source_modes=basis.copy(),
        receiver_labels=tuple(receiver_labels),
        receiver_modes=receivers.copy(),
        tau=tau,
        sigma_effective_current=sigma_current,
        sigma_effective_direct=sigma_direct,
        projected_tau=projected_tau,
        projected_sigma_effective_current=projected_current,
        projected_sigma_effective_direct=projected_direct,
        projected_coulomb_current=coulomb_current,
        projected_coulomb_direct=coulomb_direct,
    )

    velocity = (
        float(params.loading.V_L)
        if creep_velocity is None
        else float(creep_velocity)
    )
    if velocity == 0.0 or not np.isfinite(velocity):
        raise ValueError("a finite, nonzero creep_velocity is required")
    creep_start = int(np.searchsorted(y, params.W_f, side="left"))
    if creep_start >= params.Ny - 1:
        raise ValueError("W_f must leave at least one resolved creeping node")
    production_rate = np.zeros(params.Ny, dtype=float)
    production_rate[creep_start:] = velocity
    rs_endpoint_rate = np.zeros(params.Ny, dtype=float)
    rs_endpoint_rate[creep_start + 1 :] = velocity

    def full_loading(rate: np.ndarray):
        solution = solve(builder.build_RH(0.0, rate).copy())
        tau_value, _ = _fault_tractions(params, grid, stress_util, solution)
        current_value = current_fault_normal_stress(
            params, grid, stress_util, solution
        ).effective_average
        direct_value = direct_fault_normal_stress(
            params, grid, solution
        ).effective_average
        return tau_value, current_value, direct_value

    tau_production, current_production, direct_production = full_loading(
        production_rate
    )
    tau_endpoint, current_endpoint, direct_endpoint = full_loading(
        rs_endpoint_rate
    )
    wf_loading = WfEndpointLoadingComparison(
        production_rate=production_rate,
        rs_endpoint_rate=rs_endpoint_rate,
        tau_production=tau_production,
        tau_rs_endpoint=tau_endpoint,
        sigma_effective_current_production=current_production,
        sigma_effective_current_rs_endpoint=current_endpoint,
        sigma_effective_direct_production=direct_production,
        sigma_effective_direct_rs_endpoint=direct_endpoint,
    )
    return BP3InterfaceTransferResult(
        transfer=transfer,
        wf_loading=wf_loading,
    )
