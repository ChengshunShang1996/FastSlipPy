r"""Separate BP3 horizontal-domain and stretched-outer-mesh effects.

The diagnostic applies four elastic operators to exactly the same saved fault
states.  The default 2 x 2 design is

=================  ========================  =====
full x width       outer intervals / side    Nx
=================  ========================  =====
320 km             250                       901
320 km             483                       1367
600 km             250                       901
600 km             483                       1367
=================  ========================  =====

All four meshes retain a 20 km, 50 m uniform fault-normal core.  The two
off-diagonal cases separate the domain effect from the outer-resolution effect
that are mixed when only the two production configurations are compared.

For each mesh, the complete loading response (side loading plus the frozen
fault velocity) is used to predict ``d(log|V|)/dt``.  Additional right-hand
sides partition the response into side-boundary loading, shallow locked,
nucleation-band, lower-seismogenic, and deep-creep contributions.  Their sum
must reconstruct the complete response.  The nucleation-band contribution also
supplies an effective Coulomb stiffness independent of side loading.
Sparse factorizations are released between cases, so production runs are
intended to execute sequentially on an HPC node.

Recommended reciprocal runs::

    # Failing 320 km history (precursor is clearest near 12 km)
    python run_bp3_x_domain_outer_counterfactual.py CASE320/output \
      --precursor-depth-km 12 --output-dir artifacts/from_x320

    # Arresting 600 km history (precursor is clearest near 10 km)
    python run_bp3_x_domain_outer_counterfactual.py CASE600/output \
      --precursor-depth-km 10 --output-dir artifacts/from_x600
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from fastslippy.pre_processing.frictional_zones import FrictionalZones
from fastslippy.pre_processing.grid import Grid
from fastslippy.pre_processing.model_parameters import ModelParameters
from fastslippy.utilities.bp3_small_peak import solve_fault_loading_responses
from fastslippy.utilities.bp3_state_budget import (
    event_windows,
    frozen_rate_state_acceleration,
)


SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0


@dataclass(frozen=True)
class XDomainOuterCase:
    """One level of the horizontal-domain/outer-resolution design."""

    name: str
    xsize_km: float
    outer_intervals_per_side: int

    def nx(self, inner_points: int) -> int:
        return inner_points + 2 * self.outer_intervals_per_side


DEFAULT_CASES = (
    XDomainOuterCase("x320_o250", 320.0, 250),
    XDomainOuterCase("x320_o483", 320.0, 483),
    XDomainOuterCase("x600_o250", 600.0, 250),
    XDomainOuterCase("x600_o483", 600.0, 483),
)

FAULT_LOADING_COMPONENTS = (
    "shallow_locked",
    "nucleation_band",
    "lower_seismogenic",
    "deep_creep",
)


def resolve_dataall(path: Path) -> Path:
    path = path.resolve()
    for candidate in (path, path / "dataall.npz", path / "output" / "dataall.npz"):
        if candidate.is_file() and candidate.name == "dataall.npz":
            return candidate
    raise FileNotFoundError(f"Cannot find dataall.npz below {path}")


def build_parameters(
    args: argparse.Namespace,
    case: XDomainOuterCase,
) -> ModelParameters:
    """Build the production BP3 configuration for one x-mesh."""

    p = ModelParameters(
        case_type="california",
        alpha=args.alpha,
        motion_sign=-1,
        auto_motion_sign=True,
        xsize=case.xsize_km * 1e3,
        ysize=args.ysize_km * 1e3,
        Nx=case.nx(args.x_inner_points),
        Ny=args.ny,
        rho=2670.0,
        cs=3464.0,
        mu0=0.6,
        nu=0.25,
        V0=1e-6,
        a0=0.01,
        a_max=0.025,
        b0=0.015,
        L=0.008,
        Vi=1e-9,
        H=15e3,
        h=3e3,
        W_f=args.wf_km * 1e3,
        x_stretch_enabled=True,
        y_stretch_enabled=True,
        x_stretch_inner_size=args.x_inner_km * 1e3,
        y_stretch_inner_size=args.y_inner_km * 1e3,
        x_stretch_inner_points=args.x_inner_points,
        y_stretch_inner_points=args.y_inner_points,
        x_stretch_power=args.stretch_power,
        y_stretch_power=args.stretch_power,
        allow_nonuniform_solver=True,
        output_vtk_option=False,
        fallback_to_iterative_on_oom=False,
        extrapolate_surface_fault_rate=False,
    )
    p.loading.V_p = args.plate_rate
    p.loading.V_L = args.plate_rate
    p.bc.left.ux.set_fixed()
    p.bc.left.uy.set_velocity(-0.5 * p.loading.V_p)
    p.bc.right.ux.set_fixed()
    p.bc.right.uy.set_velocity(0.5 * p.loading.V_p)
    p.bc.top.set_traction_free()
    p.bc.bottom.set_traction_free()
    p.layers.set_homogeneous(top=p.ysize, bottom=2.0 * p.ysize, a=p.a0, b=p.b0)
    p.apply_bp3_motion_sign()
    return p


def select_snapshot_indices(
    time: np.ndarray,
    velocity: np.ndarray,
    y: np.ndarray,
    *,
    event_index: int,
    event_threshold: float,
    precursor_depth: float,
    precursor_prominence: float,
    exclusion_years: float,
    runaway_threshold: float,
) -> dict[str, int]:
    """Select peak, strongest decay, following minimum, and runaway onset."""

    events = event_windows(velocity, event_threshold)
    if event_index < 0 or event_index + 1 >= len(events):
        raise ValueError(
            f"event-index {event_index} requires event {event_index + 1}; "
            f"the history contains {len(events)} events."
        )
    first, second = events[event_index : event_index + 2]
    start = int(np.searchsorted(
        time, time[first.end] + exclusion_years * SECONDS_PER_YEAR, side="left"
    ))
    stop = min(
        second.start,
        int(np.searchsorted(
            time, time[second.start] - exclusion_years * SECONDS_PER_YEAR,
            side="right",
        )),
    )
    iy = int(np.argmin(np.abs(y - precursor_depth)))
    log_rate = np.log10(np.maximum(np.abs(velocity[iy]), 1e-30))
    peaks, properties = find_peaks(
        log_rate[start:stop], prominence=precursor_prominence
    )
    if peaks.size == 0:
        raise ValueError(
            f"No precursor peak found near {y[iy] / 1e3:g} km in event "
            f"interval {event_index}; change --precursor-depth-km, "
            "--event-index, or --precursor-prominence."
        )
    best = int(np.argmax(properties["prominences"]))
    peak = int(start + peaks[best])
    minimum = int(peak + np.argmin(np.abs(velocity[iy, peak:second.start])))
    local_log_acceleration = np.gradient(log_rate, time, edge_order=2)
    maximum_decay = int(
        peak + np.argmin(local_log_acceleration[peak : minimum + 1])
    )

    activity = np.max(np.abs(velocity), axis=0)
    candidates = np.flatnonzero(
        activity[peak : second.start + 1] >= runaway_threshold
    )
    if candidates.size == 0:
        raise ValueError(
            f"No crossing of {runaway_threshold:g} m/s follows the precursor."
        )
    runaway = int(peak + candidates[0])
    return {
        "precursor_peak": peak,
        "maximum_decay": maximum_decay,
        "following_minimum": minimum,
        "runaway_onset": runaway,
    }


def _node_weights(y: np.ndarray) -> np.ndarray:
    spacing = np.diff(y)
    weights = np.empty_like(y)
    weights[0] = 0.5 * spacing[0]
    weights[-1] = 0.5 * spacing[-1]
    weights[1:-1] = 0.5 * (spacing[:-1] + spacing[1:])
    return weights


def _relative_l2(difference: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(difference)
        / max(np.linalg.norm(reference), np.finfo(float).tiny)
    )


def _cell_size_at(grid: Grid, distance: float) -> float | None:
    """Return the cell containing positive-x ``distance`` from the fault."""

    if distance < 0.0 or distance > grid.x[-1]:
        return None
    index = int(np.searchsorted(grid.x, distance, side="left") - 1)
    index = min(max(index, 0), grid.dx_edges.size - 1)
    return float(grid.dx_edges[index])


def _weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    denominator = float(np.sum(weights))
    if denominator <= np.finfo(float).tiny:
        return float("nan")
    return float(np.sum(weights * values) / denominator)


def split_fault_velocity_components(
    y: np.ndarray,
    velocity: np.ndarray,
    *,
    band_top: float,
    band_bottom: float,
    creep_start: float,
) -> dict[str, np.ndarray]:
    """Partition a fault-rate history into exhaustive depth regions.

    ``lower_seismogenic`` is retained explicitly because the requested
    shallow/nucleation/deep-creep split otherwise leaves the interval between
    the bottom of the nucleation band and ``W_f`` unassigned.  The masks are
    mutually exclusive and their sum reconstructs the input exactly.
    """

    coordinates = np.asarray(y, dtype=float)
    rates = np.asarray(velocity, dtype=float)
    if rates.ndim == 1:
        rates = rates[:, None]
    if coordinates.ndim != 1 or rates.ndim != 2:
        raise ValueError(
            "y must be one-dimensional and velocity one- or two-dimensional."
        )
    if rates.shape[0] != coordinates.size:
        raise ValueError("velocity's first dimension must match y.")
    if not 0.0 <= band_top < band_bottom < creep_start <= coordinates[-1]:
        raise ValueError(
            "Require 0 <= band_top < band_bottom < creep_start <= max(y)."
        )
    masks = {
        "shallow_locked": coordinates < band_top,
        "nucleation_band": (
            (coordinates >= band_top) & (coordinates <= band_bottom)
        ),
        "lower_seismogenic": (
            (coordinates > band_bottom) & (coordinates < creep_start)
        ),
        "deep_creep": coordinates >= creep_start,
    }
    coverage = np.sum(np.column_stack(list(masks.values())), axis=1)
    if not np.all(coverage == 1):
        raise RuntimeError("Fault-loading component masks are not an exact partition.")
    return {
        name: np.where(mask[:, None], rates, 0.0)
        for name, mask in masks.items()
    }


def _factorial_effects(
    cases: tuple[XDomainOuterCase, ...],
    summaries: dict[str, dict],
    snapshot_names: list[str],
) -> dict[str, dict]:
    """Return two-factor finite contrasts for every scalar branch metric."""

    widths = sorted({case.xsize_km for case in cases})
    outers = sorted({case.outer_intervals_per_side for case in cases})
    lookup = {
        (case.xsize_km, case.outer_intervals_per_side): case.name for case in cases
    }
    if len(widths) != 2 or len(outers) != 2 or len(lookup) != 4:
        return {}
    x0, x1 = widths
    o0, o1 = outers
    if any((x, outer) not in lookup for x in widths for outer in outers):
        return {}

    fields = (
        "focus_predicted_dlnV_per_year",
        "mode_weighted_dlnV_per_year",
        "velocity_weighted_arrest_fraction",
        "effective_stiffness_pa_per_m",
        "effective_over_critical_stiffness",
    )
    result: dict[str, dict] = {}
    for snapshot in snapshot_names:
        values = {
            field: {
                (x, outer): summaries[lookup[(x, outer)]]["snapshots"][snapshot][field]
                for x in widths for outer in outers
            }
            for field in fields
        }
        result[snapshot] = {}
        for field, item in values.items():
            a = item[(x0, o0)]
            b = item[(x0, o1)]
            c = item[(x1, o0)]
            d = item[(x1, o1)]
            result[snapshot][field] = {
                f"domain_effect_at_outer_{o0:g}": float(c - a),
                f"domain_effect_at_outer_{o1:g}": float(d - b),
                f"outer_effect_at_xsize_{x0:g}_km": float(b - a),
                f"outer_effect_at_xsize_{x1:g}_km": float(d - c),
                "interaction": float(d - c - b + a),
            }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataall", type=Path)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/bp3_x_domain_outer_counterfactual"),
    )
    parser.add_argument(
        "--cases", nargs="+", choices=[case.name for case in DEFAULT_CASES],
        default=[case.name for case in DEFAULT_CASES],
    )
    parser.add_argument("--alpha", type=float, default=60.0)
    parser.add_argument("--ysize-km", type=float, default=160.0)
    parser.add_argument("--ny", type=int, default=651)
    parser.add_argument("--wf-km", type=float, default=40.0)
    parser.add_argument("--plate-rate", type=float, default=1e-9)
    parser.add_argument("--x-inner-km", type=float, default=20.0)
    parser.add_argument("--x-inner-points", type=int, default=401)
    parser.add_argument("--y-inner-km", type=float, default=20.0)
    parser.add_argument("--y-inner-points", type=int, default=401)
    parser.add_argument("--stretch-power", type=int, default=2)
    parser.add_argument("--event-index", type=int, default=0)
    parser.add_argument("--event-threshold", type=float, default=1e-3)
    parser.add_argument("--runaway-threshold", type=float, default=1e-8)
    parser.add_argument("--precursor-depth-km", type=float, default=12.0)
    parser.add_argument("--precursor-prominence", type=float, default=0.03)
    parser.add_argument("--exclusion-years", type=float, default=1.0)
    parser.add_argument(
        "--probe-depths-km", nargs="+", type=float,
        default=(10.0, 11.0, 12.0, 13.0, 15.0),
    )
    parser.add_argument("--band-top-km", type=float, default=10.0)
    parser.add_argument("--band-bottom-km", type=float, default=15.0)
    parser.add_argument(
        "--mesh-probe-distances-km", nargs="+", type=float,
        default=(10.0, 20.0, 40.0, 80.0, 120.0),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = tuple(case for case in DEFAULT_CASES if case.name in args.cases)
    if not selected:
        raise ValueError("At least one operator case must be selected.")
    dataall_path = resolve_dataall(args.dataall)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_grid = Grid(build_parameters(args, selected[0]))
    y = source_grid.y
    with np.load(dataall_path) as data:
        required = {"tm", "Vm", "taum", "sigmam", "thetam"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"dataall is missing fields: {sorted(missing)}")
        valid = np.isfinite(data["tm"]) & (data["tm"] > 0.0)
        time = np.asarray(data["tm"][valid], dtype=float)
        velocity = np.asarray(data["Vm"][:, valid], dtype=float)
        traction = np.asarray(data["taum"][:, valid], dtype=float)
        sigma = np.asarray(data["sigmam"][:, valid], dtype=float)
        theta = np.asarray(data["thetam"][:, valid], dtype=float)
    if velocity.shape[0] != y.size:
        raise ValueError(
            f"dataall Ny={velocity.shape[0]} does not match configured Ny={y.size}."
        )
    if time.size < 3 or np.any(np.diff(time) <= 0.0):
        raise ValueError("stored times must be finite and strictly increasing.")

    snapshot_indices = select_snapshot_indices(
        time, velocity, y,
        event_index=args.event_index,
        event_threshold=args.event_threshold,
        precursor_depth=args.precursor_depth_km * 1e3,
        precursor_prominence=args.precursor_prominence,
        exclusion_years=args.exclusion_years,
        runaway_threshold=args.runaway_threshold,
    )
    snapshot_names = list(snapshot_indices)
    indices = np.asarray([snapshot_indices[name] for name in snapshot_names])
    frozen_velocity = velocity[:, indices]
    frozen_sigma = sigma[:, indices]
    frozen_theta = theta[:, indices]

    observed_tau_rate = np.gradient(traction, time, axis=1, edge_order=2)
    observed_sigma_rate = np.gradient(sigma, time, axis=1, edge_order=2)
    observed_acceleration = np.gradient(
        np.log(np.maximum(np.abs(velocity), np.finfo(float).tiny)),
        time, axis=1, edge_order=2,
    )
    band = (
        (y >= args.band_top_km * 1e3)
        & (y <= args.band_bottom_km * 1e3)
    )
    if not np.any(band):
        raise ValueError("The requested diagnostic band contains no y nodes.")
    node_weights = _node_weights(y)
    velocity_components = split_fault_velocity_components(
        y,
        frozen_velocity,
        band_top=args.band_top_km * 1e3,
        band_bottom=args.band_bottom_km * 1e3,
        creep_start=args.wf_km * 1e3,
    )
    frozen_modes = velocity_components["nucleation_band"].copy()
    frozen_modes[[0, -1], :] = 0.0
    if np.any(np.sum(node_weights[:, None] * frozen_modes**2, axis=0) <= 0.0):
        raise ValueError("A frozen nucleation-band velocity mode is identically zero.")

    probe_indices = {
        depth: int(np.argmin(np.abs(y - depth * 1e3)))
        for depth in args.probe_depths_km
    }
    summaries: dict[str, dict] = {}
    probe_rows: list[dict] = []
    geometry_rows: list[dict] = []
    saved_arrays: dict[str, np.ndarray] = {
        "y": y,
        "snapshot_names": np.asarray(snapshot_names),
        "snapshot_indices": indices,
        "snapshot_times": time[indices],
        "velocity": frozen_velocity,
        "theta": frozen_theta,
        "sigma": frozen_sigma,
        "observed_tau_rate": observed_tau_rate[:, indices],
        "observed_sigma_rate": observed_sigma_rate[:, indices],
        "observed_acceleration": observed_acceleration[:, indices],
        "frozen_band_modes": frozen_modes,
    }
    for component_name, component_velocity in velocity_components.items():
        saved_arrays[f"velocity_{component_name}"] = component_velocity
    decomposition_rows: list[dict] = []

    for case in selected:
        params = build_parameters(args, case)
        grid = Grid(params)
        print(
            f"[{case.name}] xsize={case.xsize_km:g} km, Nx={params.Nx}, "
            f"DOF={grid.N:,}, outer intervals/side="
            f"{case.outer_intervals_per_side}",
            flush=True,
        )
        # Columns: complete frozen states, one zero-fault state retaining side
        # loading, then four mutually exclusive fault-depth components.  The
        # latter reconstruct the complete fault rate exactly.  Subtracting the
        # zero column removes affine side loading from each fault contribution.
        component_rate_columns = np.concatenate(
            [velocity_components[name] for name in FAULT_LOADING_COMPONENTS],
            axis=1,
        )
        solve_rates = np.column_stack((
            frozen_velocity,
            np.zeros(params.Ny),
            component_rate_columns,
        ))
        response = solve_fault_loading_responses(params, solve_rates)
        count = len(snapshot_names)
        tau_rate = response.tau_rate[:, :count]
        sigma_rate = response.sigma_effective_rate[:, :count]
        base_tau_rate = response.tau_rate[:, count]
        base_sigma_rate = response.sigma_effective_rate[:, count]
        component_tau: dict[str, np.ndarray] = {}
        component_sigma: dict[str, np.ndarray] = {}
        component_start = count + 1
        for component_index, component_name in enumerate(
            FAULT_LOADING_COMPONENTS
        ):
            start = component_start + component_index * count
            stop = start + count
            component_tau[component_name] = (
                response.tau_rate[:, start:stop] - base_tau_rate[:, None]
            )
            component_sigma[component_name] = (
                response.sigma_effective_rate[:, start:stop]
                - base_sigma_rate[:, None]
            )
        mode_tau = component_tau["nucleation_band"]
        mode_sigma = component_sigma["nucleation_band"]

        reconstructed_tau = base_tau_rate[:, None] + sum(
            component_tau.values(), start=np.zeros_like(tau_rate)
        )
        reconstructed_sigma = base_sigma_rate[:, None] + sum(
            component_sigma.values(), start=np.zeros_like(sigma_rate)
        )
        traction_decomposition_closure = {
            "tau_relative_l2": _relative_l2(
                reconstructed_tau - tau_rate, tau_rate
            ),
            "sigma_relative_l2": _relative_l2(
                reconstructed_sigma - sigma_rate, sigma_rate
            ),
        }

        friction = FrictionalZones(params, y)
        snapshot_metrics: dict[str, dict] = {}
        predicted = np.empty_like(frozen_velocity)
        shear_terms = np.empty_like(frozen_velocity)
        normal_terms = np.empty_like(frozen_velocity)
        state_terms = np.empty_like(frozen_velocity)
        for column, (name, index) in enumerate(zip(snapshot_names, indices)):
            budget = frozen_rate_state_acceleration(
                frozen_velocity[:, column], frozen_theta[:, column],
                frozen_sigma[:, column], tau_rate[:, column],
                sigma_rate[:, column], a=friction.a, b=friction.b,
                mu0=params.mu0, V0=params.V0, L=params.L, eta=params.eta,
            )
            predicted[:, column] = budget.predicted
            shear_terms[:, column] = budget.shear_loading
            normal_terms[:, column] = budget.normal_stress
            state_terms[:, column] = budget.state_evolution
            focus = (
                int(np.argmax(np.abs(frozen_velocity[:, column])))
                if name == "runaway_onset"
                else int(np.argmin(np.abs(
                    y - args.precursor_depth_km * 1e3
                )))
            )
            speed_weights = (
                node_weights[band] * np.abs(frozen_velocity[band, column]) ** 2
            )
            modal_acceleration = _weighted_average(
                budget.predicted[band], speed_weights
            )
            modal_shear = _weighted_average(
                budget.shear_loading[band], speed_weights
            )
            modal_normal = _weighted_average(
                budget.normal_stress[band], speed_weights
            )
            modal_state = _weighted_average(
                budget.state_evolution[band], speed_weights
            )
            arrest_fraction = _weighted_average(
                (budget.predicted[band] < 0.0).astype(float), speed_weights
            )

            mode = frozen_modes[:, column]
            denominator = float(np.dot(node_weights, mode * mode))
            coulomb_response = (
                mode_tau[:, column]
                - budget.friction_coefficient * mode_sigma[:, column]
            )
            effective_stiffness = -float(
                np.dot(node_weights * mode, coulomb_response) / denominator
            )
            critical_profile = (
                frozen_sigma[:, column] * (friction.b - friction.a) / params.L
            )
            critical_stiffness = float(
                np.dot(node_weights * mode * mode, critical_profile) / denominator
            )

            shear_acceleration_components = {
                "side_boundary": base_tau_rate / budget.denominator_pa,
            }
            normal_acceleration_components = {
                "side_boundary": (
                    -budget.friction_coefficient * base_sigma_rate
                    / budget.denominator_pa
                ),
            }
            for component_name in FAULT_LOADING_COMPONENTS:
                shear_acceleration_components[component_name] = (
                    component_tau[component_name][:, column]
                    / budget.denominator_pa
                )
                normal_acceleration_components[component_name] = (
                    -budget.friction_coefficient
                    * component_sigma[component_name][:, column]
                    / budget.denominator_pa
                )
            shear_acceleration_components["full_fault"] = sum(
                (
                    shear_acceleration_components[name]
                    for name in FAULT_LOADING_COMPONENTS
                ),
                start=np.zeros(params.Ny),
            )
            normal_acceleration_components["full_fault"] = sum(
                (
                    normal_acceleration_components[name]
                    for name in FAULT_LOADING_COMPONENTS
                ),
                start=np.zeros(params.Ny),
            )
            acceleration_components = {
                component_name: (
                    shear_acceleration_components[component_name]
                    + normal_acceleration_components[component_name]
                )
                for component_name in (
                    "side_boundary", *FAULT_LOADING_COMPONENTS, "full_fault"
                )
            }
            reconstructed_acceleration = (
                acceleration_components["side_boundary"]
                + acceleration_components["full_fault"]
                + budget.state_evolution
            )
            acceleration_closure = budget.predicted - reconstructed_acceleration
            loading_decomposition: dict[str, dict[str, float]] = {}
            for component_name in (
                "side_boundary",
                *FAULT_LOADING_COMPONENTS,
                "full_fault",
                "state_evolution",
            ):
                contribution = (
                    budget.state_evolution
                    if component_name == "state_evolution"
                    else acceleration_components[component_name]
                )
                shear_contribution = (
                    np.zeros(params.Ny)
                    if component_name == "state_evolution"
                    else shear_acceleration_components[component_name]
                )
                normal_contribution = (
                    np.zeros(params.Ny)
                    if component_name == "state_evolution"
                    else normal_acceleration_components[component_name]
                )
                loading_decomposition[component_name] = {
                    "focus_shear_per_year": float(
                        shear_contribution[focus] * SECONDS_PER_YEAR
                    ),
                    "focus_normal_per_year": float(
                        normal_contribution[focus] * SECONDS_PER_YEAR
                    ),
                    "focus_dlnV_per_year": float(
                        contribution[focus] * SECONDS_PER_YEAR
                    ),
                    "mode_weighted_shear_per_year": float(
                        _weighted_average(
                            shear_contribution[band], speed_weights
                        ) * SECONDS_PER_YEAR
                    ),
                    "mode_weighted_normal_per_year": float(
                        _weighted_average(
                            normal_contribution[band], speed_weights
                        ) * SECONDS_PER_YEAR
                    ),
                    "mode_weighted_dlnV_per_year": float(
                        _weighted_average(
                            contribution[band], speed_weights
                        ) * SECONDS_PER_YEAR
                    ),
                }
                decomposition_rows.append({
                    "snapshot": name,
                    "time_years": time[index] / SECONDS_PER_YEAR,
                    "case": case.name,
                    "xsize_km": case.xsize_km,
                    "outer_intervals_per_side": (
                        case.outer_intervals_per_side
                    ),
                    "component": component_name,
                    **loading_decomposition[component_name],
                })

            snapshot_metrics[name] = {
                "time_years": float(time[index] / SECONDS_PER_YEAR),
                "focus_depth_km": float(y[focus] / 1e3),
                "focus_predicted_dlnV_per_year": float(
                    budget.predicted[focus] * SECONDS_PER_YEAR
                ),
                "focus_shear_per_year": float(
                    budget.shear_loading[focus] * SECONDS_PER_YEAR
                ),
                "focus_normal_per_year": float(
                    budget.normal_stress[focus] * SECONDS_PER_YEAR
                ),
                "focus_state_per_year": float(
                    budget.state_evolution[focus] * SECONDS_PER_YEAR
                ),
                "mode_weighted_dlnV_per_year": float(
                    modal_acceleration * SECONDS_PER_YEAR
                ),
                "mode_weighted_shear_per_year": float(
                    modal_shear * SECONDS_PER_YEAR
                ),
                "mode_weighted_normal_per_year": float(
                    modal_normal * SECONDS_PER_YEAR
                ),
                "mode_weighted_state_per_year": float(
                    modal_state * SECONDS_PER_YEAR
                ),
                "velocity_weighted_arrest_fraction": arrest_fraction,
                "all_band_nodes_arresting": bool(
                    np.all(budget.predicted[band] < 0.0)
                ),
                "maximum_band_acceleration_per_year": float(
                    np.max(budget.predicted[band]) * SECONDS_PER_YEAR
                ),
                "minimum_band_acceleration_per_year": float(
                    np.min(budget.predicted[band]) * SECONDS_PER_YEAR
                ),
                "effective_stiffness_pa_per_m": effective_stiffness,
                "critical_stiffness_pa_per_m": critical_stiffness,
                "effective_over_critical_stiffness": float(
                    effective_stiffness / critical_stiffness
                ),
                "loading_decomposition": loading_decomposition,
                "acceleration_decomposition_max_abs_per_year": float(
                    np.max(np.abs(acceleration_closure)) * SECONDS_PER_YEAR
                ),
                "tau_rate_vs_history_relative_l2": _relative_l2(
                    tau_rate[band, column] - observed_tau_rate[band, index],
                    observed_tau_rate[band, index],
                ),
                "sigma_rate_vs_history_relative_l2": _relative_l2(
                    sigma_rate[band, column] - observed_sigma_rate[band, index],
                    observed_sigma_rate[band, index],
                ),
            }
            for requested_depth, iy in probe_indices.items():
                probe_rows.append({
                    "snapshot": name,
                    "time_years": time[index] / SECONDS_PER_YEAR,
                    "case": case.name,
                    "xsize_km": case.xsize_km,
                    "outer_intervals_per_side": case.outer_intervals_per_side,
                    "requested_depth_km": requested_depth,
                    "actual_depth_km": y[iy] / 1e3,
                    "velocity_ms": frozen_velocity[iy, column],
                    "observed_dlnV_per_year": (
                        observed_acceleration[iy, index] * SECONDS_PER_YEAR
                    ),
                    "predicted_dlnV_per_year": (
                        budget.predicted[iy] * SECONDS_PER_YEAR
                    ),
                    "shear_loading_per_year": (
                        budget.shear_loading[iy] * SECONDS_PER_YEAR
                    ),
                    "normal_stress_per_year": (
                        budget.normal_stress[iy] * SECONDS_PER_YEAR
                    ),
                    "state_evolution_per_year": (
                        budget.state_evolution[iy] * SECONDS_PER_YEAR
                    ),
                    "tau_rate_pa_per_s": tau_rate[iy, column],
                    "sigma_rate_pa_per_s": sigma_rate[iy, column],
                })

        geometry = {
            "full_xsize_km": case.xsize_km,
            "half_width_Lx_km": case.xsize_km / 2.0,
            "nx": params.Nx,
            "outer_intervals_per_side": case.outer_intervals_per_side,
            "dof_count": int(grid.N),
            "minimum_dx_m": float(np.min(grid.dx_edges)),
            "maximum_dx_m": float(np.max(grid.dx_edges)),
            "maximum_adjacent_dx_ratio": float(max(
                np.max(grid.dx_edges[1:] / grid.dx_edges[:-1]),
                np.max(grid.dx_edges[:-1] / grid.dx_edges[1:]),
            )),
            "cell_size_at_distance_m": {
                f"{distance:g}_km": _cell_size_at(grid, distance * 1e3)
                for distance in args.mesh_probe_distances_km
            },
        }
        summaries[case.name] = {
            "case": asdict(case),
            "geometry": geometry,
            "traction_decomposition_closure": (
                traction_decomposition_closure
            ),
            "snapshots": snapshot_metrics,
        }
        geometry_rows.append({
            "case": case.name,
            **{key: value for key, value in geometry.items()
               if key != "cell_size_at_distance_m"},
            **{
                f"dx_at_{distance}": value
                for distance, value in geometry["cell_size_at_distance_m"].items()
            },
        })
        saved_arrays[f"{case.name}_tau_rate"] = tau_rate
        saved_arrays[f"{case.name}_sigma_rate"] = sigma_rate
        saved_arrays[f"{case.name}_mode_tau"] = mode_tau
        saved_arrays[f"{case.name}_mode_sigma"] = mode_sigma
        saved_arrays[f"{case.name}_side_tau_rate"] = base_tau_rate
        saved_arrays[f"{case.name}_side_sigma_rate"] = base_sigma_rate
        for component_name in FAULT_LOADING_COMPONENTS:
            saved_arrays[f"{case.name}_{component_name}_tau_rate"] = (
                component_tau[component_name]
            )
            saved_arrays[f"{case.name}_{component_name}_sigma_rate"] = (
                component_sigma[component_name]
            )
        saved_arrays[f"{case.name}_predicted_acceleration"] = predicted
        saved_arrays[f"{case.name}_shear_term"] = shear_terms
        saved_arrays[f"{case.name}_normal_term"] = normal_terms
        saved_arrays[f"{case.name}_state_term"] = state_terms
        (args.output_dir / f"{case.name}_summary.json").write_text(
            json.dumps(summaries[case.name], indent=2), encoding="utf-8"
        )
        del (
            params, grid, component_rate_columns, solve_rates, response,
            tau_rate, sigma_rate,
            base_tau_rate, base_sigma_rate, mode_tau, mode_sigma, predicted,
            component_tau, component_sigma, reconstructed_tau,
            reconstructed_sigma, shear_terms, normal_terms, state_terms,
        )
        gc.collect()

    factorial = _factorial_effects(selected, summaries, snapshot_names)
    payload = {
        "source_dataall": str(dataall_path),
        "configuration": {
            "event_index": args.event_index,
            "precursor_depth_km": args.precursor_depth_km,
            "band_km": [args.band_top_km, args.band_bottom_km],
            "ysize_km": args.ysize_km,
            "ny": args.ny,
            "x_inner_km": args.x_inner_km,
            "x_inner_points": args.x_inner_points,
            "core_dx_m": args.x_inner_km * 1e3 / (args.x_inner_points - 1),
            "fault_loading_components": list(FAULT_LOADING_COMPONENTS),
        },
        "selected_snapshots": {
            name: {
                "index": int(index),
                "time_years": float(time[index] / SECONDS_PER_YEAR),
                "maximum_velocity_ms": float(np.max(np.abs(velocity[:, index]))),
                "maximum_velocity_depth_km": float(
                    y[np.argmax(np.abs(velocity[:, index]))] / 1e3
                ),
            }
            for name, index in zip(snapshot_names, indices)
        },
        "cases": summaries,
        "factorial_effects": factorial,
        "decision_rule": (
            "A domain-driven branch change produces the same acceleration-sign "
            "change at both outer-resolution levels. An outer-mesh-driven change "
            "produces it at both domain widths. A change restricted to the "
            "production diagonal appears as a large interaction and is not "
            "identifiable from the two existing dynamic cases alone."
        ),
        "sign_convention": {
            "tau": "positive internal shear-traction magnitude",
            "sigma": "compression-positive effective normal stress",
            "arrest": "negative d(log|V|)/dt",
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "probe_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(probe_rows[0]))
        writer.writeheader()
        writer.writerows(probe_rows)
    with (args.output_dir / "mesh_geometry.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(geometry_rows[0]))
        writer.writeheader()
        writer.writerows(geometry_rows)
    with (args.output_dir / "loading_decomposition.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(decomposition_rows[0])
        )
        writer.writeheader()
        writer.writerows(decomposition_rows)
    np.savez_compressed(
        args.output_dir / "counterfactual_profiles.npz", **saved_arrays
    )

    lines = [
        "# BP3 frozen x-domain × outer-resolution counterfactual",
        "",
        f"Source: `{dataall_path}`",
        "",
        (
            "Every row uses the same saved V, theta, and effective normal "
            "stress. Rate columns are per year."
        ),
        "",
        (
            "| snapshot | case | xsize | outer/side | focus net | mode net | "
            "arrest fraction | Keff/Kcrit |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in snapshot_names:
        for case in selected:
            item = summaries[case.name]["snapshots"][name]
            lines.append(
                f"| {name} | {case.name} | {case.xsize_km:g} km | "
                f"{case.outer_intervals_per_side} | "
                f"{item['focus_predicted_dlnV_per_year']:.6g} | "
                f"{item['mode_weighted_dlnV_per_year']:.6g} | "
                f"{item['velocity_weighted_arrest_fraction']:.4f} | "
                f"{item['effective_over_critical_stiffness']:.6g} |"
            )
    lines.extend([
        "",
        "Finite two-factor contrasts are stored in `summary.json`; full depth "
        "profiles are in `counterfactual_profiles.npz`.",
        "",
        "## Loading decomposition closure",
        "",
        "| case | tau relative L2 | sigma relative L2 | max acceleration closure |",
        "|---|---:|---:|---:|",
    ])
    for case in selected:
        closure = summaries[case.name]["traction_decomposition_closure"]
        maximum_acceleration_closure = max(
            item["acceleration_decomposition_max_abs_per_year"]
            for item in summaries[case.name]["snapshots"].values()
        )
        lines.append(
            f"| {case.name} | {closure['tau_relative_l2']:.6e} | "
            f"{closure['sigma_relative_l2']:.6e} | "
            f"{maximum_acceleration_closure:.6e} /yr |"
        )
    lines.extend([
        "",
        "Component contributions are stored in `loading_decomposition.csv`. "
        "`lower_seismogenic` covers the otherwise unassigned interval between "
        "the nucleation band and W_f.",
    ])
    (args.output_dir / "diagnostic_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Wrote x-domain/outer-mesh diagnostic to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
