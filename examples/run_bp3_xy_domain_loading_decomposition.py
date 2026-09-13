r"""Decompose BP3-QD loading across horizontal and vertical domain sizes.

This diagnostic applies several elastic operators to the *same* saved BP3
fault state.  It is intended to distinguish three effects that are mixed in a
long dynamic run:

1. horizontal side-boundary distance;
2. vertical traction-free boundary distance;
3. the extra deep-creep fault added when ``ysize`` is increased.

The default comparison is deliberately small enough to answer whether a
``320 x 100 km`` domain can arrest the precursor because of its aspect ratio::

    python run_bp3_xy_domain_loading_decomposition.py CASE600/output \
      --output-dir artifacts/bp3_xy_domain_loading

The input history defaults to the production ``600 x 160 km``, 50 m-core
mesh.  The target operators need not have the same ``Ny``: saved fields are
interpolated only in the common fault interval, and prescribed plate-rate
creep is extended below ``W_f``.  The loading is split into side-boundary,
shallow, nucleation, lower-seismogenic, and three deep-creep depth ranges.

The largest cases require a production HPC node.  Factorizations are released
between operator cases.
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
class XYDomainCase:
    """One target elastic operator."""

    name: str
    xsize_km: float
    ysize_km: float
    nx: int
    ny: int


# Nx/Ny choices retain the 20 km, 50 m core and approximately 1.1 km maximum
# outer cells.  The production x800 mesh used Nx=1367 (max dx about 1.56 km);
# it is retained here because the preceding outer-resolution experiment showed
# that this difference is negligible compared with the domain effect.
CASE_CATALOG = (
    XYDomainCase("x320_y100", 320.0, 100.0, 901, 545),
    XYDomainCase("x320_y160", 320.0, 160.0, 901, 651),
    XYDomainCase("x600_y100", 600.0, 100.0, 1367, 545),
    XYDomainCase("x600_y160", 600.0, 160.0, 1367, 651),
    XYDomainCase("x600_y300", 600.0, 300.0, 1367, 901),
    XYDomainCase("x800_y160", 800.0, 160.0, 1367, 651),
    XYDomainCase("x800_y300", 800.0, 300.0, 1367, 901),
)

DEFAULT_CASE_NAMES = (
    "x320_y100",
    "x320_y160",
    "x600_y100",
    "x600_y160",
    "x600_y300",
)

FAULT_LOADING_COMPONENTS = (
    "shallow_locked",
    "nucleation_band",
    "lower_seismogenic",
    "deep_creep_near",
    "deep_creep_middle",
    "deep_creep_far",
)


def _snap_boundary(coordinates: np.ndarray, value: float) -> float:
    """Snap a nominal boundary to a coincident node within roundoff.

    Piecewise stretched coordinates can represent, for example, 15 km as
    ``15000.000000000002`` on one otherwise identical mesh.  Strict floating
    comparisons must not move that physical node between diagnostic regions.
    """

    coordinates = np.asarray(coordinates, dtype=float)
    index = int(np.argmin(np.abs(coordinates - value)))
    tolerance = 64.0 * np.finfo(float).eps * max(abs(value), 1.0)
    if abs(coordinates[index] - value) <= tolerance:
        return float(coordinates[index])
    return float(value)


def _closed_interval_mask(
    coordinates: np.ndarray, lower: float, upper: float
) -> np.ndarray:
    lower_node = _snap_boundary(coordinates, lower)
    upper_node = _snap_boundary(coordinates, upper)
    return (coordinates >= lower_node) & (coordinates <= upper_node)


def resolve_dataall(path: Path) -> Path:
    """Resolve a case directory, output directory, or explicit history."""

    path = path.resolve()
    for candidate in (path, path / "dataall.npz", path / "output" / "dataall.npz"):
        if candidate.is_file() and candidate.name == "dataall.npz":
            return candidate
    raise FileNotFoundError(f"Cannot find dataall.npz below {path}")


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


def _weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    denominator = float(np.sum(weights))
    if denominator <= np.finfo(float).tiny:
        return float("nan")
    return float(np.sum(weights * values) / denominator)


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
            f"interval {event_index}."
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


def build_parameters(args: argparse.Namespace, case: XYDomainCase) -> ModelParameters:
    """Build one production-style BP3 target operator."""

    p = ModelParameters(
        case_type="california",
        alpha=args.alpha,
        motion_sign=-1,
        auto_motion_sign=True,
        xsize=case.xsize_km * 1e3,
        ysize=case.ysize_km * 1e3,
        Nx=case.nx,
        Ny=case.ny,
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


def split_fault_velocity_components(
    y: np.ndarray,
    velocity: np.ndarray,
    *,
    band_top: float,
    band_bottom: float,
    creep_start: float,
    deep_split_1: float,
    deep_split_2: float,
) -> dict[str, np.ndarray]:
    """Return an exhaustive split that identifies newly added deep creep."""

    coordinates = np.asarray(y, dtype=float)
    rates = np.asarray(velocity, dtype=float)
    if rates.ndim == 1:
        rates = rates[:, None]
    if rates.ndim != 2 or rates.shape[0] != coordinates.size:
        raise ValueError("velocity must have shape (len(y), number_of_snapshots).")
    if not (
        0.0 <= band_top < band_bottom < creep_start
        < deep_split_1 < deep_split_2
    ):
        raise ValueError(
            "Require band_top < band_bottom < W_f < deep_split_1 < deep_split_2."
        )
    band_top_node = _snap_boundary(coordinates, band_top)
    band_bottom_node = _snap_boundary(coordinates, band_bottom)
    creep_start_node = _snap_boundary(coordinates, creep_start)
    deep_split_1_node = _snap_boundary(coordinates, deep_split_1)
    deep_split_2_node = _snap_boundary(coordinates, deep_split_2)
    masks = {
        "shallow_locked": coordinates < band_top_node,
        "nucleation_band": (
            (coordinates >= band_top_node) & (coordinates <= band_bottom_node)
        ),
        "lower_seismogenic": (
            (coordinates > band_bottom_node) & (coordinates < creep_start_node)
        ),
        "deep_creep_near": (
            (coordinates >= creep_start_node)
            & (coordinates <= deep_split_1_node)
        ),
        "deep_creep_middle": (
            (coordinates > deep_split_1_node)
            & (coordinates <= deep_split_2_node)
        ),
        "deep_creep_far": coordinates > deep_split_2_node,
    }
    coverage = np.sum(np.column_stack(list(masks.values())), axis=1)
    if not np.all(coverage == 1):
        raise RuntimeError("Fault-loading masks are not an exact partition.")
    return {
        name: np.where(mask[:, None], rates, 0.0)
        for name, mask in masks.items()
    }


def interpolate_frozen_states(
    source_y: np.ndarray,
    velocity: np.ndarray,
    theta: np.ndarray,
    sigma: np.ndarray,
    target_y: np.ndarray,
    *,
    creep_start: float,
    creep_velocity: float,
    state_length: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map saved states to a target y-grid and extend prescribed deep creep."""

    arrays = tuple(np.asarray(item, dtype=float) for item in (velocity, theta, sigma))
    if any(item.ndim != 2 or item.shape[0] != source_y.size for item in arrays):
        raise ValueError("Frozen source arrays must have shape (len(source_y), nsnapshot).")
    if any(item.shape[1] != arrays[0].shape[1] for item in arrays):
        raise ValueError("Frozen source arrays must contain the same snapshots.")

    def interpolate(values: np.ndarray) -> np.ndarray:
        return np.column_stack([
            np.interp(target_y, source_y, values[:, column])
            for column in range(values.shape[1])
        ])

    target_velocity = interpolate(arrays[0])
    target_theta = interpolate(arrays[1])
    target_sigma = interpolate(arrays[2])
    deep = target_y >= creep_start
    target_velocity[deep, :] = creep_velocity
    target_theta[deep, :] = state_length / abs(creep_velocity)
    if np.any(target_theta <= 0.0) or np.any(target_sigma <= 0.0):
        raise ValueError("Interpolated theta and effective normal stress must be positive.")
    return target_velocity, target_theta, target_sigma


def _source_grid(args: argparse.Namespace) -> Grid:
    case = XYDomainCase(
        "source",
        args.source_xsize_km,
        args.source_ysize_km,
        args.source_nx,
        args.source_ny,
    )
    return Grid(build_parameters(args, case))


def _snapshot_indices(
    args: argparse.Namespace,
    time: np.ndarray,
    velocity: np.ndarray,
    y: np.ndarray,
) -> dict[str, int]:
    selected = select_snapshot_indices(
        time,
        velocity,
        y,
        event_index=args.event_index,
        event_threshold=args.event_threshold,
        precursor_depth=args.precursor_depth_km * 1e3,
        precursor_prominence=args.precursor_prominence,
        exclusion_years=args.exclusion_years,
        runaway_threshold=args.runaway_threshold,
    )
    events = event_windows(velocity, args.event_threshold)
    previous = events[args.event_index]
    branch_time = time[previous.peak] + args.branch_age_years * SECONDS_PER_YEAR
    branch_entry = int(np.argmin(np.abs(time - branch_time)))
    selected = {"branch_entry": branch_entry, **selected}
    return {name: selected[name] for name in args.snapshots}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataall", type=Path)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/bp3_xy_domain_loading"),
    )
    parser.add_argument(
        "--cases", nargs="+", choices=[case.name for case in CASE_CATALOG],
        default=list(DEFAULT_CASE_NAMES),
    )
    parser.add_argument(
        "--snapshots", nargs="+",
        choices=(
            "branch_entry", "precursor_peak", "maximum_decay",
            "following_minimum", "runaway_onset",
        ),
        default=("branch_entry", "precursor_peak", "following_minimum"),
    )
    parser.add_argument("--source-xsize-km", type=float, default=600.0)
    parser.add_argument("--source-ysize-km", type=float, default=160.0)
    parser.add_argument("--source-nx", type=int, default=1367)
    parser.add_argument("--source-ny", type=int, default=651)
    parser.add_argument("--alpha", type=float, default=60.0)
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
    parser.add_argument("--precursor-depth-km", type=float, default=10.0)
    parser.add_argument("--precursor-prominence", type=float, default=0.03)
    parser.add_argument("--exclusion-years", type=float, default=1.0)
    parser.add_argument("--branch-age-years", type=float, default=68.8)
    parser.add_argument("--band-top-km", type=float, default=10.0)
    parser.add_argument("--band-bottom-km", type=float, default=15.0)
    parser.add_argument("--deep-split-km", nargs=2, type=float, default=(100.0, 160.0))
    parser.add_argument(
        "--probe-depths-km", nargs="+", type=float,
        default=(10.0, 11.0, 12.0, 15.0),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_cases = tuple(case for case in CASE_CATALOG if case.name in args.cases)
    if not selected_cases:
        raise ValueError("At least one target operator case is required.")
    if max(args.deep_split_km) <= args.wf_km:
        raise ValueError("Deep-creep split depths must lie below W_f.")

    dataall_path = resolve_dataall(args.dataall)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_grid = _source_grid(args)
    source_y = source_grid.y
    with np.load(dataall_path) as data:
        required = {"tm", "Vm", "sigmam", "thetam"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"dataall is missing fields: {sorted(missing)}")
        valid = np.isfinite(data["tm"]) & (data["tm"] > 0.0)
        time = np.asarray(data["tm"][valid], dtype=float)
        velocity = np.asarray(data["Vm"][:, valid], dtype=float)
        theta = np.asarray(data["thetam"][:, valid], dtype=float)
        sigma = np.asarray(data["sigmam"][:, valid], dtype=float)
    if velocity.shape[0] != source_y.size:
        raise ValueError(
            f"dataall Ny={velocity.shape[0]} does not match source Ny={source_y.size}."
        )
    if time.size < 3 or np.any(np.diff(time) <= 0.0):
        raise ValueError("Stored times must be strictly increasing.")

    snapshot_indices = _snapshot_indices(args, time, velocity, source_y)
    snapshot_names = list(snapshot_indices)
    indices = np.asarray(list(snapshot_indices.values()), dtype=int)
    source_velocity = velocity[:, indices]
    source_theta = theta[:, indices]
    source_sigma = sigma[:, indices]

    summaries: dict[str, dict] = {}
    decomposition_rows: list[dict] = []
    probe_rows: list[dict] = []
    profile_arrays: dict[str, np.ndarray] = {
        "source_y": source_y,
        "snapshot_names": np.asarray(snapshot_names),
        "snapshot_indices": indices,
        "snapshot_times": time[indices],
    }

    for case in selected_cases:
        params = build_parameters(args, case)
        grid = Grid(params)
        target_velocity, target_theta, target_sigma = interpolate_frozen_states(
            source_y,
            source_velocity,
            source_theta,
            source_sigma,
            grid.y,
            creep_start=params.W_f,
            creep_velocity=params.loading.V_L,
            state_length=params.L,
        )
        components = split_fault_velocity_components(
            grid.y,
            target_velocity,
            band_top=args.band_top_km * 1e3,
            band_bottom=args.band_bottom_km * 1e3,
            creep_start=params.W_f,
            deep_split_1=args.deep_split_km[0] * 1e3,
            deep_split_2=args.deep_split_km[1] * 1e3,
        )
        nsnapshot = len(snapshot_names)
        component_columns = np.concatenate(
            [components[name] for name in FAULT_LOADING_COMPONENTS], axis=1
        )
        solve_rates = np.column_stack((
            target_velocity,
            np.zeros(params.Ny),
            component_columns,
        ))
        print(
            f"[{case.name}] x={case.xsize_km:g} km, y={case.ysize_km:g} km, "
            f"Nx={case.nx}, Ny={case.ny}, DOF={grid.N:,}",
            flush=True,
        )
        response = solve_fault_loading_responses(params, solve_rates)
        total_tau = response.tau_rate[:, :nsnapshot]
        total_sigma = response.sigma_effective_rate[:, :nsnapshot]
        side_tau = response.tau_rate[:, nsnapshot]
        side_sigma = response.sigma_effective_rate[:, nsnapshot]
        component_tau: dict[str, np.ndarray] = {}
        component_sigma: dict[str, np.ndarray] = {}
        offset = nsnapshot + 1
        for component_index, component_name in enumerate(FAULT_LOADING_COMPONENTS):
            start = offset + component_index * nsnapshot
            stop = start + nsnapshot
            component_tau[component_name] = (
                response.tau_rate[:, start:stop] - side_tau[:, None]
            )
            component_sigma[component_name] = (
                response.sigma_effective_rate[:, start:stop] - side_sigma[:, None]
            )

        reconstructed_tau = side_tau[:, None] + sum(
            component_tau.values(), start=np.zeros_like(total_tau)
        )
        reconstructed_sigma = side_sigma[:, None] + sum(
            component_sigma.values(), start=np.zeros_like(total_sigma)
        )
        closure = {
            "tau_relative_l2": _relative_l2(
                reconstructed_tau - total_tau, total_tau
            ),
            "sigma_relative_l2": _relative_l2(
                reconstructed_sigma - total_sigma, total_sigma
            ),
        }

        friction = FrictionalZones(params, grid.y)
        weights = _node_weights(grid.y)
        band = _closed_interval_mask(
            grid.y,
            args.band_top_km * 1e3,
            args.band_bottom_km * 1e3,
        )
        focus = int(np.argmin(np.abs(grid.y - args.precursor_depth_km * 1e3)))
        snapshot_payload: dict[str, dict] = {}
        predicted_profiles = np.empty_like(target_velocity)
        for column, name in enumerate(snapshot_names):
            budget = frozen_rate_state_acceleration(
                target_velocity[:, column],
                target_theta[:, column],
                target_sigma[:, column],
                total_tau[:, column],
                total_sigma[:, column],
                a=friction.a,
                b=friction.b,
                mu0=params.mu0,
                V0=params.V0,
                L=params.L,
                eta=params.eta,
            )
            predicted_profiles[:, column] = budget.predicted
            speed_weights = weights[band] * np.abs(target_velocity[band, column]) ** 2

            shear_parts = {
                "side_boundary": side_tau / budget.denominator_pa,
            }
            normal_parts = {
                "side_boundary": (
                    -budget.friction_coefficient * side_sigma
                    / budget.denominator_pa
                ),
            }
            for component_name in FAULT_LOADING_COMPONENTS:
                shear_parts[component_name] = (
                    component_tau[component_name][:, column]
                    / budget.denominator_pa
                )
                normal_parts[component_name] = (
                    -budget.friction_coefficient
                    * component_sigma[component_name][:, column]
                    / budget.denominator_pa
                )
            shear_parts["full_fault"] = sum(
                (shear_parts[name] for name in FAULT_LOADING_COMPONENTS),
                start=np.zeros(params.Ny),
            )
            normal_parts["full_fault"] = sum(
                (normal_parts[name] for name in FAULT_LOADING_COMPONENTS),
                start=np.zeros(params.Ny),
            )
            acceleration_parts = {
                name_: shear_parts[name_] + normal_parts[name_]
                for name_ in ("side_boundary", *FAULT_LOADING_COMPONENTS, "full_fault")
            }
            acceleration_parts["state_evolution"] = budget.state_evolution
            reconstructed_acceleration = (
                acceleration_parts["side_boundary"]
                + acceleration_parts["full_fault"]
                + acceleration_parts["state_evolution"]
            )
            acceleration_error = budget.predicted - reconstructed_acceleration

            loading_payload: dict[str, dict[str, float]] = {}
            for component_name in (
                "side_boundary", *FAULT_LOADING_COMPONENTS,
                "full_fault", "state_evolution",
            ):
                contribution = acceleration_parts[component_name]
                shear = (
                    np.zeros(params.Ny)
                    if component_name == "state_evolution"
                    else shear_parts[component_name]
                )
                normal = (
                    np.zeros(params.Ny)
                    if component_name == "state_evolution"
                    else normal_parts[component_name]
                )
                values = {
                    "focus_shear_per_year": float(shear[focus] * SECONDS_PER_YEAR),
                    "focus_normal_per_year": float(normal[focus] * SECONDS_PER_YEAR),
                    "focus_dlnV_per_year": float(
                        contribution[focus] * SECONDS_PER_YEAR
                    ),
                    "mode_weighted_shear_per_year": float(
                        _weighted_average(shear[band], speed_weights)
                        * SECONDS_PER_YEAR
                    ),
                    "mode_weighted_normal_per_year": float(
                        _weighted_average(normal[band], speed_weights)
                        * SECONDS_PER_YEAR
                    ),
                    "mode_weighted_dlnV_per_year": float(
                        _weighted_average(contribution[band], speed_weights)
                        * SECONDS_PER_YEAR
                    ),
                }
                loading_payload[component_name] = values
                decomposition_rows.append({
                    "case": case.name,
                    "xsize_km": case.xsize_km,
                    "ysize_km": case.ysize_km,
                    "aspect_half_width_over_depth": (
                        0.5 * case.xsize_km / case.ysize_km
                    ),
                    "snapshot": name,
                    "time_years": time[indices[column]] / SECONDS_PER_YEAR,
                    "component": component_name,
                    **values,
                })

            snapshot_payload[name] = {
                "time_years": float(time[indices[column]] / SECONDS_PER_YEAR),
                "focus_depth_km": float(grid.y[focus] / 1e3),
                "focus_dlnV_per_year": float(
                    budget.predicted[focus] * SECONDS_PER_YEAR
                ),
                "mode_weighted_dlnV_per_year": float(
                    _weighted_average(budget.predicted[band], speed_weights)
                    * SECONDS_PER_YEAR
                ),
                "velocity_weighted_arrest_fraction": float(
                    _weighted_average(
                        (budget.predicted[band] < 0.0).astype(float), speed_weights
                    )
                ),
                "all_band_nodes_arresting": bool(
                    np.all(budget.predicted[band] < 0.0)
                ),
                "minimum_band_acceleration_per_year": float(
                    np.min(budget.predicted[band]) * SECONDS_PER_YEAR
                ),
                "maximum_band_acceleration_per_year": float(
                    np.max(budget.predicted[band]) * SECONDS_PER_YEAR
                ),
                "acceleration_decomposition_max_abs_per_year": float(
                    np.max(np.abs(acceleration_error)) * SECONDS_PER_YEAR
                ),
                "loading_decomposition": loading_payload,
            }
            for requested_depth in args.probe_depths_km:
                iy = int(np.argmin(np.abs(grid.y - requested_depth * 1e3)))
                probe_rows.append({
                    "case": case.name,
                    "snapshot": name,
                    "requested_depth_km": requested_depth,
                    "actual_depth_km": grid.y[iy] / 1e3,
                    "velocity_ms": target_velocity[iy, column],
                    "theta_s": target_theta[iy, column],
                    "normal_stress_pa": target_sigma[iy, column],
                    "predicted_dlnV_per_year": (
                        budget.predicted[iy] * SECONDS_PER_YEAR
                    ),
                })

        geometry = {
            "full_xsize_km": case.xsize_km,
            "half_width_Lx_km": 0.5 * case.xsize_km,
            "depth_Lz_km": case.ysize_km,
            "aspect_half_width_over_depth": 0.5 * case.xsize_km / case.ysize_km,
            "nx": case.nx,
            "ny": case.ny,
            "dof_count": int(grid.N),
            "minimum_dx_m": float(np.min(grid.dx_edges)),
            "maximum_dx_m": float(np.max(grid.dx_edges)),
            "minimum_dy_m": float(np.min(grid.dy_edges)),
            "maximum_dy_m": float(np.max(grid.dy_edges)),
        }
        summaries[case.name] = {
            "case": asdict(case),
            "geometry": geometry,
            "traction_decomposition_closure": closure,
            "snapshots": snapshot_payload,
        }
        profile_arrays[f"{case.name}_y"] = grid.y
        profile_arrays[f"{case.name}_velocity"] = target_velocity
        profile_arrays[f"{case.name}_predicted_acceleration"] = predicted_profiles
        profile_arrays[f"{case.name}_total_tau_rate"] = total_tau
        profile_arrays[f"{case.name}_total_sigma_rate"] = total_sigma
        profile_arrays[f"{case.name}_side_tau_rate"] = side_tau
        profile_arrays[f"{case.name}_side_sigma_rate"] = side_sigma
        for component_name in FAULT_LOADING_COMPONENTS:
            profile_arrays[f"{case.name}_{component_name}_tau_rate"] = (
                component_tau[component_name]
            )
            profile_arrays[f"{case.name}_{component_name}_sigma_rate"] = (
                component_sigma[component_name]
            )
        (args.output_dir / f"{case.name}_summary.json").write_text(
            json.dumps(summaries[case.name], indent=2), encoding="utf-8"
        )
        del (
            params, grid, target_velocity, target_theta, target_sigma,
            components, component_columns, solve_rates, response,
            total_tau, total_sigma, side_tau, side_sigma,
            component_tau, component_sigma, reconstructed_tau,
            reconstructed_sigma, friction, predicted_profiles,
        )
        gc.collect()

    payload = {
        "source_dataall": str(dataall_path),
        "source_geometry": {
            "xsize_km": args.source_xsize_km,
            "ysize_km": args.source_ysize_km,
            "nx": args.source_nx,
            "ny": args.source_ny,
        },
        "configuration": {
            "event_index": args.event_index,
            "branch_age_years": args.branch_age_years,
            "snapshots": snapshot_names,
            "band_km": [args.band_top_km, args.band_bottom_km],
            "wf_km": args.wf_km,
            "deep_split_km": list(args.deep_split_km),
            "fault_loading_components": list(FAULT_LOADING_COMPONENTS),
        },
        "selected_snapshots": {
            name: {
                "index": int(index),
                "time_years": float(time[index] / SECONDS_PER_YEAR),
                "maximum_velocity_ms": float(np.max(np.abs(velocity[:, index]))),
                "maximum_velocity_depth_km": float(
                    source_y[np.argmax(np.abs(velocity[:, index]))] / 1e3
                ),
            }
            for name, index in snapshot_indices.items()
        },
        "cases": summaries,
        "interpretation": (
            "Compare x320_y100 with x320_y160 for the vertical/aspect effect, "
            "and x320_y160 with x600_y160 for the horizontal-distance effect. "
            "Within a deeper operator, deep_creep_far isolates loading from "
            "fault added below 160 km; changes in side_boundary and the common "
            "fault components are geometry/boundary-kernel effects."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "loading_decomposition.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(decomposition_rows[0]))
        writer.writeheader()
        writer.writerows(decomposition_rows)
    with (args.output_dir / "probe_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(probe_rows[0]))
        writer.writeheader()
        writer.writerows(probe_rows)
    np.savez_compressed(
        args.output_dir / "counterfactual_profiles.npz", **profile_arrays
    )

    lines = [
        "# BP3 frozen xsize x ysize loading decomposition",
        "",
        f"Source: `{dataall_path}`",
        "",
        "Every target operator uses the same saved shallow fault state.",
        "",
        "| snapshot | case | Lx/Lz | focus net | band net | arrest fraction |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for snapshot in snapshot_names:
        for case in selected_cases:
            item = summaries[case.name]["snapshots"][snapshot]
            lines.append(
                f"| {snapshot} | {case.name} | "
                f"{summaries[case.name]['geometry']['aspect_half_width_over_depth']:.4f} | "
                f"{item['focus_dlnV_per_year']:.6g} | "
                f"{item['mode_weighted_dlnV_per_year']:.6g} | "
                f"{item['velocity_weighted_arrest_fraction']:.4f} |"
            )
    lines.extend([
        "",
        "Negative dlnV/dt denotes instantaneous arrest. Component shear and "
        "normal-stress contributions are in `loading_decomposition.csv`.",
        "",
        "A static sign change is strong evidence for a domain-controlled branch, "
        "but it does not by itself prove a long-term dynamic limit cycle.",
    ])
    (args.output_dir / "diagnostic_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Wrote x-y domain loading diagnostic to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
