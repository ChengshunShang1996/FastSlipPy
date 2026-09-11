r"""Apply several x-core elastic operators to identical saved BP3 fault states.

The script selects three second-cycle snapshots from ``dataall.npz``:

1. a small precursor peak at a chosen depth;
2. the following local velocity minimum;
3. the first global crossing of a runaway-rate threshold.

For every x-core mesh, the saved ``V(y)`` is held fixed and only the static
elastic velocity problem is solved.  Differences in ``d(log|V|)/dt`` then
come solely from the spatial operator, not time integration or state history.
Production cases require substantial sparse-LU memory and are intended for
sequential HPC execution.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

try:
    from examples.run_bp3_x_core_convergence import (
        CORE_CASES,
        DEPTH_CASE,
        XCoreCase,
        build_parameters,
    )
except ModuleNotFoundError:  # Support copying the runners into one HPC case.
    from run_bp3_x_core_convergence import (  # type: ignore[no-redef]
        CORE_CASES,
        DEPTH_CASE,
        XCoreCase,
        build_parameters,
    )

from fastslippy.pre_processing.frictional_zones import FrictionalZones
from fastslippy.pre_processing.grid import Grid
from fastslippy.utilities.bp3_small_peak import solve_fault_loading_responses
from fastslippy.utilities.bp3_state_budget import (
    event_windows,
    frozen_rate_state_acceleration,
)


SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0


def resolve_dataall(path: Path) -> Path:
    path = path.resolve()
    for candidate in (path, path / "dataall.npz", path / "output" / "dataall.npz"):
        if candidate.is_file() and candidate.name == "dataall.npz":
            return candidate
    raise FileNotFoundError(f"Cannot find dataall.npz below {path}")


def select_snapshot_indices(
    time: np.ndarray,
    velocity: np.ndarray,
    y: np.ndarray,
    *,
    event_threshold: float,
    precursor_depth: float,
    precursor_prominence: float,
    exclusion_years: float,
    runaway_threshold: float,
) -> dict[str, int]:
    """Select precursor peak/minimum and runaway snapshots."""

    events = event_windows(velocity, event_threshold)
    if len(events) < 2:
        raise ValueError(f"At least two events are required; found {len(events)}.")
    first, second = events[:2]
    start = int(np.searchsorted(
        time, time[first.end] + exclusion_years * SECONDS_PER_YEAR,
        side="left",
    ))
    stop = int(np.searchsorted(
        time, time[second.start] - exclusion_years * SECONDS_PER_YEAR,
        side="right",
    ))
    stop = min(stop, second.start)
    iy = int(np.argmin(np.abs(y - precursor_depth)))
    log_rate = np.log10(np.maximum(np.abs(velocity[iy]), 1e-30))
    peaks, properties = find_peaks(
        log_rate[start:stop], prominence=precursor_prominence
    )
    if peaks.size == 0:
        raise ValueError(
            f"No precursor peak found near {y[iy] / 1e3:g} km; "
            "reduce --precursor-prominence or change --precursor-depth-km."
        )
    best = int(np.argmax(properties["prominences"]))
    peak = int(start + peaks[best])
    minimum = int(peak + np.argmin(np.abs(velocity[iy, peak:second.start])))

    activity = np.max(np.abs(velocity), axis=0)
    runaway_candidates = np.flatnonzero(
        activity[start : second.start + 1] >= runaway_threshold
    )
    if runaway_candidates.size == 0:
        raise ValueError(
            f"No second-cycle crossing of {runaway_threshold:g} m/s was found."
        )
    runaway = int(start + runaway_candidates[0])
    return {
        "precursor_peak": peak,
        "following_minimum": minimum,
        "runaway_onset": runaway,
    }


def _relative_l2(difference: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(difference)
        / max(np.linalg.norm(reference), np.finfo(float).tiny)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataall", type=Path)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/bp3_frozen_operator_counterfactual"),
    )
    parser.add_argument(
        "--cases", nargs="+",
        choices=[case.name for case in CORE_CASES],
        default=[case.name for case in CORE_CASES],
    )
    parser.add_argument("--xsize-km", type=float, default=320.0)
    parser.add_argument("--alpha", type=float, default=60.0)
    parser.add_argument("--wf-km", type=float, default=40.0)
    parser.add_argument("--plate-rate", type=float, default=1e-9)
    parser.add_argument("--x-inner-km", type=float, default=20.0)
    parser.add_argument("--y-inner-km", type=float, default=20.0)
    parser.add_argument("--y-inner-points", type=int, default=401)
    parser.add_argument("--stretch-power", type=int, default=2)
    parser.add_argument("--event-threshold", type=float, default=1e-3)
    parser.add_argument("--runaway-threshold", type=float, default=1e-8)
    parser.add_argument("--precursor-depth-km", type=float, default=12.0)
    parser.add_argument("--precursor-prominence", type=float, default=0.03)
    parser.add_argument("--exclusion-years", type=float, default=1.0)
    parser.add_argument(
        "--probe-depths-km", nargs="+", type=float,
        default=(10.0, 11.0, 12.0, 13.0, 15.0),
    )
    parser.add_argument("--band-top-km", type=float, default=8.0)
    parser.add_argument("--band-bottom-km", type=float, default=16.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataall_path = resolve_dataall(args.dataall)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_cases = [case for case in CORE_CASES if case.name in args.cases]

    # Every x-core case deliberately shares this y grid.
    source_core = next(
        (case for case in CORE_CASES if case.name == "dx50"),
        selected_cases[0],
    )
    args.nx = source_core.nx
    args.x_inner_points = source_core.x_inner_points
    source_params = build_parameters(args, DEPTH_CASE)
    source_grid = Grid(source_params)
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
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("stored times must be strictly increasing.")

    snapshot_indices = select_snapshot_indices(
        time, velocity, y,
        event_threshold=args.event_threshold,
        precursor_depth=args.precursor_depth_km * 1e3,
        precursor_prominence=args.precursor_prominence,
        exclusion_years=args.exclusion_years,
        runaway_threshold=args.runaway_threshold,
    )
    names = list(snapshot_indices)
    indices = np.array([snapshot_indices[name] for name in names], dtype=int)
    frozen_velocity = velocity[:, indices]
    frozen_theta = theta[:, indices]
    frozen_sigma = sigma[:, indices]

    friction = FrictionalZones(source_params, y)
    eta = source_params.eta
    observed_tau_rate = np.gradient(traction, time, axis=1, edge_order=2)
    observed_sigma_rate = np.gradient(sigma, time, axis=1, edge_order=2)
    observed_log_acceleration = np.gradient(
        np.log(np.maximum(np.abs(velocity), np.finfo(float).tiny)),
        time, axis=1, edge_order=2,
    )
    band = (y >= args.band_top_km * 1e3) & (y <= args.band_bottom_km * 1e3)
    probe_indices = {
        depth: int(np.argmin(np.abs(y - depth * 1e3)))
        for depth in args.probe_depths_km
    }

    rows: list[dict] = []
    case_summaries: dict[str, dict] = {}
    saved_arrays = {
        "y": y,
        "snapshot_names": np.asarray(names),
        "snapshot_indices": indices,
        "snapshot_times": time[indices],
        "velocity": frozen_velocity,
        "theta": frozen_theta,
        "sigma": frozen_sigma,
        "observed_tau_rate": observed_tau_rate[:, indices],
        "observed_sigma_rate": observed_sigma_rate[:, indices],
        "observed_log_acceleration": observed_log_acceleration[:, indices],
    }

    for core in selected_cases:
        print(
            f"[{core.name}] frozen operator solve: dx={core.core_dx_m:g} m, "
            f"Nx={core.nx}, {len(names)} snapshots",
            flush=True,
        )
        args.nx = core.nx
        args.x_inner_points = core.x_inner_points
        params = build_parameters(args, DEPTH_CASE)
        response = solve_fault_loading_responses(params, frozen_velocity)
        if not np.allclose(response.y, y, rtol=0.0, atol=1e-10):
            raise ValueError(f"{core.name} does not share the source y grid.")
        predicted = np.empty_like(frozen_velocity)
        shear_terms = np.empty_like(frozen_velocity)
        normal_terms = np.empty_like(frozen_velocity)
        state_terms = np.empty_like(frozen_velocity)
        snapshot_metrics = {}
        for column, (name, index) in enumerate(zip(names, indices)):
            budget = frozen_rate_state_acceleration(
                frozen_velocity[:, column], frozen_theta[:, column],
                frozen_sigma[:, column], response.tau_rate[:, column],
                response.sigma_effective_rate[:, column],
                a=friction.a, b=friction.b, mu0=params.mu0,
                V0=params.V0, L=params.L, eta=eta,
            )
            predicted[:, column] = budget.predicted
            shear_terms[:, column] = budget.shear_loading
            normal_terms[:, column] = budget.normal_stress
            state_terms[:, column] = budget.state_evolution
            focus = (
                int(np.argmax(np.abs(frozen_velocity[:, column])))
                if name == "runaway_onset"
                else int(np.argmin(np.abs(y - args.precursor_depth_km * 1e3)))
            )
            maximum = int(np.flatnonzero(band)[
                np.argmax(budget.predicted[band])
            ])
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
                "maximum_band_acceleration_per_year": float(
                    budget.predicted[maximum] * SECONDS_PER_YEAR
                ),
                "maximum_band_acceleration_depth_km": float(y[maximum] / 1e3),
                "tau_rate_vs_history_relative_l2": _relative_l2(
                    response.tau_rate[band, column]
                    - observed_tau_rate[band, index],
                    observed_tau_rate[band, index],
                ),
                "sigma_rate_vs_history_relative_l2": _relative_l2(
                    response.sigma_effective_rate[band, column]
                    - observed_sigma_rate[band, index],
                    observed_sigma_rate[band, index],
                ),
                "acceleration_vs_history_relative_l2": _relative_l2(
                    budget.predicted[band]
                    - observed_log_acceleration[band, index],
                    observed_log_acceleration[band, index],
                ),
            }
            for requested_depth, iy in probe_indices.items():
                rows.append({
                    "snapshot": name,
                    "time_years": time[index] / SECONDS_PER_YEAR,
                    "case": core.name,
                    "core_dx_m": core.core_dx_m,
                    "requested_depth_km": requested_depth,
                    "actual_depth_km": y[iy] / 1e3,
                    "velocity_ms": frozen_velocity[iy, column],
                    "observed_dlnV_per_year": (
                        observed_log_acceleration[iy, index] * SECONDS_PER_YEAR
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
                    "tau_rate_pa_per_s": response.tau_rate[iy, column],
                    "sigma_rate_pa_per_s": response.sigma_effective_rate[iy, column],
                })
        case_summaries[core.name] = {
            "case": asdict(core),
            "dof_count": int(2 * (params.Nx + 1) * (params.Ny + 1)),
            "snapshots": snapshot_metrics,
        }
        case_arrays = {
            "y": y,
            "snapshot_names": np.asarray(names),
            "snapshot_indices": indices,
            "snapshot_times": time[indices],
            "velocity": frozen_velocity,
            "theta": frozen_theta,
            "sigma": frozen_sigma,
            "tau_rate": response.tau_rate,
            "sigma_rate": response.sigma_effective_rate,
            "predicted_acceleration": predicted,
            "shear_term": shear_terms,
            "normal_term": normal_terms,
            "state_term": state_terms,
        }
        np.savez_compressed(
            args.output_dir / f"{core.name}_counterfactual.npz", **case_arrays
        )
        (args.output_dir / f"{core.name}_summary.json").write_text(
            json.dumps(case_summaries[core.name], indent=2), encoding="utf-8"
        )
        saved_arrays[f"{core.name}_tau_rate"] = response.tau_rate
        saved_arrays[f"{core.name}_sigma_rate"] = response.sigma_effective_rate
        saved_arrays[f"{core.name}_predicted_acceleration"] = predicted
        saved_arrays[f"{core.name}_shear_term"] = shear_terms
        saved_arrays[f"{core.name}_normal_term"] = normal_terms
        saved_arrays[f"{core.name}_state_term"] = state_terms
        del (
            params, response, predicted, shear_terms, normal_terms,
            state_terms, case_arrays,
        )
        gc.collect()

    with (args.output_dir / "probe_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.output_dir / "counterfactual_profiles.npz", **saved_arrays)

    baseline_name = "dx50" if "dx50" in case_summaries else selected_cases[0].name
    decisions = {}
    for name in names:
        baseline = case_summaries[baseline_name]["snapshots"][name]
        decisions[name] = {}
        for core in selected_cases:
            item = case_summaries[core.name]["snapshots"][name]
            decisions[name][core.name] = {
                "focus_acceleration_change_from_baseline_per_year": float(
                    item["focus_predicted_dlnV_per_year"]
                    - baseline["focus_predicted_dlnV_per_year"]
                ),
                "focus_acceleration_sign": (
                    "growing" if item["focus_predicted_dlnV_per_year"] > 0.0
                    else "arresting"
                ),
            }
    payload = {
        "source_dataall": str(dataall_path),
        "configuration": {
            "depth_case": asdict(DEPTH_CASE),
            "xsize_km": args.xsize_km,
            "alpha": args.alpha,
            "precursor_depth_km": args.precursor_depth_km,
            "runaway_threshold": args.runaway_threshold,
            "band_km": [args.band_top_km, args.band_bottom_km],
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
            for name, index in zip(names, indices)
        },
        "baseline_case": baseline_name,
        "cases": case_summaries,
        "counterfactual_decision": decisions,
        "decision_rule": (
            "If x-core refinement changes the focus acceleration from positive "
            "to negative at the precursor peak or minimum, spatial operator "
            "error can change the branch. If all converged cases retain the "
            "same sign and similar magnitude, prioritize the y/interface/BC "
            "operator or the inherited non-steady state path instead."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    lines = [
        "# BP3 frozen-state x-core operator counterfactual", "",
        "The same saved V, theta, and normal stress are used for every row.", "",
        "| snapshot | case | focus depth | shear | normal | state | net dlnV/dt | sign |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for name in names:
        for core in selected_cases:
            item = case_summaries[core.name]["snapshots"][name]
            sign = "growing" if item["focus_predicted_dlnV_per_year"] > 0 else "arresting"
            lines.append(
                f"| {name} | {core.name} | {item['focus_depth_km']:.3f} km | "
                f"{item['focus_shear_per_year']:.6g} | "
                f"{item['focus_normal_per_year']:.6g} | "
                f"{item['focus_state_per_year']:.6g} | "
                f"{item['focus_predicted_dlnV_per_year']:.6g} | {sign} |"
            )
    lines.extend([
        "", "All rate columns are per year. Full profiles are stored in "
        "`counterfactual_profiles.npz`; selected depths are in `probe_comparison.csv`.",
    ])
    (args.output_dir / "diagnostic_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Wrote frozen-operator diagnostic to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
