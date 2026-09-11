r"""Locate and decompose the BP3-QD post-event branching interval.

This history-only diagnostic compares FastSlipPy with DFRA at equal elapsed
times after the first event and decomposes ``d(log|V|)/dt`` into shear-loading,
normal-stress, and state-evolution terms.  It performs no elastic solve.

Example::

    python examples/run_bp3_branch_budget_diagnostic.py CASE/output \
      --reference-dir CASE/output \
      --output-dir artifacts/bp3_branch_budget_60
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from fastslippy.utilities.bp3_state_budget import (
    EventWindow,
    event_windows,
    rate_state_log_velocity_budget,
)


SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0


def resolve_dataall(path: Path) -> Path:
    path = path.resolve()
    for candidate in (path, path / "dataall.npz", path / "output" / "dataall.npz"):
        if candidate.is_file() and candidate.name == "dataall.npz":
            return candidate
    raise FileNotFoundError(f"Cannot find dataall.npz below {path}")


def _reference_path(directory: Path, depth_km: float, angle: int, motion: str) -> Path:
    code = int(round(depth_km * 10.0))
    path = directory / f"{angle}-dfra-onfault-dp{code:03d}-{motion}.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _closest(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def _largest_precursor(
    time: np.ndarray,
    velocity: np.ndarray,
    first: EventWindow,
    second: EventWindow,
    *,
    prominence: float,
    exclusion_years: float,
) -> tuple[int, int] | None:
    start_time = time[first.end] + exclusion_years * SECONDS_PER_YEAR
    stop_time = time[second.start] - exclusion_years * SECONDS_PER_YEAR
    start = max(first.end + 1, int(np.searchsorted(time, start_time)))
    stop = min(second.start, int(np.searchsorted(time, stop_time, side="right")))
    if stop - start < 3:
        return None
    log_rate = np.log10(np.maximum(np.abs(velocity), 1e-30))
    peaks, properties = find_peaks(log_rate[start:stop], prominence=prominence)
    if peaks.size == 0:
        return None
    best = int(np.argmax(properties["prominences"]))
    peak = int(start + peaks[best])
    minimum = int(peak + np.argmin(np.abs(velocity[peak:second.start])))
    return peak, minimum


def _budget_payload(budget, index: int) -> dict[str, float]:
    scale = SECONDS_PER_YEAR
    return {
        "observed_dlnV_per_year": float(budget.observed[index] * scale),
        "predicted_dlnV_per_year": float(budget.predicted[index] * scale),
        "shear_loading_per_year": float(budget.shear_loading[index] * scale),
        "normal_stress_per_year": float(budget.normal_stress[index] * scale),
        "state_evolution_per_year": float(budget.state_evolution[index] * scale),
        "closure_residual_per_year": float(budget.closure_residual[index] * scale),
    }


def _state_payload(time, velocity, traction, sigma, theta, slip, index) -> dict:
    return {
        "index": int(index),
        "time_years": float(time[index] / SECONDS_PER_YEAR),
        "velocity_ms": float(velocity[index]),
        "positive_shear_traction_mpa": float(traction[index] / 1e6),
        "normal_stress_mpa": float(sigma[index] / 1e6),
        "theta_s": float(theta[index]),
        "slip_m": float(slip[index]),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataall", type=Path)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/bp3_branch_budget"),
    )
    parser.add_argument("--angle", type=int, default=60)
    parser.add_argument("--motion", choices=("normal", "thrust"), default="normal")
    parser.add_argument("--depths-km", nargs="+", type=float, default=(5.0, 10.0, 15.0))
    parser.add_argument("--dy-m", type=float, default=50.0)
    parser.add_argument("--event-threshold", type=float, default=1e-3)
    parser.add_argument("--nucleation-thresholds", nargs="+", type=float,
                        default=(1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3))
    parser.add_argument("--nucleation-exclusion-years", type=float, default=1.0)
    parser.add_argument("--precursor-prominence", type=float, default=0.03)
    parser.add_argument("--precursor-exclusion-years", type=float, default=1.0)
    parser.add_argument("--ages-years", nargs="+", type=float,
                        default=(1.0, 10.0, 30.0, 50.0, 60.0, 68.0))
    parser.add_argument("--mu0", type=float, default=0.6)
    parser.add_argument("--V0", type=float, default=1e-6)
    parser.add_argument("--L", type=float, default=0.008)
    parser.add_argument("--a", type=float, default=0.01)
    parser.add_argument("--b", type=float, default=0.015)
    parser.add_argument("--rho", type=float, default=2670.0)
    parser.add_argument("--cs", type=float, default=3464.0)
    args = parser.parse_args()

    dataall_path = resolve_dataall(args.dataall)
    reference_dir = (
        args.reference_dir.resolve() if args.reference_dir else dataall_path.parent
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(dataall_path) as data:
        required = {"tm", "Vm", "Um", "taum", "sigmam", "thetam"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"dataall is missing fields: {sorted(missing)}")
        valid = np.isfinite(data["tm"]) & (data["tm"] > 0.0)
        time = np.asarray(data["tm"][valid], dtype=float)
        velocity = np.abs(np.asarray(data["Vm"][:, valid], dtype=float))
        slip = -np.asarray(data["Um"][:, valid], dtype=float)
        # Internal tau is positive in the rate-state equation; BP3 ASCII shear
        # stress is the negative of this quantity for the normal-fault case.
        traction = np.asarray(data["taum"][:, valid], dtype=float)
        sigma = np.asarray(data["sigmam"][:, valid], dtype=float)
        theta = np.asarray(data["thetam"][:, valid], dtype=float)

    if time.size < 3 or np.any(np.diff(time) <= 0.0):
        raise ValueError("stored times must be finite and strictly increasing.")
    fast_events = event_windows(velocity, args.event_threshold)
    if len(fast_events) < 2:
        raise ValueError(f"FastSlipPy history contains only {len(fast_events)} event(s).")
    eta = args.rho * args.cs / 2.0

    references: dict[float, dict[str, np.ndarray | list[EventWindow]]] = {}
    for depth in args.depths_km:
        raw = np.loadtxt(_reference_path(reference_dir, depth, args.angle, args.motion))
        if raw.ndim != 2 or raw.shape[1] < 6:
            raise ValueError(f"DFRA file at {depth:g} km must have at least six columns.")
        ref = {
            "time": raw[:, 0],
            "slip": raw[:, 1],
            "velocity": 10.0 ** raw[:, 2],
            "traction": -raw[:, 3] * 1e6,
            "sigma": raw[:, 4] * 1e6,
            "theta": 10.0 ** raw[:, 5],
        }
        ref["events"] = event_windows(ref["velocity"], args.event_threshold)
        if len(ref["events"]) < 2:
            raise ValueError(f"DFRA history at {depth:g} km has fewer than two events.")
        references[depth] = ref

    rows: list[dict] = []
    station_summaries: dict[str, dict] = {}
    for depth in args.depths_km:
        iy = int(round(depth * 1e3 / args.dy_m))
        if iy >= velocity.shape[0]:
            raise ValueError(f"depth {depth:g} km is outside the FastSlipPy history.")
        ref = references[depth]
        ref_events = ref["events"]
        fast_budget = rate_state_log_velocity_budget(
            time, velocity[iy], traction[iy], sigma[iy], theta[iy],
            a=args.a, b=args.b, mu0=args.mu0, V0=args.V0, L=args.L, eta=eta,
        )
        ref_budget = rate_state_log_velocity_budget(
            ref["time"], ref["velocity"], ref["traction"], ref["sigma"], ref["theta"],
            a=args.a, b=args.b, mu0=args.mu0, V0=args.V0, L=args.L, eta=eta,
        )
        for age in args.ages_years:
            fast_index = _closest(
                time, time[fast_events[0].end] + age * SECONDS_PER_YEAR
            )
            ref_index = _closest(
                ref["time"], ref["time"][ref_events[0].end] + age * SECONDS_PER_YEAR
            )
            row = {"depth_km": depth, "elapsed_years": age}
            for name, fast_value, ref_value, scale in (
                ("velocity_ms", velocity[iy, fast_index], ref["velocity"][ref_index], 1.0),
                ("slip_m", slip[iy, fast_index], ref["slip"][ref_index], 1.0),
                ("traction_mpa", traction[iy, fast_index], ref["traction"][ref_index], 1e-6),
                ("normal_stress_mpa", sigma[iy, fast_index], ref["sigma"][ref_index], 1e-6),
                ("theta_s", theta[iy, fast_index], ref["theta"][ref_index], 1.0),
            ):
                fast_scaled = float(fast_value * scale)
                ref_scaled = float(ref_value * scale)
                row[f"fast_{name}"] = fast_scaled
                row[f"dfra_{name}"] = ref_scaled
                row[f"difference_{name}"] = fast_scaled - ref_scaled
            rows.append(row)

        fast_precursor = _largest_precursor(
            time, velocity[iy], fast_events[0], fast_events[1],
            prominence=args.precursor_prominence,
            exclusion_years=args.precursor_exclusion_years,
        )
        ref_precursor = _largest_precursor(
            ref["time"], ref["velocity"], ref_events[0], ref_events[1],
            prominence=args.precursor_prominence,
            exclusion_years=args.precursor_exclusion_years,
        )

        def precursor_payload(indices, source, budget):
            if indices is None:
                return None
            peak, minimum = indices
            return {
                "peak": {
                    **_state_payload(
                        source["time"], source["velocity"], source["traction"],
                        source["sigma"], source["theta"], source["slip"], peak,
                    ),
                    **_budget_payload(budget, peak),
                },
                "following_minimum": {
                    **_state_payload(
                        source["time"], source["velocity"], source["traction"],
                        source["sigma"], source["theta"], source["slip"], minimum,
                    ),
                    **_budget_payload(budget, minimum),
                },
                "drop_decades": float(
                    np.log10(source["velocity"][peak] / source["velocity"][minimum])
                ),
            }

        fast_source = {
            "time": time, "velocity": velocity[iy], "traction": traction[iy],
            "sigma": sigma[iy], "theta": theta[iy], "slip": slip[iy],
        }
        station_summaries[f"{depth:g}_km"] = {
            "fast_precursor": precursor_payload(fast_precursor, fast_source, fast_budget),
            "dfra_precursor": precursor_payload(ref_precursor, ref, ref_budget),
        }

    _write_csv(args.output_dir / "same_age_station_comparison.csv", rows)

    # Track where the FastSlipPy precursor first crosses successively larger
    # rate thresholds.  This identifies the actual nucleation depth rather than
    # assuming that the requested 10 km station is the source.
    activity = np.max(velocity, axis=0)
    first_end = fast_events[0].end
    second_start = fast_events[1].start
    nucleation_start = int(np.searchsorted(
        time,
        time[first_end] + args.nucleation_exclusion_years * SECONDS_PER_YEAR,
        side="left",
    ))
    nucleation_path = []
    for threshold in args.nucleation_thresholds:
        candidates = np.flatnonzero(
            activity[nucleation_start : second_start + 1] >= threshold
        )
        if candidates.size == 0:
            continue
        index = int(nucleation_start + candidates[0])
        iy = int(np.argmax(velocity[:, index]))
        local_budget = rate_state_log_velocity_budget(
            time, velocity[iy], traction[iy], sigma[iy], theta[iy],
            a=args.a, b=args.b, mu0=args.mu0, V0=args.V0, L=args.L, eta=eta,
        )
        nucleation_path.append({
            "threshold_ms": threshold,
            "time_years": float(time[index] / SECONDS_PER_YEAR),
            "years_after_first_event": float(
                (time[index] - time[first_end]) / SECONDS_PER_YEAR
            ),
            "depth_km": float(iy * args.dy_m / 1e3),
            "maximum_velocity_ms": float(activity[index]),
            **_budget_payload(local_budget, index),
        })

    ref10 = references[min(args.depths_km, key=lambda value: abs(value - 10.0))]
    fast_recurrence = (
        time[fast_events[1].peak] - time[fast_events[0].peak]
    ) / SECONDS_PER_YEAR
    ref_recurrence = (
        ref10["time"][ref10["events"][1].peak]
        - ref10["time"][ref10["events"][0].peak]
    ) / SECONDS_PER_YEAR
    summary = {
        "source_dataall": str(dataall_path),
        "reference_directory": str(reference_dir),
        "fast_events_found": len(fast_events),
        "fast_first_two_peak_years": [
            float(time[event.peak] / SECONDS_PER_YEAR) for event in fast_events[:2]
        ],
        "dfra_first_two_peak_years_at_nearest_10km_station": [
            float(ref10["time"][event.peak] / SECONDS_PER_YEAR)
            for event in ref10["events"][:2]
        ],
        "fast_recurrence_years": float(fast_recurrence),
        "dfra_recurrence_years": float(ref_recurrence),
        "recurrence_shortfall_years": float(ref_recurrence - fast_recurrence),
        "fast_nucleation_path": nucleation_path,
        "station_precursors": station_summaries,
        "sign_convention": {
            "positive_shear_traction": "FastSlipPy taum = -BP3/DFRA shear column",
            "normal_stress": "compression positive",
        },
        "interpretation_limit": (
            "The differentiated balance identifies which saved-state term drives "
            "or arrests V. It localizes the branch but does not by itself identify "
            "which bulk stencil generated the shear-loading history."
        ),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    # Equal-age mismatch plot.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for depth in args.depths_km:
        selected = [row for row in rows if row["depth_km"] == depth]
        age = [row["elapsed_years"] for row in selected]
        axes[0, 0].plot(age, [row["difference_traction_mpa"] for row in selected], marker="o", label=f"{depth:g} km")
        axes[0, 1].plot(age, [row["difference_normal_stress_mpa"] for row in selected], marker="o")
        axes[1, 0].plot(age, [np.log10(row["fast_theta_s"] / row["dfra_theta_s"]) for row in selected], marker="o")
        axes[1, 1].plot(age, [np.log10(max(row["fast_velocity_ms"], 1e-30) / max(row["dfra_velocity_ms"], 1e-30)) for row in selected], marker="o")
    axes[0, 0].set_ylabel(r"$\Delta T$ [MPa]")
    axes[0, 1].set_ylabel(r"$\Delta \sigma_n$ [MPa]")
    axes[1, 0].set_ylabel(r"$\log_{10}(\theta_F/\theta_D)$")
    axes[1, 1].set_ylabel(r"$\log_{10}(V_F/V_D)$")
    for axis in axes[1]:
        axis.set_xlabel("Years after first event end")
    axes[0, 0].legend()
    for axis in axes.ravel():
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_dir / "same_age_mismatch.png", dpi=180)
    plt.close(fig)

    # The station comparison cannot reveal a nucleus between published DFRA
    # stations.  Plot the full FastSlipPy depth-time history around the branch.
    depth = np.arange(velocity.shape[0], dtype=float) * args.dy_m / 1e3
    depth_mask = (depth >= 8.0) & (depth <= 16.0)
    elapsed = (time - time[first_end]) / SECONDS_PER_YEAR
    time_mask = (
        (elapsed >= max(0.0, fast_recurrence - 20.0))
        & (time <= time[fast_events[1].peak])
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    mesh = ax.pcolormesh(
        elapsed[time_mask], depth[depth_mask],
        np.log10(np.maximum(velocity[np.ix_(depth_mask, time_mask)], 1e-30)),
        shading="auto", cmap="magma", vmin=-12.0, vmax=-3.0,
    )
    for item in nucleation_path:
        ax.plot(item["years_after_first_event"], item["depth_km"], "co", ms=4)
    ax.invert_yaxis()
    ax.set_xlabel("Years after first event end")
    ax.set_ylabel("Down-dip distance [km]")
    ax.set_title("FastSlipPy second-cycle nucleation migration")
    fig.colorbar(mesh, ax=ax, label=r"$\log_{10}|V|$ [m/s]")
    fig.tight_layout()
    fig.savefig(args.output_dir / "fast_nucleation_migration.png", dpi=180)
    plt.close(fig)

    lines = [
        "# BP3 first-event branching budget", "",
        f"- FastSlipPy recurrence: {fast_recurrence:.3f} yr",
        f"- DFRA recurrence (nearest 10 km station): {ref_recurrence:.3f} yr",
        f"- Shortfall: {ref_recurrence - fast_recurrence:.3f} yr", "",
        "## FastSlipPy nucleation path", "",
        "| V threshold (m/s) | time (yr) | depth (km) | shear | normal | state | predicted dlnV/dt |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in nucleation_path:
        lines.append(
            f"| {item['threshold_ms']:.1e} | {item['time_years']:.6f} | "
            f"{item['depth_km']:.3f} | {item['shear_loading_per_year']:.3g} | "
            f"{item['normal_stress_per_year']:.3g} | "
            f"{item['state_evolution_per_year']:.3g} | "
            f"{item['predicted_dlnV_per_year']:.3g} |"
        )
    lines.extend([
        "", "Rates in the last four columns are per year. Positive terms accelerate slip; negative terms arrest it.",
        "", "See `same_age_station_comparison.csv`, `same_age_mismatch.png`, "
        "`fast_nucleation_migration.png`, and `summary.json` for the full comparison.",
    ])
    (args.output_dir / "diagnostic_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Wrote BP3 branching diagnostic to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
