r"""Project an existing BP3-QD cycle onto nucleation stiffness modes.

No elastic factorization or earthquake-cycle integration is performed.  The
script combines an existing ``dataall.npz`` with a stiffness-mode ``.npz`` and
diagnoses whether an arrested local velocity peak transfers into broader soft
modes before the next event.
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

from fastslippy.utilities.bp3_small_peak import (
    project_fault_history_onto_modes,
    reduced_rate_state_jacobian,
)


# Match FastSlipPy output and the BP3 plotting convention exactly.
SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0


def resolve_dataall(path: Path) -> Path:
    path = path.resolve()
    candidates = (path, path / "dataall.npz", path / "output" / "dataall.npz")
    for candidate in candidates:
        if candidate.is_file() and candidate.name == "dataall.npz":
            return candidate
    raise FileNotFoundError(f"Cannot find dataall.npz below {path}")


def checkpoint_time_level_metadata(dataall_path: Path) -> dict:
    checkpoints = sorted(dataall_path.parent.glob("data_*.npz"))
    if not checkpoints:
        return {
            "checkpoint_examined": None,
            "time_integrator": None,
            "state_time_level": None,
        }
    checkpoint = checkpoints[0]
    with np.load(checkpoint) as data:
        return {
            "checkpoint_examined": str(checkpoint),
            "time_integrator": (
                str(np.asarray(data["time_integrator"]).item())
                if "time_integrator" in data else None
            ),
            "state_time_level": (
                str(np.asarray(data["state_time_level"]).item())
                if "state_time_level" in data else None
            ),
        }


def event_windows(velocity: np.ndarray, threshold: float) -> list[tuple[int, int, int]]:
    activity = np.max(np.abs(velocity), axis=0)
    active = activity >= threshold
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    ends = np.flatnonzero(active & ~np.r_[active[1:], False])
    windows = []
    for start, end in zip(starts, ends):
        peak = int(start + np.argmax(activity[start : end + 1]))
        windows.append((int(start), peak, int(end)))
    return windows


def closest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def friction_profiles(y: np.ndarray, *, a0: float, a_max: float,
                      b0: float, H: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    a = np.full(y.shape, a_max, dtype=float)
    a[y < H] = a0
    transition = (y >= H) & (y < H + h)
    a[transition] = a0 + (a_max - a0) * (y[transition] - H) / h
    return a, np.full(y.shape, b0, dtype=float)


def precursor_at_depth(
    time_years: np.ndarray,
    velocity: np.ndarray,
    depth_index: int,
    start: int,
    stop: int,
    prominence: float,
) -> dict | None:
    if stop - start < 3:
        return None
    log_velocity = np.log10(np.maximum(np.abs(velocity[depth_index]), 1e-30))
    local_peaks, properties = find_peaks(
        log_velocity[start:stop], prominence=prominence
    )
    if local_peaks.size == 0:
        return None
    best = int(np.argmax(properties["prominences"]))
    peak = int(start + local_peaks[best])
    following = log_velocity[peak:stop]
    minimum = int(peak + np.argmin(following))
    return {
        "peak_index": peak,
        "peak_time_years": float(time_years[peak]),
        "peak_velocity_ms": float(velocity[depth_index, peak]),
        "peak_log10_abs_velocity": float(log_velocity[peak]),
        "prominence_decades": float(properties["prominences"][best]),
        "following_minimum_index": minimum,
        "following_minimum_time_years": float(time_years[minimum]),
        "following_minimum_velocity_ms": float(velocity[depth_index, minimum]),
        "following_drop_decades": float(log_velocity[peak] - log_velocity[minimum]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataall", type=Path)
    parser.add_argument("modes", type=Path)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/bp3_nucleation_path"),
    )
    parser.add_argument(
        "--cycle", type=int, default=1,
        help="One-based inter-event interval (default: after event 1).",
    )
    parser.add_argument("--event-threshold", type=float, default=1e-3)
    parser.add_argument("--exclude-after-event-years", type=float, default=0.5)
    parser.add_argument("--exclude-before-event-years", type=float, default=1.0)
    parser.add_argument("--precursor-prominence", type=float, default=0.05)
    parser.add_argument(
        "--probe-depths-km", nargs="+", type=float,
        default=(10.0, 11.0, 12.0, 13.0),
    )
    parser.add_argument("--baseline-band-top-km", type=float, default=8.0)
    parser.add_argument("--baseline-band-bottom-km", type=float, default=14.5)
    parser.add_argument("--velocity-floor", type=float, default=1e-30)
    parser.add_argument("--mu0", type=float, default=0.6)
    parser.add_argument("--V0", type=float, default=1e-6)
    parser.add_argument("--L", type=float, default=0.008)
    parser.add_argument("--a0", type=float, default=0.01)
    parser.add_argument("--a-max", type=float, default=0.025)
    parser.add_argument("--b0", type=float, default=0.015)
    parser.add_argument("--H-km", type=float, default=15.0)
    parser.add_argument("--h-km", type=float, default=3.0)
    parser.add_argument("--rho", type=float, default=2670.0)
    parser.add_argument("--cs", type=float, default=3464.0)
    parser.add_argument("--skip-jacobian", action="store_true")
    args = parser.parse_args()

    dataall_path = resolve_dataall(args.dataall)
    mode_path = args.modes.resolve()
    if not mode_path.is_file():
        raise FileNotFoundError(mode_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(mode_path) as modal_data:
        y = np.asarray(modal_data["y"], dtype=float)
        modes = np.asarray(modal_data["spatial_modes"], dtype=float)
        ratios = np.asarray(modal_data["stiffness_ratios"], dtype=float)
        tau_responses = np.asarray(modal_data["tau_responses"], dtype=float)
        sigma_responses = np.asarray(
            modal_data["sigma_effective_responses"], dtype=float
        )
        critical_metric = np.asarray(modal_data["critical_stiffness"], dtype=float)

    with np.load(dataall_path) as history:
        required = {"tm", "Vm", "Um", "thetam", "sigmam"}
        missing = required.difference(history.files)
        if missing:
            raise ValueError(f"dataall is missing fields: {sorted(missing)}")
        raw_time = np.asarray(history["tm"], dtype=float)
        valid = np.isfinite(raw_time) & (raw_time > 0.0)
        time = raw_time[valid]
        velocity = np.asarray(history["Vm"][:, valid], dtype=float)
        slip = np.asarray(history["Um"][:, valid], dtype=float)
        theta = np.asarray(history["thetam"][:, valid], dtype=float)
        sigma = np.asarray(history["sigmam"][:, valid], dtype=float)
        stored_dt = (
            np.asarray(history["dtm"][valid], dtype=float)
            if "dtm" in history else np.full(time.shape, np.nan)
        )
    time_level_metadata = checkpoint_time_level_metadata(dataall_path)

    if time.size < 2 or np.any(np.diff(time) <= 0.0):
        raise ValueError("Stored times must be finite and strictly increasing.")
    if velocity.shape[0] < y.size:
        raise ValueError(
            f"dataall has only {velocity.shape[0]} fault nodes; modes require {y.size}."
        )
    # Longer-depth cases share the same 50 m inner grid.  Only the common
    # prefix is needed because every stiffness mode is confined above 15 km.
    velocity = velocity[: y.size]
    slip = slip[: y.size]
    theta = theta[: y.size]
    sigma = sigma[: y.size]
    time_years = time / SECONDS_PER_YEAR

    windows = event_windows(velocity, args.event_threshold)
    interval_index = args.cycle - 1
    if interval_index < 0 or interval_index + 1 >= len(windows):
        raise ValueError(
            f"cycle={args.cycle} requires at least {args.cycle + 1} events; "
            f"found {len(windows)}."
        )
    previous_event = windows[interval_index]
    next_event = windows[interval_index + 1]
    interval_start = previous_event[2] + 1
    interval_stop = next_event[0]
    analysis_start = int(np.searchsorted(
        time_years,
        time_years[previous_event[2]] + args.exclude_after_event_years,
        side="left",
    ))
    analysis_stop = int(np.searchsorted(
        time_years,
        time_years[next_event[0]] - args.exclude_before_event_years,
        side="right",
    ))
    analysis_start = max(interval_start, analysis_start)
    analysis_stop = min(interval_stop, analysis_stop)
    if analysis_stop - analysis_start < 3:
        raise ValueError("Event exclusions leave too few inter-event snapshots.")

    baseline_band = (
        (y >= args.baseline_band_top_km * 1e3)
        & (y <= args.baseline_band_bottom_km * 1e3)
    )
    median_log_velocity = np.median(
        np.log10(np.maximum(np.abs(velocity[baseline_band]), args.velocity_floor)),
        axis=0,
    )
    baseline_index = int(
        analysis_start
        + np.argmin(median_log_velocity[analysis_start:analysis_stop])
    )

    log_velocity = np.log10(np.maximum(np.abs(velocity), args.velocity_floor))
    velocity_projection = project_fault_history_onto_modes(
        y, modes, velocity, reference_index=baseline_index,
        metric_profile=critical_metric,
    )
    slip_projection = project_fault_history_onto_modes(
        y, modes, slip, reference_index=baseline_index,
        metric_profile=critical_metric,
    )
    log_projection = project_fault_history_onto_modes(
        y, modes, log_velocity, reference_index=baseline_index,
        metric_profile=critical_metric,
    )
    pre_event_index = closest_index(
        time_years, time_years[next_event[0]] - 1.0
    )

    probe_rows = []
    probe_histories = {}
    for requested_depth in args.probe_depths_km:
        depth_index = closest_index(y, requested_depth * 1e3)
        result = precursor_at_depth(
            time_years, velocity, depth_index,
            analysis_start, analysis_stop, args.precursor_prominence,
        )
        row = {
            "requested_depth_km": requested_depth,
            "actual_depth_km": float(y[depth_index] / 1e3),
            "found": result is not None,
        }
        if result is not None:
            row.update({key: value for key, value in result.items() if not key.endswith("index")})
            peak = result["peak_index"]
            minimum = result["following_minimum_index"]
            row.update({
                "mode1_log_amplitude_at_peak": float(log_projection.coefficients[0, peak]),
                "mode1_log_amplitude_at_following_minimum": float(
                    log_projection.coefficients[0, minimum]
                ),
                "mode1_log_change_during_local_decay": float(
                    log_projection.coefficients[0, minimum]
                    - log_projection.coefficients[0, peak]
                ),
                "log_profile_broadening_during_local_decay": bool(
                    log_projection.coefficients[0, minimum]
                    > log_projection.coefficients[0, peak]
                ),
                "mode1_velocity_at_peak_ms": float(
                    velocity_projection.coefficients[0, peak]
                ),
                "mode1_velocity_at_following_minimum_ms": float(
                    velocity_projection.coefficients[0, minimum]
                ),
                "mode1_velocity_one_year_before_event_ms": float(
                    velocity_projection.coefficients[0, pre_event_index]
                ),
                "mode1_velocity_decay_fraction": float(
                    1.0
                    - abs(velocity_projection.coefficients[0, minimum])
                    / max(
                        abs(velocity_projection.coefficients[0, peak]),
                        np.finfo(float).tiny,
                    )
                ),
                "mode1_reactivation_factor_after_minimum": float(
                    abs(velocity_projection.coefficients[0, pre_event_index])
                    / max(
                        abs(velocity_projection.coefficients[0, minimum]),
                        np.finfo(float).tiny,
                    )
                ),
            })
        probe_rows.append(row)
        probe_histories[f"V_{requested_depth:g}km"] = velocity[depth_index]

    growth_rate = np.full(time.size, np.nan)
    growth_frequency = np.full(time.size, np.nan)
    a = b = None
    if not args.skip_jacobian:
        a, b = friction_profiles(
            y, a0=args.a0, a_max=args.a_max, b0=args.b0,
            H=args.H_km * 1e3, h=args.h_km * 1e3,
        )
        eta = args.rho * args.cs / 2.0
        for index in range(interval_start, interval_stop):
            jacobian = reduced_rate_state_jacobian(
                y, modes, tau_responses, sigma_responses,
                velocity=velocity[:, index], theta=theta[:, index],
                sigma_effective=sigma[:, index], a=a, b=b,
                mu0=args.mu0, V0=args.V0, L=args.L, eta=eta,
                metric_profile=critical_metric,
            )
            eigenvalues = np.linalg.eigvals(jacobian)
            dominant = eigenvalues[int(np.argmax(eigenvalues.real))]
            growth_rate[index] = dominant.real * SECONDS_PER_YEAR
            growth_frequency[index] = abs(dominant.imag) * SECONDS_PER_YEAR / (2.0 * np.pi)

    def counterfactual_growth(
        index: int, *, velocity_index: int | None = None,
        theta_index: int | None = None, sigma_index: int | None = None,
    ) -> float | None:
        if args.skip_jacobian or a is None or b is None:
            return None
        jacobian = reduced_rate_state_jacobian(
            y, modes, tau_responses, sigma_responses,
            velocity=velocity[:, index if velocity_index is None else velocity_index],
            theta=theta[:, index if theta_index is None else theta_index],
            sigma_effective=sigma[:, index if sigma_index is None else sigma_index],
            a=a, b=b, mu0=args.mu0, V0=args.V0, L=args.L,
            eta=args.rho * args.cs / 2.0, metric_profile=critical_metric,
        )
        return float(np.max(np.linalg.eigvals(jacobian).real) * SECONDS_PER_YEAR)

    representative = next((row for row in probe_rows if row["found"]), None)
    preferred = next(
        (row for row in probe_rows if row["found"] and abs(row["actual_depth_km"] - 11.0) < 0.1),
        representative,
    )
    key_indices = {
        "baseline": baseline_index,
        "one_year_before_next_event": pre_event_index,
    }
    if preferred is not None:
        key_indices["representative_peak"] = closest_index(
            time_years, preferred["peak_time_years"]
        )
        key_indices["following_minimum"] = closest_index(
            time_years, preferred["following_minimum_time_years"]
        )

    snapshots = {}
    for name, index in key_indices.items():
        snapshots[name] = {
            "time_years": float(time_years[index]),
            "max_abs_velocity_ms": float(np.max(np.abs(velocity[:, index]))),
            "mode_log_amplitudes": log_projection.coefficients[:, index].tolist(),
            "mode_velocity_amplitudes_ms": velocity_projection.coefficients[:, index].tolist(),
            "mode_slip_amplitudes_m": slip_projection.coefficients[:, index].tolist(),
            "log_projection_captured_fraction": float(
                log_projection.captured_fraction[index]
            ),
            "instantaneous_max_growth_per_year": (
                float(growth_rate[index]) if np.isfinite(growth_rate[index]) else None
            ),
            "instantaneous_frequency_cycles_per_year": (
                float(growth_frequency[index])
                if np.isfinite(growth_frequency[index]) else None
            ),
            "growth_with_baseline_sigma_per_year": counterfactual_growth(
                index, sigma_index=baseline_index
            ),
            "growth_with_baseline_theta_per_year": counterfactual_growth(
                index, theta_index=baseline_index
            ),
            "growth_with_baseline_velocity_per_year": counterfactual_growth(
                index, velocity_index=baseline_index
            ),
        }

    summary = {
        "source_dataall": str(dataall_path),
        "source_modes": str(mode_path),
        "cycle": args.cycle,
        "events_found": len(windows),
        "previous_event_peak_year": float(time_years[previous_event[1]]),
        "next_event_peak_year": float(time_years[next_event[1]]),
        "recurrence_years": float(time_years[next_event[1]] - time_years[previous_event[1]]),
        "baseline_time_years": float(time_years[baseline_index]),
        "source_time_level_metadata": time_level_metadata,
        "maximum_stored_step_years": float(
            np.nanmax(stored_dt[interval_start:interval_stop])
            / SECONDS_PER_YEAR
        ),
        "mode_stiffness_ratios": ratios.tolist(),
        "probe_precursors": probe_rows,
        "snapshots": snapshots,
        "interpretation_limits": (
            "Modal coefficients are projections onto a fixed checkpoint-based subspace. "
            "Instantaneous Jacobian eigenvalues are local growth indicators along a "
            "time-dependent non-steady trajectory, not finite-time stability exponents."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    segment = slice(interval_start, interval_stop)
    np.savez_compressed(
        args.output_dir / "nucleation_path_history.npz",
        time_years=time_years[segment],
        y=y,
        stiffness_ratios=ratios,
        mode_velocity_amplitudes=velocity_projection.coefficients[:, segment],
        mode_slip_amplitudes=slip_projection.coefficients[:, segment],
        mode_log_velocity_amplitudes=log_projection.coefficients[:, segment],
        velocity_projection_captured=velocity_projection.captured_fraction[segment],
        slip_projection_captured=slip_projection.captured_fraction[segment],
        log_projection_captured=log_projection.captured_fraction[segment],
        instantaneous_max_growth_per_year=growth_rate[segment],
        instantaneous_frequency_cycles_per_year=growth_frequency[segment],
        baseline_time_years=time_years[baseline_index],
        **{name: values[segment] for name, values in probe_histories.items()},
    )

    with (args.output_dir / "precursor_peaks.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fieldnames = sorted({key for row in probe_rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(probe_rows)

    plot_segment = slice(analysis_start, analysis_stop)
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for requested_depth in args.probe_depths_km:
        axes[0, 0].plot(
            time_years[plot_segment],
            np.log10(np.maximum(
                np.abs(probe_histories[f"V_{requested_depth:g}km"][plot_segment]),
                args.velocity_floor,
            )),
            label=f"{requested_depth:g} km",
        )
    axes[0, 0].set_ylabel(r"$\log_{10}|V|$ (m/s)")
    axes[0, 0].legend(fontsize=8)

    mode_count = min(4, modes.shape[1])
    for mode_index in range(mode_count):
        axes[0, 1].plot(
            time_years[plot_segment],
            log_projection.coefficients[mode_index, plot_segment],
            label=f"mode {mode_index + 1} ({ratios[mode_index]:.3f})",
        )
        axes[1, 0].plot(
            time_years[plot_segment],
            velocity_projection.coefficients[mode_index, plot_segment] / 1e-9,
            label=f"mode {mode_index + 1}",
        )
    axes[0, 1].set_ylabel("Log-profile modal amplitude (decades)")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].set_ylabel(r"$\Delta V$ modal amplitude / $V_p$")
    axes[1, 0].set_yscale("symlog", linthresh=0.05)
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot(
        time_years[plot_segment], growth_rate[plot_segment], color="tab:red"
    )
    axes[1, 1].axhline(0.0, color="black", linestyle=":", linewidth=1)
    axes[1, 1].set_ylabel(r"Instantaneous max Re($\lambda$) (yr$^{-1}$)")

    marker_indices = [baseline_index]
    if preferred is not None:
        marker_indices.extend((
            key_indices["representative_peak"], key_indices["following_minimum"]
        ))
    for axis in axes.flat:
        for index in marker_indices:
            axis.axvline(time_years[index], color="0.35", linestyle="--", linewidth=0.8)
        axis.grid(alpha=0.25)
        axis.set_xlabel("Time (yr)")
    figure.suptitle(
        f"BP3 nucleation path: event {args.cycle} to {args.cycle + 1}; "
        f"baseline {time_years[baseline_index]:.2f} yr"
    )
    figure.tight_layout()
    figure.savefig(args.output_dir / "nucleation_path_modes.png", dpi=180)
    plt.close(figure)

    lines = [
        "# BP3 nucleation-path diagnostic",
        "",
        f"- Data: `{dataall_path}`",
        f"- Modes: `{mode_path}`",
        f"- Event peaks: {time_years[previous_event[1]]:.6f} and "
        f"{time_years[next_event[1]]:.6f} yr "
        f"(recurrence {summary['recurrence_years']:.6f} yr)",
        f"- Automatic post-event baseline: {time_years[baseline_index]:.6f} yr",
    ]
    if time_level_metadata["state_time_level"] != "end":
        lines.extend((
            "- Warning: the checkpoint has no synchronized end-state metadata. "
            "For a legacy Euler run, V/tau can be one accepted step older than "
            "U/theta/sigma; instantaneous Jacobian values are therefore approximate. "
            f"The maximum stored inter-event step is "
            f"{summary['maximum_stored_step_years']:.5f} yr.",
        ))
    lines.extend((
        "",
        "| depth (km) | precursor | time (yr) | log10|V| | drop after peak "
        "(decades) | mode-1 V decay | later mode-1 reactivation | log-profile broadening |",
        "|---:|:---:|---:|---:|---:|---:|---:|:---:|",
    ))
    for row in probe_rows:
        if not row["found"]:
            lines.append(
                f"| {row['actual_depth_km']:.3f} | no |  |  |  |  |  |"
            )
        else:
            lines.append(
                f"| {row['actual_depth_km']:.3f} | yes | "
                f"{row['peak_time_years']:.6f} | "
                f"{row['peak_log10_abs_velocity']:.4f} | "
                f"{row['following_drop_decades']:.4f} | "
                f"{row['mode1_velocity_decay_fraction']:.3f} | "
                f"{row['mode1_reactivation_factor_after_minimum']:.3f}x | "
                f"{str(row['log_profile_broadening_during_local_decay']).lower()} |"
            )
    if not args.skip_jacobian:
        lines.extend((
            "",
            "| snapshot | actual max Re(lambda) | baseline sigma | baseline theta | baseline V |",
            "|---|---:|---:|---:|---:|",
        ))
        for name, values in snapshots.items():
            lines.append(
                f"| {name.replace('_', ' ')} | "
                f"{values['instantaneous_max_growth_per_year']:.6f} | "
                f"{values['growth_with_baseline_sigma_per_year']:.6f} | "
                f"{values['growth_with_baseline_theta_per_year']:.6f} | "
                f"{values['growth_with_baseline_velocity_per_year']:.6f} |"
            )
    lines.extend((
        "",
        "`Mode-1 V decay` is the fractional decline of the physical velocity-mode "
        "coefficient from the local precursor to its following minimum. `Later "
        "mode-1 reactivation` compares its magnitude one year before the event with "
        "that minimum. Log-profile broadening only describes the shape of a velocity "
        "field spanning many decades; it is not an energy-transfer measure.",
        "",
        "The Jacobian curve is instantaneous. Positive values must not be read as a "
        "finite-time Lyapunov exponent for the complete non-steady cycle. Snapshot "
        "counterfactuals freeze one field at its baseline profile; they are "
        "non-additive sensitivity tests, not a causal decomposition.",
    ))
    (args.output_dir / "diagnostic_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Saved diagnostic to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
