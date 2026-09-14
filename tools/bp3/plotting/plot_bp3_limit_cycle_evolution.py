"""Visualize BP3-QD limit-cycle evolution and compare it with DFRA.

Example
-------
python -m tools.bp3.plotting.plot_bp3_limit_cycle_evolution \
  --case "800x160 km" /path/to/800-case/output/dataall.npz tab:green \
  --case "600x160 km" /path/to/600-case/output/dataall.npz tab:blue \
  --reference-dir /path/to/dfra/files

Running this file without command-line arguments (for example with the IDE
Run button) uses the 800x160 km and 600x160 km histories configured in
``DEFAULT_RESULTS_ROOT`` below.

The animation advances one event-to-event path at a time.  Earlier complete
cycles remain as faint traces, the current cycle is revealed progressively,
and a representative late DFRA cycle remains fixed as the target attractor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from fastslippy.utilities.bp3_limit_cycle import (
    LimitCycle,
    StationHistory,
    concatenate_cycle_paths,
    event_peak_indices,
    limit_cycles,
    load_dfra_station,
    load_fastslippy_station,
    representative_reference_cycle,
    resolve_dataall,
)


PANEL_ATTRIBUTES = (
    ("shear_stress_mpa", "Shear stress [MPa]"),
    ("log10_theta", r"$\log_{10}\theta$ [s]"),
    ("normal_stress_mpa", "Effective normal stress [MPa]"),
)


# Repository-relative IDE defaults.  Supply explicit command-line paths for
# production data, or place the comparison histories below this directory.
DEFAULT_RESULTS_ROOT = Path("artifacts") / "bp3_limit_cycle_inputs"
DEFAULT_800_CASE = DEFAULT_RESULTS_ROOT / (
    "x-small-benchmark-bp3-QD-hpc-50-1e-9-fix-v-"
    "800-160-xy-less-60degree"
)
DEFAULT_600_CASE = DEFAULT_RESULTS_ROOT / (
    "x-small-benchmark-bp3-QD-hpc-50-1e-9-fix-v-"
    "600-160-xy-less-60degree-part2"
)
DEFAULT_REFERENCE_DIR = DEFAULT_800_CASE / "output"
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "bp3_limit_cycle_800_vs_600"
)


def ide_default_arguments() -> list[str]:
    """Arguments used when the script is launched directly from an IDE."""

    return [
        "--case",
        "800x160 km (correct branch)",
        str(DEFAULT_800_CASE),
        "tab:green",
        "--case",
        "600x160 km (alternate branch)",
        str(DEFAULT_600_CASE),
        "tab:blue",
        "--reference-dir",
        str(DEFAULT_REFERENCE_DIR),
        "--output-dir",
        str(DEFAULT_OUTPUT_DIR),
    ]


def padded_limits(values: list[np.ndarray], fraction: float = 0.05) -> tuple[float, float]:
    finite = np.concatenate([item[np.isfinite(item)] for item in values])
    low = float(np.min(finite))
    high = float(np.max(finite))
    span = high - low
    if span == 0.0:
        span = max(abs(low), 1.0)
    return low - fraction * span, high + fraction * span


def configure_phase_axes(
    axes: list[plt.Axes], histories: list[StationHistory], reference: StationHistory,
    *, state_length_m: float,
) -> None:
    all_histories = histories + [reference]
    x_limits = padded_limits([item.log10_velocity for item in all_histories], 0.02)
    for axis, (attribute, ylabel) in zip(axes[:3], PANEL_ATTRIBUTES):
        axis.set_xlim(*x_limits)
        axis.set_ylim(
            *padded_limits([getattr(item, attribute) for item in all_histories])
        )
        axis.set_xlabel(r"$\log_{10}|V|$ [m/s]")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    steady_x = np.linspace(*x_limits, 200)
    axes[1].plot(
        steady_x,
        np.log10(state_length_m) - steady_x,
        color="0.45",
        linestyle=":",
        linewidth=1.2,
        label=r"steady state $V\theta/L=1$",
    )
    axes[3].set_xlabel("Complete cycle number")
    axes[3].set_ylabel("Peak-to-peak recurrence [yr]")
    axes[3].grid(alpha=0.25)


def add_reference_paths(
    axes: list[plt.Axes], reference: StationHistory, cycle: LimitCycle
) -> None:
    for axis, (attribute, _) in zip(axes[:3], PANEL_ATTRIBUTES):
        axis.plot(
            reference.log10_velocity[cycle.slice],
            getattr(reference, attribute)[cycle.slice],
            color="black",
            linewidth=2.4,
            alpha=0.8,
            label=f"DFRA representative cycle {cycle.number}",
            zorder=2,
        )


def plot_static_summary(
    histories: list[StationHistory],
    cycle_sets: list[list[LimitCycle]],
    colors: list[str],
    reference: StationHistory,
    reference_cycles: list[LimitCycle],
    representative: LimitCycle,
    output_path: Path,
    *,
    depth_km: float,
    state_length_m: float,
) -> None:
    fig, raw_axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    axes = list(raw_axes.flat)
    configure_phase_axes(axes, histories, reference, state_length_m=state_length_m)
    add_reference_paths(axes, reference, representative)

    for history, cycles, color in zip(histories, cycle_sets, colors):
        denominator = max(len(cycles) - 1, 1)
        for index, cycle in enumerate(cycles):
            alpha = 0.18 + 0.72 * index / denominator
            label = history.label if index == len(cycles) - 1 else None
            for axis, (attribute, _) in zip(axes[:3], PANEL_ATTRIBUTES):
                axis.plot(
                    history.log10_velocity[cycle.slice],
                    getattr(history, attribute)[cycle.slice],
                    color=color,
                    alpha=alpha,
                    linewidth=1.1 if index < len(cycles) - 1 else 2.0,
                    label=label,
                )
        axes[3].plot(
            [cycle.number for cycle in cycles],
            [cycle.recurrence_years for cycle in cycles],
            "o-",
            color=color,
            linewidth=1.8,
            markersize=5,
            label=history.label,
        )

    axes[3].plot(
        [cycle.number for cycle in reference_cycles],
        [cycle.recurrence_years for cycle in reference_cycles],
        "o-",
        color="black",
        alpha=0.7,
        linewidth=1.8,
        markersize=4,
        label="DFRA",
    )
    recurrence_values = [
        np.asarray([cycle.recurrence_years for cycle in cycles])
        for cycles in cycle_sets + [reference_cycles]
        if cycles
    ]
    axes[3].set_ylim(*padded_limits(recurrence_values, 0.1))
    axes[0].legend(fontsize=8, loc="best")
    axes[1].legend(fontsize=8, loc="best")
    axes[3].legend(fontsize=8, loc="best")
    fig.suptitle(
        f"BP3-QD 60° normal-fault limit cycles at {depth_km:g} km\n"
        "Faint paths are early cycles; opaque paths are later cycles",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def create_animation(
    histories: list[StationHistory],
    cycle_sets: list[list[LimitCycle]],
    colors: list[str],
    reference: StationHistory,
    reference_cycles: list[LimitCycle],
    representative: LimitCycle,
    output_path: Path,
    *,
    depth_km: float,
    state_length_m: float,
    frames_per_cycle: int,
    fps: int,
) -> None:
    if frames_per_cycle < 2:
        raise ValueError("frames_per_cycle must be at least two.")
    fig, raw_axes = plt.subplots(2, 2, figsize=(12.8, 8.4))
    axes = list(raw_axes.flat)
    configure_phase_axes(axes, histories, reference, state_length_m=state_length_m)
    add_reference_paths(axes, reference, representative)

    past_lines: list[list[plt.Line2D]] = []
    current_lines: list[list[plt.Line2D]] = []
    markers: list[list[plt.Line2D]] = []
    recurrence_lines: list[plt.Line2D] = []
    recurrence_markers: list[plt.Line2D] = []
    for history, color in zip(histories, colors):
        history_past: list[plt.Line2D] = []
        history_current: list[plt.Line2D] = []
        history_markers: list[plt.Line2D] = []
        for axis in axes[:3]:
            past, = axis.plot([], [], color=color, alpha=0.2, linewidth=1.0)
            current, = axis.plot(
                [], [], color=color, linewidth=2.1, label=history.label, zorder=3
            )
            marker, = axis.plot(
                [], [], "o", color=color, markersize=5, zorder=4
            )
            history_past.append(past)
            history_current.append(current)
            history_markers.append(marker)
        recurrence, = axes[3].plot(
            [], [], "o-", color=color, linewidth=1.8,
            markersize=5, label=history.label,
        )
        recurrence_marker, = axes[3].plot(
            [], [], "o", color=color, markersize=9, markerfacecolor="none",
            markeredgewidth=1.8,
        )
        past_lines.append(history_past)
        current_lines.append(history_current)
        markers.append(history_markers)
        recurrence_lines.append(recurrence)
        recurrence_markers.append(recurrence_marker)

    axes[3].plot(
        [cycle.number for cycle in reference_cycles],
        [cycle.recurrence_years for cycle in reference_cycles],
        "o-", color="black", alpha=0.7, linewidth=1.7,
        markersize=4, label="DFRA",
    )
    recurrence_values = [
        np.asarray([cycle.recurrence_years for cycle in cycles])
        for cycles in cycle_sets + [reference_cycles]
        if cycles
    ]
    axes[3].set_ylim(*padded_limits(recurrence_values, 0.1))
    axes[0].legend(fontsize=8, loc="best")
    axes[1].legend(fontsize=8, loc="best")
    axes[3].legend(fontsize=8, loc="best")
    title = fig.suptitle("", fontsize=13)

    total_cycles = max(len(cycles) for cycles in cycle_sets)
    total_frames = total_cycles * frames_per_cycle

    def update(frame: int):
        cycle_index = min(frame // frames_per_cycle, total_cycles - 1)
        local_frame = frame % frames_per_cycle
        fraction = local_frame / (frames_per_cycle - 1)
        artists: list[plt.Artist] = [title]
        for case_index, (history, cycles) in enumerate(zip(histories, cycle_sets)):
            completed_before = cycles[: min(cycle_index, len(cycles))]
            for panel_index, (attribute, _) in enumerate(PANEL_ATTRIBUTES):
                x_past, y_past = concatenate_cycle_paths(
                    history, completed_before, attribute
                )
                past_lines[case_index][panel_index].set_data(x_past, y_past)
                current_lines[case_index][panel_index].set_data([], [])
                markers[case_index][panel_index].set_data([], [])
                if cycle_index < len(cycles):
                    cycle = cycles[cycle_index]
                    values = getattr(history, attribute)[cycle.slice]
                    velocity = history.log10_velocity[cycle.slice]
                    count = max(1, int(np.ceil(fraction * velocity.size)))
                    current_lines[case_index][panel_index].set_data(
                        velocity[:count], values[:count]
                    )
                    markers[case_index][panel_index].set_data(
                        [velocity[count - 1]], [values[count - 1]]
                    )
                artists.extend(
                    (
                        past_lines[case_index][panel_index],
                        current_lines[case_index][panel_index],
                        markers[case_index][panel_index],
                    )
                )

            visible_count = min(cycle_index + 1, len(cycles))
            visible = cycles[:visible_count]
            recurrence_lines[case_index].set_data(
                [cycle.number for cycle in visible],
                [cycle.recurrence_years for cycle in visible],
            )
            if cycle_index < len(cycles):
                cycle = cycles[cycle_index]
                recurrence_markers[case_index].set_data(
                    [cycle.number], [cycle.recurrence_years]
                )
            else:
                recurrence_markers[case_index].set_data([], [])
            artists.extend(
                (recurrence_lines[case_index], recurrence_markers[case_index])
            )

        title.set_text(
            f"BP3-QD limit-cycle evolution at {depth_km:g} km — "
            f"cycle {cycle_index + 1}, stored-path progress {fraction:3.0%}"
        )
        return artists

    animation = FuncAnimation(
        fig, update, frames=total_frames, interval=1000 / fps, blit=False
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    animation.save(output_path, writer=PillowWriter(fps=fps), dpi=105)
    plt.close(fig)


def write_summary(
    output_path: Path,
    histories: list[StationHistory],
    cycle_sets: list[list[LimitCycle]],
    source_paths: list[Path],
    reference: StationHistory,
    reference_cycles: list[LimitCycle],
    reference_path: Path,
    representative: LimitCycle,
    *,
    depth_km: float,
    event_threshold: float,
) -> None:
    cases = []
    for history, cycles, source in zip(histories, cycle_sets, source_paths):
        peaks = event_peak_indices(history, event_threshold=event_threshold)
        cases.append(
            {
                "label": history.label,
                "source": str(resolve_dataall(source)),
                "event_peak_times_years": history.time_years[peaks].tolist(),
                "recurrence_years": [cycle.recurrence_years for cycle in cycles],
                "complete_cycles": len(cycles),
            }
        )
    reference_peaks = event_peak_indices(
        reference, event_threshold=event_threshold
    )
    payload = {
        "depth_km": depth_km,
        "event_threshold_ms": event_threshold,
        "cases": cases,
        "reference": {
            "label": reference.label,
            "source": str(reference_path.resolve()),
            "event_peak_times_years": reference.time_years[reference_peaks].tolist(),
            "recurrence_years": [
                cycle.recurrence_years for cycle in reference_cycles
            ],
            "representative_cycle": representative.number,
        },
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", action="append", nargs=3, required=True,
        metavar=("LABEL", "PATH", "COLOR"),
        help="Repeat for each FastSlipPy history to compare.",
    )
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/bp3_limit_cycle_comparison"),
    )
    parser.add_argument("--depth-km", type=float, default=10.0)
    parser.add_argument("--inner-spacing-m", type=float, default=50.0)
    parser.add_argument("--event-threshold", type=float, default=1e-3)
    parser.add_argument("--state-length-m", type=float, default=0.008)
    parser.add_argument("--reference-last-cycles", type=int, default=4)
    parser.add_argument("--frames-per-cycle", type=int, default=28)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--no-animation", action="store_true")
    supplied = list(sys.argv[1:] if argv is None else argv)
    has_case = "--case" in supplied
    has_reference = "--reference-dir" in supplied
    requests_help = "--help" in supplied or "-h" in supplied
    if not requests_help and not has_case and not has_reference:
        supplied = ide_default_arguments() + supplied
        print(
            "No data-source arguments supplied; using the configured "
            "800x160 km, 600x160 km, and DFRA defaults."
        )
    return parser.parse_args(supplied)


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)

    if args.fps < 1:
        parser.error("--fps must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = [spec[0] for spec in args.case]
    paths = [Path(spec[1]) for spec in args.case]
    colors = [spec[2] for spec in args.case]
    histories = [
        load_fastslippy_station(
            path,
            label=label,
            depth_km=args.depth_km,
            inner_spacing_m=args.inner_spacing_m,
        )
        for label, path in zip(labels, paths)
    ]
    reference = load_dfra_station(
        args.reference_dir, depth_km=args.depth_km
    )
    cycle_sets = [
        limit_cycles(history, event_threshold=args.event_threshold)
        for history in histories
    ]
    if any(not cycles for cycles in cycle_sets):
        missing = [
            history.label
            for history, cycles in zip(histories, cycle_sets)
            if not cycles
        ]
        raise ValueError(f"No complete cycles detected for: {missing}")
    reference_cycles = limit_cycles(
        reference, event_threshold=args.event_threshold
    )
    reference_index = representative_reference_cycle(
        reference_cycles, count=args.reference_last_cycles
    )
    representative = reference_cycles[reference_index]

    plot_static_summary(
        histories,
        cycle_sets,
        colors,
        reference,
        reference_cycles,
        representative,
        args.output_dir / "limit_cycle_summary.png",
        depth_km=args.depth_km,
        state_length_m=args.state_length_m,
    )
    write_summary(
        args.output_dir / "summary.json",
        histories,
        cycle_sets,
        paths,
        reference,
        reference_cycles,
        args.reference_dir,
        representative,
        depth_km=args.depth_km,
        event_threshold=args.event_threshold,
    )
    if not args.no_animation:
        create_animation(
            histories,
            cycle_sets,
            colors,
            reference,
            reference_cycles,
            representative,
            args.output_dir / "limit_cycle_evolution.gif",
            depth_km=args.depth_km,
            state_length_m=args.state_length_m,
            frames_per_cycle=args.frames_per_cycle,
            fps=args.fps,
        )

    print(f"Wrote limit-cycle diagnostics to {args.output_dir.resolve()}")
    for history, cycles in zip(histories, cycle_sets):
        recurrence = ", ".join(f"{item.recurrence_years:.3f}" for item in cycles)
        print(f"{history.label}: {len(cycles)} complete cycles; recurrence = {recurrence} yr")


if __name__ == "__main__":
    main()
