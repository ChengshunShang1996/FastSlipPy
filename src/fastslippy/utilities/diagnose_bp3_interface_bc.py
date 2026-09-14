"""Diagnose BP3 fault-normal stress and finite-boundary behaviour.

The utility compares one or more FastSlipPy case directories with the
on-fault reference files copied into each result directory.  Surface histories
are loaded from BP3 ASCII output, from ``dataall.npz`` (new outputs), or are
reconstructed at checkpoint times for older results.

Example
-------
python -m fastslippy.utilities.diagnose_bp3_interface_bc CASE300 CASE400
"""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


YEAR = 365.0 * 24.0 * 3600.0
SURFACE_NAMES = ("-16", "-00", "+00", "+16")
SURFACE_X = np.array([-16e3, -50.0, 50.0, 16e3])


@dataclass
class Geometry:
    alpha: float
    motion_sign: int
    plate_rate: float
    x: np.ndarray
    y: np.ndarray


@dataclass
class CaseData:
    label: str
    output_dir: Path
    geometry: Geometry
    time: np.ndarray
    velocity: np.ndarray
    normal_stress: np.ndarray
    surface_time: np.ndarray
    surface_disp1: np.ndarray
    surface_disp2: np.ndarray
    surface_source: str
    interface_time: np.ndarray
    interface_sigma_left: np.ndarray
    interface_sigma_right: np.ndarray


def _number(node: ast.AST):
    """Evaluate the small numeric expression subset used in case files."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (bool, int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _number(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left, right = _number(node.left), _number(node.right)
        operations = {
            ast.Add: lambda a, b: a + b,
            ast.Sub: lambda a, b: a - b,
            ast.Mult: lambda a, b: a * b,
            ast.Div: lambda a, b: a / b,
            ast.Pow: lambda a, b: a**b,
        }
        for kind, operation in operations.items():
            if isinstance(node.op, kind):
                return operation(left, right)
    raise ValueError(f"unsupported case-file expression: {ast.unparse(node)}")


def _case_parameters(case_file: Path) -> dict[str, float | int | bool]:
    tree = ast.parse(case_file.read_text(encoding="utf-8-sig"), filename=str(case_file))
    values: dict[str, float | int | bool] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name != "ModelParameters":
            continue
        for keyword in node.keywords:
            if keyword.arg is None:
                continue
            try:
                values[keyword.arg] = _number(keyword.value)
            except ValueError:
                pass
        break
    if not values:
        raise ValueError(f"no ModelParameters(...) call found in {case_file}")
    return values


def _piecewise_axis(length, points, inner_length, inner_points, power):
    total_intervals = points - 1
    inner_intervals = inner_points - 1
    s = np.linspace(0.0, 1.0, points)
    if inner_intervals >= total_intervals:
        return length * s
    boundary = inner_intervals / total_intervals
    linear_scale = total_intervals * inner_length / inner_intervals
    result = linear_scale * s
    outer = s > boundary
    chi = (s[outer] - boundary) / (1.0 - boundary)
    result[outer] = linear_scale * s[outer] + (
        length - linear_scale
    ) * chi**power
    return result


def _geometry(case_file: Path) -> Geometry:
    p = _case_parameters(case_file)
    xsize, ysize = float(p["xsize"]), float(p["ysize"])
    nx, ny = int(p["Nx"]), int(p["Ny"])
    if p.get("x_stretch_enabled", False):
        half = _piecewise_axis(
            xsize / 2.0,
            (nx - 1) // 2 + 1,
            float(p["x_stretch_inner_size"]) / 2.0,
            (int(p["x_stretch_inner_points"]) + 1) // 2,
            int(p["x_stretch_power"]),
        )
        x = np.concatenate((-half[:0:-1], half))
    else:
        x = np.linspace(-xsize / 2.0, xsize / 2.0, nx)
    if p.get("y_stretch_enabled", False):
        y = _piecewise_axis(
            ysize,
            ny,
            float(p["y_stretch_inner_size"]),
            int(p["y_stretch_inner_points"]),
            int(p["y_stretch_power"]),
        )
    else:
        y = np.linspace(0.0, ysize, ny)
    return Geometry(
        alpha=float(p.get("alpha", 90.0)),
        motion_sign=int(p.get("motion_sign", 1)),
        plate_rate=1e-9,
        x=x,
        y=y,
    )


def _resolve_case(path: Path) -> tuple[Path, Path]:
    path = path.resolve()
    candidates = (path / "dataall.npz", path / "output" / "dataall.npz")
    for data_file in candidates:
        if data_file.is_file():
            output_dir = data_file.parent
            case_file = output_dir.parent / "run_case_bp3.py"
            if not case_file.is_file():
                raise FileNotFoundError(f"cannot find run_case_bp3.py above {output_dir}")
            return output_dir, case_file
    raise FileNotFoundError(f"cannot find dataall.npz below {path}")


def _numeric_rows(path: Path, columns: int) -> np.ndarray:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                row = [float(value) for value in line.split()[:columns]]
            except ValueError:
                continue
            if len(row) >= columns:
                rows.append(row[:columns])
    if not rows:
        raise ValueError(f"no numeric data found in {path}")
    return np.asarray(rows)


def _surface_from_ascii(output_dir: Path):
    directory = output_dir / "output_BP3_QD"
    paths = [directory / f"srfst_fn{name}" for name in SURFACE_NAMES]
    if not all(path.is_file() for path in paths):
        return None
    tables = [_numeric_rows(path, 5) for path in paths]
    common_time = tables[0][:, 0]
    if not all(np.array_equal(table[:, 0], common_time) for table in tables[1:]):
        return None
    return common_time, np.stack([table[:, 1] for table in tables]), np.stack(
        [table[:, 2] for table in tables]
    ), "BP3 ASCII stations"


def _surface_from_dataall(data, valid):
    required = {"bp3_surface_x", "bp3_surface_disp1", "bp3_surface_disp2"}
    if not required.issubset(data.files):
        return None
    saved_x = data["bp3_surface_x"]
    indices = [int(np.argmin(np.abs(saved_x - target))) for target in SURFACE_X]
    return (
        data["tm"][valid],
        data["bp3_surface_disp1"][indices][:, valid],
        data["bp3_surface_disp2"][indices][:, valid],
        "dataall surface histories",
    )


def _surface_at_checkpoint(path: Path, geometry: Geometry):
    with np.load(path) as data:
        time = float(data["t"])
        ux, uy = data["ux"], data["uy"]
    xp = np.empty(geometry.x.size + 1)
    xp[1:-1] = 0.5 * (geometry.x[:-1] + geometry.x[1:])
    xp[0] = geometry.x[0] - 0.5 * (geometry.x[1] - geometry.x[0])
    xp[-1] = geometry.x[-1] + 0.5 * (geometry.x[-1] - geometry.x[-2])
    local_dx = geometry.x[geometry.x.size // 2 + 1] - geometry.x[geometry.x.size // 2]
    targets = np.array([-16e3, -local_dx / 2.0, local_dx / 2.0, 16e3])
    normal = np.mean(ux[:2, :], axis=0)
    un = np.interp(targets, geometry.x, normal)
    ut = np.interp(targets, xp, uy[0, :])
    angle = np.deg2rad(geometry.alpha)
    return time, un + ut * np.cos(angle), ut * np.sin(angle)


def _staggered(nodes: np.ndarray) -> np.ndarray:
    staggered = np.empty(nodes.size + 1)
    staggered[1:-1] = 0.5 * (nodes[:-1] + nodes[1:])
    staggered[0] = nodes[0] - 0.5 * (nodes[1] - nodes[0])
    staggered[-1] = nodes[-1] + 0.5 * (nodes[-1] - nodes[-2])
    return staggered


def _interface_from_checkpoints(
    output_dir: Path, geometry: Geometry, station_km: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample independently recovered normal stress on both fault faces.

    ``sigmaqs`` is cell-centred.  The solver obtains the fault value by first
    recovering the two columns adjacent to the fault and then averaging them.
    Retaining the two values separately exposes an interface-continuity error
    which is otherwise hidden in ``sigmam``.
    """
    paths = list(output_dir.glob("data_[0-9]*.npz"))
    if not paths:
        empty = np.empty(0)
        return empty, empty, empty
    nx, ny = geometry.x.size, geometry.y.size
    left_column = (nx - 1) // 2 - 1
    right_column = (nx - 1) // 2
    sigma_y = _staggered(geometry.y)[1:ny]
    station = station_km * 1e3
    rows = []
    for path in paths:
        with np.load(path) as data:
            if "sigmaqs" not in data.files:
                continue
            sigmaqs = data["sigmaqs"]
            rows.append(
                (
                    float(data["t"]),
                    float(np.interp(station, sigma_y, sigmaqs[:, left_column])),
                    float(np.interp(station, sigma_y, sigmaqs[:, right_column])),
                )
            )
    if not rows:
        empty = np.empty(0)
        return empty, empty, empty
    rows.sort(key=lambda row: row[0])
    values = np.asarray(rows)
    unique = np.concatenate(([True], np.diff(values[:, 0]) > 0.0))
    return values[unique, 0], values[unique, 1], values[unique, 2]


def _surface_from_checkpoints(output_dir: Path, geometry: Geometry):
    paths = list(output_dir.glob("data_[0-9]*.npz"))
    if not paths:
        return None
    rows = [_surface_at_checkpoint(path, geometry) for path in paths]
    rows.sort(key=lambda row: row[0])
    times = np.asarray([row[0] for row in rows])
    unique = np.concatenate(([True], np.diff(times) > 0.0))
    return (
        times[unique],
        np.stack([row[1] for row in rows], axis=1)[:, unique],
        np.stack([row[2] for row in rows], axis=1)[:, unique],
        "checkpoint reconstruction",
    )


def load_case(path: Path, label: str | None = None, station_km: float = 10.0) -> CaseData:
    output_dir, case_file = _resolve_case(path)
    geometry = _geometry(case_file)
    with np.load(output_dir / "dataall.npz") as data:
        valid = np.asarray(data["tm"] > 0.0)
        surface = _surface_from_dataall(data, valid)
        time = np.asarray(data["tm"][valid])
        velocity = np.asarray(data["Vm"][:, valid])
        normal_stress = np.asarray(data["sigmam"][:, valid])
    if surface is None:
        surface = _surface_from_ascii(output_dir)
    if surface is None:
        surface = _surface_from_checkpoints(output_dir, geometry)
    if surface is None:
        surface = (np.empty(0), np.empty((4, 0)), np.empty((4, 0)), "unavailable")
    interface = _interface_from_checkpoints(output_dir, geometry, station_km)
    inferred = re.sub(r"^x-small-benchmark-bp3-QD-hpc-", "", output_dir.parent.name)
    return CaseData(
        label=label or inferred,
        output_dir=output_dir,
        geometry=geometry,
        time=time,
        velocity=velocity,
        normal_stress=normal_stress,
        surface_time=surface[0],
        surface_disp1=surface[1],
        surface_disp2=surface[2],
        surface_source=surface[3],
        interface_time=interface[0],
        interface_sigma_left=interface[1],
        interface_sigma_right=interface[2],
    )


def _reference_fault(cases: list[CaseData], station_km: float):
    code = f"{int(round(station_km * 10)):03d}"
    for case in cases:
        angle = int(round(case.geometry.alpha))
        candidates = (
            case.output_dir / f"on-fault-dp{code}-{angle}deg.txt",
            case.output_dir / "output_BP3_QD" / f"fltst_dp{code}",
        )
        for path in candidates:
            if path.is_file():
                return _numeric_rows(path, 6), path
    return None, None


def _event_time(time: np.ndarray, velocity: np.ndarray) -> float:
    maximum = np.max(np.abs(velocity), axis=0)
    indices = np.flatnonzero(maximum >= 1e-3)
    return float(time[indices[0]]) if indices.size else float(time[-1])


def _reference_event_time(reference: np.ndarray) -> float:
    indices = np.flatnonzero(reference[:, 2] >= -3.0)
    return float(reference[indices[0], 0]) if indices.size else float(reference[-1, 0])


def _plot_normal(cases, station_km, reference, destination):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ref_time = ref_sigma = None
    if reference is not None:
        ref_time, ref_logv, ref_sigma = reference[:, 0], reference[:, 2], reference[:, 4]
        axes[0].plot(ref_time / YEAR, ref_logv, "k--", lw=1.2, label="reference")
        axes[1].plot(ref_time / YEAR, ref_sigma, "k--", lw=1.2, label="reference")
    for case in cases:
        iy = int(np.argmin(np.abs(case.geometry.y - station_km * 1e3)))
        axes[0].plot(case.time / YEAR, np.log10(np.maximum(np.abs(case.velocity[iy]), 1e-30)), label=case.label)
        sigma = case.normal_stress[iy] / 1e6
        axes[1].plot(case.time / YEAR, sigma, label=case.label)
        if ref_time is not None:
            # Do not turn a small event-timing offset into a spurious stress
            # error by comparing one solution during rupture with the other
            # solution immediately before rupture.
            stop = min(
                _event_time(case.time, case.velocity),
                _reference_event_time(reference),
            ) - YEAR
            overlap = case.time <= stop
            difference = sigma[overlap] - np.interp(case.time[overlap], ref_time, ref_sigma)
            axes[2].plot(case.time[overlap] / YEAR, difference, label=case.label)
    axes[0].set_ylabel(r"$\log_{10}|V|$ [m/s]")
    axes[1].set_ylabel(r"$\bar\sigma_n$ [MPa]")
    axes[2].set_ylabel("FastSlipPy - ref [MPa]")
    axes[2].set_xlabel("Time [yr]")
    axes[0].set_title(f"BP3 fault diagnostics at $x_d={station_km:g}$ km")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _load_reference_surface(directory: Path | None):
    if directory is None:
        return None
    paths = [directory / f"srfst_fn{name}" for name in SURFACE_NAMES]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError(f"reference surface files are incomplete in {directory}")
    tables = [_numeric_rows(path, 5) for path in paths]
    return tables


def _plot_surface(cases, reference, destination):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    groups = ((1, 2), (0, 3))
    components = (("Horizontal", "disp1"), ("Vertical", "disp2"))
    for row, (component_name, attribute) in enumerate(components):
        for col, indices in enumerate(groups):
            axis = axes[row, col]
            for case in cases:
                values = getattr(case, f"surface_{attribute}")
                for index in indices:
                    axis.plot(
                        case.surface_time / YEAR,
                        values[index],
                        label=f"{case.label}, x={SURFACE_NAMES[index]} km",
                    )
            if reference is not None:
                for index in indices:
                    table = reference[index]
                    column = 1 if attribute == "disp1" else 2
                    axis.plot(table[:, 0] / YEAR, table[:, column], "k--", lw=1.0,
                              label=f"reference, x={SURFACE_NAMES[index]} km")
            elif col == 1 and any(case.surface_time.size for case in cases):
                # Far from the fault trace the benchmark solution tends to
                # the signed rigid-body plate motion (BP3 equations 8a,b).
                geometry = cases[0].geometry
                end = max(case.surface_time[-1] for case in cases if case.surface_time.size)
                rigid_time = np.array([0.0, end])
                signed_rate = geometry.motion_sign * geometry.plate_rate
                component = np.cos(np.deg2rad(geometry.alpha)) if row == 0 else np.sin(
                    np.deg2rad(geometry.alpha)
                )
                for side, index in ((1.0, 0), (-1.0, 3)):
                    axis.plot(
                        rigid_time / YEAR,
                        side * signed_rate * rigid_time * component / 2.0,
                        "--",
                        color="0.35",
                        lw=1.0,
                        label=f"rigid, x={SURFACE_NAMES[index]} km",
                    )
            axis.set_title("fault trace $0^-/0^+$" if col == 0 else "far stations $-16/+16$ km")
            axis.set_ylabel(f"{component_name} displacement [m]")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7)
    axes[1, 0].set_xlabel("Time [yr]")
    axes[1, 1].set_xlabel("Time [yr]")
    fig.suptitle("BP3 surface-displacement boundary/interface diagnostic")
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _plot_interface(cases, station_km, destination):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for case in cases:
        if not case.interface_time.size:
            continue
        axes[0].plot(
            case.interface_time / YEAR,
            case.interface_sigma_left / 1e6,
            label=f"{case.label}, left",
        )
        axes[0].plot(
            case.interface_time / YEAR,
            case.interface_sigma_right / 1e6,
            "--",
            label=f"{case.label}, right",
        )
        axes[1].plot(
            case.interface_time / YEAR,
            (case.interface_sigma_left - case.interface_sigma_right) / 1e6,
            label=case.label,
        )
    axes[0].set_ylabel(r"Recovered $\Delta\sigma_n$ [MPa]")
    axes[1].set_ylabel(r"Left - right [MPa]")
    axes[1].set_xlabel("Time [yr]")
    axes[0].set_title(
        f"BP3 fault-face normal-stress continuity at $x_d={station_km:g}$ km"
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _write_report(cases, station_km, reference, reference_path, destination):
    lines = [f"Fault station: {station_km:g} km", f"Reference: {reference_path or 'not found'}", ""]
    for case in cases:
        iy = int(np.argmin(np.abs(case.geometry.y - station_km * 1e3)))
        event = _event_time(case.time, case.velocity)
        lines.append(f"[{case.label}]")
        lines.append(f"surface source: {case.surface_source}")
        lines.append(f"first max|V| >= 1e-3 m/s: {event / YEAR:.6f} yr")
        if case.interface_time.size:
            all_jump = (
                case.interface_sigma_left - case.interface_sigma_right
            ) / 1e6
            lines.append(
                "fault-face normal-stress jump max (all checkpoint samples): "
                f"{np.max(np.abs(all_jump)):.6e} MPa"
            )
            interface_stop = event - YEAR
            interface_mask = case.interface_time <= interface_stop
            jump = (
                case.interface_sigma_left[interface_mask]
                - case.interface_sigma_right[interface_mask]
            ) / 1e6
            if jump.size:
                lines.append(
                    "pre-event fault-face normal-stress jump RMS/max: "
                    f"{np.sqrt(np.mean(jump**2)):.6e} / "
                    f"{np.max(np.abs(jump)):.6e} MPa"
                )
        if reference is not None:
            stop = min(event, _reference_event_time(reference)) - YEAR
            mask = case.time <= stop
            sigma = case.normal_stress[iy, mask] / 1e6
            ref_sigma = np.interp(case.time[mask], reference[:, 0], reference[:, 4])
            difference = sigma - ref_sigma
            lines.append(f"pre-event normal-stress RMS error: {np.sqrt(np.mean(difference**2)):.6e} MPa")
            lines.append(f"pre-event normal-stress max error: {np.max(np.abs(difference)):.6e} MPa")
            early = case.time[mask] <= 150.0 * YEAR
            if np.any(early):
                lines.append(
                    "0-150 yr normal-stress RMS error: "
                    f"{np.sqrt(np.mean(difference[early]**2)):.6e} MPa"
                )
                lines.append(
                    "0-150 yr normal-stress max error: "
                    f"{np.max(np.abs(difference[early])):.6e} MPa"
                )
            event_window = np.abs(case.time - event) <= YEAR
            reference_event = _reference_event_time(reference)
            reference_window = np.abs(reference[:, 0] - reference_event) <= YEAR
            if np.any(event_window) and np.any(reference_window):
                case_event_sigma = case.normal_stress[iy, event_window] / 1e6
                ref_event_sigma = reference[reference_window, 4]
                lines.append(
                    "first-event normal-stress range (case/reference): "
                    f"[{case_event_sigma.min():.6f}, {case_event_sigma.max():.6f}] / "
                    f"[{ref_event_sigma.min():.6f}, {ref_event_sigma.max():.6f}] MPa"
                )
        lines.append("")
    if len(cases) == 2 and all(case.surface_time.size for case in cases):
        stop = min(_event_time(case.time, case.velocity) for case in cases) - YEAR
        lines.append("[surface difference between the two cases before first event]")
        first, second = cases
        start = max(first.surface_time[0], second.surface_time[0])
        sample_time = first.surface_time[
            (first.surface_time >= start) & (first.surface_time <= stop)
        ]
        if sample_time.size == 0:
            lines.append("no overlapping pre-event surface samples")
            destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return
        for index, name in enumerate(SURFACE_NAMES):
            for component in ("disp1", "disp2"):
                left = np.interp(
                    sample_time,
                    first.surface_time,
                    getattr(first, f"surface_{component}")[index],
                )
                right = np.interp(sample_time, second.surface_time,
                                  getattr(second, f"surface_{component}")[index])
                difference = np.abs(left - right)
                lines.append(
                    f"x={name} km {component} max difference: "
                    f"{np.max(difference):.6e} m"
                )
                early = sample_time <= 150.0 * YEAR
                if np.any(early):
                    lines.append(
                        f"x={name} km {component} max difference (<=150 yr): "
                        f"{np.max(difference[early]):.6e} m"
                    )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="+", type=Path, help="case or output directories")
    parser.add_argument("--label", action="append", help="plot label; repeat for each case")
    parser.add_argument("--station-km", type=float, default=10.0)
    parser.add_argument("--reference-surface-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("bp3_diagnostics"))
    args = parser.parse_args(argv)
    if args.label and len(args.label) != len(args.cases):
        parser.error("--label must be supplied once per case")
    labels = args.label or [None] * len(args.cases)
    cases = [
        load_case(path, label, args.station_km)
        for path, label in zip(args.cases, labels)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference, reference_path = _reference_fault(cases, args.station_km)
    reference_surface = _load_reference_surface(args.reference_surface_dir)
    _plot_normal(cases, args.station_km, reference, args.output_dir / "normal_stress_diagnostic.png")
    _plot_surface(cases, reference_surface, args.output_dir / "surface_displacement_diagnostic.png")
    _plot_interface(
        cases,
        args.station_km,
        args.output_dir / "interface_normal_stress_jump.png",
    )
    _write_report(cases, args.station_km, reference, reference_path, args.output_dir / "diagnostic_summary.txt")
    print(f"Wrote diagnostics to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
