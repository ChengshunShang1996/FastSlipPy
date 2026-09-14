"""Limit-cycle diagnostics for BP3-QD station histories.

The functions in this module deliberately operate on one on-fault station.
This keeps the memory footprint modest even when ``dataall.npz`` contains a
large two-dimensional history and makes the same phase portraits available
for both FastSlipPy and the published DFRA station files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0


@dataclass(frozen=True)
class StationHistory:
    """One-station history in the BP3 plotting sign convention."""

    label: str
    time_years: np.ndarray
    log10_velocity: np.ndarray
    shear_stress_mpa: np.ndarray
    normal_stress_mpa: np.ndarray
    log10_theta: np.ndarray

    def __post_init__(self) -> None:
        arrays = (
            self.time_years,
            self.log10_velocity,
            self.shear_stress_mpa,
            self.normal_stress_mpa,
            self.log10_theta,
        )
        if any(np.asarray(value).ndim != 1 for value in arrays):
            raise ValueError("Station-history arrays must be one-dimensional.")
        if len({np.asarray(value).size for value in arrays}) != 1:
            raise ValueError("Station-history arrays must have equal lengths.")
        if self.time_years.size < 2:
            raise ValueError("A station history needs at least two samples.")
        if np.any(~np.isfinite(np.concatenate(arrays))):
            raise ValueError("Station-history arrays must be finite.")
        if np.any(np.diff(self.time_years) <= 0.0):
            raise ValueError("Station-history times must be strictly increasing.")


@dataclass(frozen=True)
class LimitCycle:
    """One event-to-event portion of a station history."""

    number: int
    start: int
    stop: int
    recurrence_years: float

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop + 1)


def resolve_dataall(path: Path) -> Path:
    """Resolve a result directory, output directory, or NPZ file."""

    path = path.expanduser().resolve()
    candidates = (path, path / "dataall.npz", path / "output" / "dataall.npz")
    for candidate in candidates:
        if candidate.is_file() and candidate.name == "dataall.npz":
            return candidate
    raise FileNotFoundError(f"Cannot find dataall.npz below {path}")


def fault_station_index(depth_km: float, inner_spacing_m: float, ny: int) -> int:
    """Return the node index for a depth inside the uniform refined region."""

    if depth_km < 0.0:
        raise ValueError("depth_km must be non-negative.")
    if inner_spacing_m <= 0.0:
        raise ValueError("inner_spacing_m must be positive.")
    index = int(round(depth_km * 1000.0 / inner_spacing_m))
    if not 0 <= index < ny:
        raise ValueError(
            f"Depth {depth_km:g} km maps to index {index}, outside Ny={ny}."
        )
    return index


def load_fastslippy_station(
    path: Path,
    *,
    label: str,
    depth_km: float = 10.0,
    inner_spacing_m: float = 50.0,
    velocity_floor: float = 1e-30,
) -> StationHistory:
    """Load one station from a FastSlipPy ``dataall.npz`` history.

    FastSlipPy stores positive ``taum`` for the 60-degree normal case, whereas
    the BP3/DFRA files use negative shear stress.  The sign is converted here
    so that the phase portraits are directly comparable.
    """

    dataall = resolve_dataall(path)
    with np.load(dataall) as data:
        required = {"tm", "Vm", "taum", "sigmam", "thetam"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"dataall is missing fields: {sorted(missing)}")

        raw_time = np.asarray(data["tm"], dtype=float)
        valid = np.isfinite(raw_time) & (raw_time > 0.0)
        if np.count_nonzero(valid) < 2:
            raise ValueError(f"{dataall} has fewer than two valid snapshots.")
        index = fault_station_index(
            depth_km, inner_spacing_m, int(data["Vm"].shape[0])
        )
        time = raw_time[valid] / SECONDS_PER_YEAR
        velocity = np.asarray(data["Vm"][index, valid], dtype=float)
        shear = -np.asarray(data["taum"][index, valid], dtype=float) / 1e6
        normal = np.asarray(data["sigmam"][index, valid], dtype=float) / 1e6
        theta = np.asarray(data["thetam"][index, valid], dtype=float)

    return StationHistory(
        label=label,
        time_years=time,
        log10_velocity=np.log10(np.maximum(np.abs(velocity), velocity_floor)),
        shear_stress_mpa=shear,
        normal_stress_mpa=normal,
        log10_theta=np.log10(np.maximum(theta, velocity_floor)),
    )


def dfra_filename(depth_km: float) -> str:
    """Return the standard 60-degree-normal DFRA station filename."""

    depth_token = int(round(depth_km * 10.0))
    return f"60-dfra-onfault-dp{depth_token:03d}-normal.txt"


def load_dfra_station(
    reference_dir: Path,
    *,
    depth_km: float = 10.0,
    label: str = "DFRA",
) -> StationHistory:
    """Load a published BP3/DFRA on-fault station history."""

    path = reference_dir.expanduser().resolve()
    if path.is_file():
        station_file = path
    else:
        candidates = (
            path / dfra_filename(depth_km),
            path / "output" / dfra_filename(depth_km),
        )
        station_file = next((item for item in candidates if item.is_file()), None)
        if station_file is None:
            raise FileNotFoundError(
                f"Cannot find {dfra_filename(depth_km)} below {path}"
            )

    values = np.loadtxt(station_file, dtype=float)
    if values.ndim != 2 or values.shape[1] < 6:
        raise ValueError(f"DFRA file must have at least six columns: {station_file}")
    return StationHistory(
        label=label,
        time_years=values[:, 0] / SECONDS_PER_YEAR,
        log10_velocity=values[:, 2],
        shear_stress_mpa=values[:, 3],
        normal_stress_mpa=values[:, 4],
        log10_theta=values[:, 5],
    )


def event_peak_indices(
    history: StationHistory, *, event_threshold: float = 1e-3
) -> np.ndarray:
    """Detect one peak per contiguous interval above the event threshold."""

    if event_threshold <= 0.0:
        raise ValueError("event_threshold must be positive.")
    active = history.log10_velocity >= np.log10(event_threshold)
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    ends = np.flatnonzero(active & ~np.r_[active[1:], False])
    peaks = [
        int(start + np.argmax(history.log10_velocity[start : end + 1]))
        for start, end in zip(starts, ends)
    ]
    return np.asarray(peaks, dtype=int)


def limit_cycles(
    history: StationHistory, *, event_threshold: float = 1e-3
) -> list[LimitCycle]:
    """Split a station history into complete peak-to-peak limit-cycle paths."""

    peaks = event_peak_indices(history, event_threshold=event_threshold)
    return [
        LimitCycle(
            number=number + 1,
            start=int(start),
            stop=int(stop),
            recurrence_years=float(
                history.time_years[stop] - history.time_years[start]
            ),
        )
        for number, (start, stop) in enumerate(zip(peaks[:-1], peaks[1:]))
    ]


def representative_reference_cycle(cycles: list[LimitCycle], count: int = 4) -> int:
    """Choose a late cycle whose recurrence is closest to the late median."""

    if not cycles:
        raise ValueError("At least one complete reference cycle is required.")
    if count < 1:
        raise ValueError("count must be at least one.")
    candidates = cycles[-count:]
    recurrence = np.asarray([cycle.recurrence_years for cycle in candidates])
    median = float(np.median(recurrence))
    return int(
        min(
            range(len(candidates)),
            key=lambda index: abs(candidates[index].recurrence_years - median),
        )
        + len(cycles)
        - len(candidates)
    )


def concatenate_cycle_paths(
    history: StationHistory,
    cycles: list[LimitCycle],
    attribute: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate complete paths with NaN separators for plotting."""

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    values = np.asarray(getattr(history, attribute), dtype=float)
    for cycle in cycles:
        x_parts.extend((history.log10_velocity[cycle.slice], np.asarray([np.nan])))
        y_parts.extend((values[cycle.slice], np.asarray([np.nan])))
    if not x_parts:
        return np.empty(0), np.empty(0)
    return np.concatenate(x_parts), np.concatenate(y_parts)
