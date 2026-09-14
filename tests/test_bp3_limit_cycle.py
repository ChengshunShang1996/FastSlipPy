from pathlib import Path

import numpy as np

from fastslippy.utilities.bp3_limit_cycle import (
    SECONDS_PER_YEAR,
    dfra_filename,
    event_peak_indices,
    fault_station_index,
    limit_cycles,
    load_dfra_station,
    load_fastslippy_station,
    representative_reference_cycle,
)
from examples.plot_bp3_limit_cycle_evolution import (
    DEFAULT_600_CASE,
    DEFAULT_800_CASE,
    DEFAULT_REFERENCE_DIR,
    parse_arguments,
)


def _write_fast_history(path: Path) -> None:
    time_years = np.arange(12, dtype=float) + 1.0
    velocity = np.full((3, 12), 1e-12)
    velocity[2, [1, 2, 6, 10]] = [1e-2, 2e-1, 3e-1, 2e-1]
    shear = np.tile(np.linspace(20e6, 30e6, 12), (3, 1))
    normal = np.tile(np.linspace(49e6, 50e6, 12), (3, 1))
    theta = np.tile(np.linspace(1e6, 2e6, 12), (3, 1))
    np.savez(
        path,
        tm=time_years * SECONDS_PER_YEAR,
        Vm=velocity,
        taum=shear,
        sigmam=normal,
        thetam=theta,
    )


def test_load_fast_station_and_detect_cycles(tmp_path):
    dataall = tmp_path / "dataall.npz"
    _write_fast_history(dataall)
    history = load_fastslippy_station(
        dataall, label="test", depth_km=0.1, inner_spacing_m=50.0
    )

    np.testing.assert_array_equal(event_peak_indices(history), [2, 6, 10])
    cycles = limit_cycles(history)
    assert [cycle.recurrence_years for cycle in cycles] == [4.0, 4.0]
    assert history.shear_stress_mpa[0] == -20.0
    assert history.normal_stress_mpa[-1] == 50.0


def test_load_dfra_station(tmp_path):
    values = np.asarray(
        [
            [SECONDS_PER_YEAR, 0.0, -12.0, -30.0, 50.0, 8.0],
            [2 * SECONDS_PER_YEAR, 0.0, -11.0, -31.0, 49.0, 7.0],
        ]
    )
    np.savetxt(tmp_path / dfra_filename(10.0), values)
    history = load_dfra_station(tmp_path, depth_km=10.0)

    np.testing.assert_allclose(history.time_years, [1.0, 2.0])
    np.testing.assert_allclose(history.log10_theta, [8.0, 7.0])


def test_reference_cycle_is_late_and_median_like(tmp_path):
    dataall = tmp_path / "dataall.npz"
    _write_fast_history(dataall)
    history = load_fastslippy_station(
        dataall, label="test", depth_km=0.1, inner_spacing_m=50.0
    )
    cycles = limit_cycles(history)
    assert representative_reference_cycle(cycles, count=2) == 0


def test_station_index_validation():
    assert fault_station_index(10.0, 50.0, 651) == 200
    try:
        fault_station_index(100.0, 50.0, 100)
    except ValueError as error:
        assert "outside" in str(error)
    else:
        raise AssertionError("Out-of-range depth should fail.")


def test_no_cli_arguments_select_ide_defaults():
    args = parse_arguments([])
    assert args.case[0][0] == "800x160 km (correct branch)"
    assert Path(args.case[0][1]) == DEFAULT_800_CASE
    assert Path(args.case[1][1]) == DEFAULT_600_CASE
    assert args.reference_dir == DEFAULT_REFERENCE_DIR

    no_animation = parse_arguments(["--no-animation", "--depth-km", "15"])
    assert no_animation.no_animation
    assert no_animation.depth_km == 15.0
    assert len(no_animation.case) == 2
