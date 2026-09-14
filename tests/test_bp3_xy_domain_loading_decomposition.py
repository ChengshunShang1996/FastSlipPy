"""Tests for the frozen BP3 xsize-by-ysize loading decomposition."""

import json
import sys

import numpy as np

from tools.bp3.studies import run_bp3_xy_domain_loading_decomposition as runner


def _write_synthetic_history(path, ny):
    year = runner.SECONDS_PER_YEAR
    time = np.arange(1.0, 41.0) * year
    velocity = np.full((ny, time.size), 1e-12)
    velocity[:, 5] = 2e-3
    velocity[:, 33] = 2e-3
    focus = 10
    profile = np.exp(-0.5 * ((np.arange(ny) - focus) / 3.0) ** 2)
    velocity[:, 18] += 5e-9 * profile
    velocity[:, 24] += 1e-11 * profile
    velocity[:, 30] += 2e-8 * profile
    np.savez(
        path,
        tm=time,
        Vm=velocity,
        sigmam=np.full_like(velocity, 50e6),
        thetam=np.full_like(velocity, 0.008 / 1e-9),
    )


def test_default_case_geometry_targets_320_by_100():
    cases = {case.name: case for case in runner.CASE_CATALOG}
    target = cases["x320_y100"]
    assert target.xsize_km == 320.0
    assert target.ysize_km == 100.0
    assert target.nx == 901
    assert target.ny == 545
    assert 0.5 * target.xsize_km / target.ysize_km == 1.6


def test_deep_creep_components_are_exhaustive_and_identify_added_depth():
    y = np.arange(0.0, 9.0)
    velocity = np.column_stack((y + 1.0, 2.0 * y + 3.0))
    components = runner.split_fault_velocity_components(
        y,
        velocity,
        band_top=1.0,
        band_bottom=2.0,
        creep_start=4.0,
        deep_split_1=6.0,
        deep_split_2=7.0,
    )
    assert tuple(components) == runner.FAULT_LOADING_COMPONENTS
    np.testing.assert_array_equal(sum(components.values()), velocity)
    active_counts = sum(item != 0.0 for item in components.values())
    np.testing.assert_array_equal(active_counts, np.ones_like(velocity))
    assert np.all(components["deep_creep_far"][y > 7.0] != 0.0)


def test_nominal_boundary_node_is_not_lost_to_floating_roundoff():
    y = np.array([
        0.0, 10e3, 15e3 + 2e-12, 20e3, 40e3, 100e3, 160e3,
    ])
    velocity = np.ones((y.size, 1))
    components = runner.split_fault_velocity_components(
        y,
        velocity,
        band_top=10e3,
        band_bottom=15e3,
        creep_start=40e3,
        deep_split_1=100e3,
        deep_split_2=160e3,
    )
    assert components["nucleation_band"][2, 0] == 1.0
    assert components["lower_seismogenic"][2, 0] == 0.0
    mask = runner._closed_interval_mask(y, 10e3, 15e3)
    np.testing.assert_array_equal(mask, [False, True, True, False, False, False, False])


def test_interpolation_extends_prescribed_creep_below_source_domain():
    source_y = np.array([0.0, 1.0, 2.0, 3.0])
    target_y = np.arange(0.0, 7.0)
    velocity = np.array([[1.0], [2.0], [3.0], [4.0]])
    theta = np.full_like(velocity, 8.0)
    sigma = np.full_like(velocity, 50.0)
    mapped_v, mapped_theta, mapped_sigma = runner.interpolate_frozen_states(
        source_y,
        velocity,
        theta,
        sigma,
        target_y,
        creep_start=3.0,
        creep_velocity=0.5,
        state_length=4.0,
    )
    np.testing.assert_array_equal(mapped_v[target_y >= 3.0], 0.5)
    np.testing.assert_array_equal(mapped_theta[target_y >= 3.0], 8.0)
    assert np.all(mapped_sigma > 0.0)


def test_small_xy_counterfactual_writes_closed_decomposition(monkeypatch, tmp_path):
    cases = (
        runner.XYDomainCase("x20_y6", 20.0, 6.0, 31, 31),
        runner.XYDomainCase("x30_y8", 30.0, 8.0, 41, 41),
    )
    monkeypatch.setattr(runner, "CASE_CATALOG", cases)
    monkeypatch.setattr(runner, "DEFAULT_CASE_NAMES", tuple(case.name for case in cases))
    dataall = tmp_path / "dataall.npz"
    _write_synthetic_history(dataall, ny=31)
    output = tmp_path / "diagnostic"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_bp3_xy_domain_loading_decomposition.py",
            str(dataall),
            "--output-dir", str(output),
            "--cases", *(case.name for case in cases),
            "--source-xsize-km", "20",
            "--source-ysize-km", "6",
            "--source-nx", "31",
            "--source-ny", "31",
            "--x-inner-km", "4",
            "--x-inner-points", "21",
            "--y-inner-km", "4",
            "--y-inner-points", "21",
            "--wf-km", "4.5",
            "--deep-split-km", "5", "5.5",
            "--band-top-km", "1",
            "--band-bottom-km", "3",
            "--precursor-depth-km", "2",
            "--branch-age-years", "10",
            "--probe-depths-km", "1", "2", "3",
        ],
    )

    runner.main()

    payload = json.loads((output / "summary.json").read_text())
    assert set(payload["cases"]) == {case.name for case in cases}
    for case in cases:
        item = payload["cases"][case.name]
        assert item["geometry"]["nx"] == case.nx
        assert item["geometry"]["ny"] == case.ny
        closure = item["traction_decomposition_closure"]
        assert closure["tau_relative_l2"] < 1e-10
        assert closure["sigma_relative_l2"] < 1e-10
        for snapshot in payload["configuration"]["snapshots"]:
            metrics = item["snapshots"][snapshot]
            assert np.isfinite(metrics["mode_weighted_dlnV_per_year"])
            assert 0.0 <= metrics["velocity_weighted_arrest_fraction"] <= 1.0
            assert set(metrics["loading_decomposition"]) == {
                "side_boundary",
                *runner.FAULT_LOADING_COMPONENTS,
                "full_fault",
                "state_evolution",
            }
            assert metrics["acceleration_decomposition_max_abs_per_year"] < 1e-6
    assert (output / "loading_decomposition.csv").is_file()
    assert (output / "probe_comparison.csv").is_file()
    assert (output / "counterfactual_profiles.npz").is_file()
    assert (output / "diagnostic_summary.md").is_file()
