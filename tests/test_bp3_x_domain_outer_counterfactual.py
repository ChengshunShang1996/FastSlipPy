"""Smoke tests for the frozen x-domain/outer-resolution diagnostic."""

import json
import sys

import numpy as np

from examples import run_bp3_x_domain_outer_counterfactual as runner


def _write_synthetic_history(path, ny):
    year = runner.SECONDS_PER_YEAR
    time = np.arange(1.0, 41.0) * year
    velocity = np.full((ny, time.size), 1e-12)
    velocity[:, 5] = 2e-3
    velocity[:, 33] = 2e-3
    focus = 10
    velocity[focus, 18] = 5e-9
    velocity[focus, 24] = 1e-11
    velocity[focus, 30] = 2e-8
    # Give the frozen modes a smooth, nonzero spatial footprint.
    profile = np.exp(-0.5 * ((np.arange(ny) - focus) / 3.0) ** 2)
    velocity[:, 18] += 4e-9 * profile
    velocity[:, 24] += 1e-11 * profile
    velocity[:, 30] += 2e-8 * profile
    np.savez(
        path,
        tm=time,
        Vm=velocity,
        taum=np.full_like(velocity, 30e6),
        sigmam=np.full_like(velocity, 50e6),
        thetam=np.full_like(velocity, 0.008 / 1e-9),
    )


def test_factorial_effects_recover_main_effects_and_interaction():
    cases = (
        runner.XDomainOuterCase("a", 20.0, 5),
        runner.XDomainOuterCase("b", 20.0, 10),
        runner.XDomainOuterCase("c", 30.0, 5),
        runner.XDomainOuterCase("d", 30.0, 10),
    )
    fields = (
        "focus_predicted_dlnV_per_year",
        "mode_weighted_dlnV_per_year",
        "velocity_weighted_arrest_fraction",
        "effective_stiffness_pa_per_m",
        "effective_over_critical_stiffness",
    )
    values = {"a": 1.0, "b": 3.0, "c": 4.0, "d": 10.0}
    summaries = {
        name: {"snapshots": {"peak": {field: value for field in fields}}}
        for name, value in values.items()
    }
    effects = runner._factorial_effects(cases, summaries, ["peak"])
    item = effects["peak"]["mode_weighted_dlnV_per_year"]
    assert item["domain_effect_at_outer_5"] == 3.0
    assert item["domain_effect_at_outer_10"] == 7.0
    assert item["outer_effect_at_xsize_20_km"] == 2.0
    assert item["outer_effect_at_xsize_30_km"] == 6.0
    assert item["interaction"] == 4.0


def test_small_counterfactual_runs_full_factorial(monkeypatch, tmp_path):
    cases = (
        runner.XDomainOuterCase("x20_o5", 20.0, 5),
        runner.XDomainOuterCase("x20_o10", 20.0, 10),
        runner.XDomainOuterCase("x30_o5", 30.0, 5),
        runner.XDomainOuterCase("x30_o10", 30.0, 10),
    )
    monkeypatch.setattr(runner, "DEFAULT_CASES", cases)
    dataall = tmp_path / "dataall.npz"
    _write_synthetic_history(dataall, ny=31)
    output = tmp_path / "diagnostic"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_bp3_x_domain_outer_counterfactual.py",
            str(dataall),
            "--output-dir", str(output),
            "--cases", *(case.name for case in cases),
            "--ysize-km", "6",
            "--ny", "31",
            "--wf-km", "5",
            "--x-inner-km", "4",
            "--x-inner-points", "21",
            "--y-inner-km", "4",
            "--y-inner-points", "21",
            "--precursor-depth-km", "2",
            "--band-top-km", "1",
            "--band-bottom-km", "3",
            "--probe-depths-km", "1", "2", "3",
            "--mesh-probe-distances-km", "1", "4",
        ],
    )

    runner.main()

    payload = json.loads((output / "summary.json").read_text())
    assert set(payload["cases"]) == {case.name for case in cases}
    assert set(payload["factorial_effects"]) == {
        "precursor_peak", "following_minimum", "runaway_onset"
    }
    for case in cases:
        item = payload["cases"][case.name]
        assert item["geometry"]["nx"] == case.nx(21)
        assert item["geometry"]["maximum_dx_m"] > 0.0
        for snapshot in payload["factorial_effects"]:
            metrics = item["snapshots"][snapshot]
            assert np.isfinite(metrics["mode_weighted_dlnV_per_year"])
            assert np.isfinite(metrics["effective_over_critical_stiffness"])
            assert 0.0 <= metrics["velocity_weighted_arrest_fraction"] <= 1.0
    assert (output / "counterfactual_profiles.npz").is_file()
    assert (output / "probe_comparison.csv").is_file()
    assert (output / "mesh_geometry.csv").is_file()
    assert (output / "diagnostic_summary.md").is_file()
