import json
import sys

import numpy as np

from examples import run_bp3_frozen_operator_counterfactual as runner


def test_frozen_operator_counterfactual_smoke(monkeypatch, tmp_path):
    cases = (
        runner.XCoreCase("coarse", 150.0, 61, 41),
        runner.XCoreCase("fine", 100.0, 81, 61),
    )
    monkeypatch.setattr(runner, "CORE_CASES", cases)
    monkeypatch.setattr(
        runner, "DEPTH_CASE", runner.DEPTH_CASE.__class__("lz15", 15.0, 101)
    )

    years = np.arange(1.0, 201.0)
    time = years * runner.SECONDS_PER_YEAR
    ny = 101
    velocity = np.full((ny, years.size), 1e-10)
    velocity[80:, :] = 1e-9
    # Two threshold-defined earthquakes.
    velocity[:80, 19] = 1e-2
    velocity[:80, 179] = 1e-2
    # An arrested pulse followed later by renewed acceleration at 5 km.
    pulse = 3e-9 * np.exp(-0.5 * ((years - 80.0) / 4.0) ** 2)
    renewed = np.where(
        years > 130.0, 1e-10 * np.exp((years - 130.0) / 4.0), 0.0
    )
    velocity[50, :] += pulse + renewed
    theta = np.full_like(velocity, 8e6)
    sigma = np.full_like(velocity, 50e6)
    traction = np.full_like(velocity, 30e6)
    dataall = tmp_path / "dataall.npz"
    np.savez(
        dataall, tm=time, Vm=velocity, thetam=theta,
        sigmam=sigma, taum=traction,
    )
    output = tmp_path / "diagnostic"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_bp3_frozen_operator_counterfactual.py",
            str(dataall),
            "--cases", "coarse", "fine",
            "--output-dir", str(output),
            "--xsize-km", "20",
            "--wf-km", "12",
            "--x-inner-km", "6",
            "--y-inner-km", "8",
            "--y-inner-points", "81",
            "--precursor-depth-km", "5",
            "--probe-depths-km", "4", "5", "6",
            "--band-top-km", "4",
            "--band-bottom-km", "6",
            "--runaway-threshold", "1e-8",
        ],
    )

    runner.main()

    assert {
        "summary.json", "diagnostic_summary.md",
        "probe_comparison.csv", "counterfactual_profiles.npz",
        "coarse_counterfactual.npz", "fine_counterfactual.npz",
        "coarse_summary.json", "fine_summary.json",
    }.issubset(path.name for path in output.iterdir())
    summary = json.loads((output / "summary.json").read_text())
    assert set(summary["cases"]) == {"coarse", "fine"}
    assert set(summary["selected_snapshots"]) == {
        "precursor_peak", "following_minimum", "runaway_onset"
    }
    for case in cases:
        assert summary["cases"][case.name]["dof_count"] > 0
