"""End-to-end smoke test for static BP3 x-core convergence."""

import json
import sys

from examples import run_bp3_x_core_convergence as runner


def test_x_core_convergence_runs_cases_sequentially(monkeypatch, tmp_path):
    cases = (
        runner.XCoreCase("coarse", 150.0, 61, 41),
        runner.XCoreCase("fine", 100.0, 81, 61),
    )
    monkeypatch.setattr(runner, "CORE_CASES", cases)
    monkeypatch.setattr(
        runner,
        "DEPTH_CASE",
        runner.DepthCase("lz15", 15.0, 101),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_bp3_x_core_convergence.py",
            "--cases",
            "coarse",
            "fine",
            "--output-dir",
            str(tmp_path),
            "--xsize-km",
            "20",
            "--wf-km",
            "12",
            "--x-inner-km",
            "6",
            "--y-inner-km",
            "8",
            "--y-inner-points",
            "81",
            "--fixed-deep-depth-km",
            "10",
        ],
    )

    runner.main()

    expected = {
        "coarse_traction_audit.npz",
        "fine_traction_audit.npz",
        "coarse_summary.json",
        "fine_summary.json",
        "summary.json",
        "diagnostic_summary.md",
    }
    assert expected.issubset(path.name for path in tmp_path.iterdir())
    payload = json.loads((tmp_path / "summary.json").read_text())
    assert set(payload["results"]) == {"coarse", "fine"}
    assert "fine_vs_coarse" in payload["convergence"]
    for case in cases:
        result = payload["results"][case.name]
        assert result["dof_count"] > 0
        assert result["min_dx_m"] > 0.0
        assert result["direct_over_production_nucleation_self"] > 0.0
