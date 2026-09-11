"""End-to-end smoke test for the BP3 fault-traction audit."""

import json
import sys

import numpy as np

from examples import run_bp3_fault_traction_audit as runner


def test_fault_traction_audit_writes_unsymmetrized_recoveries(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        runner,
        "DEFAULT_CASES",
        (runner.DepthCase("lz160", 15.0, 101),),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_bp3_fault_traction_audit.py",
            "--cases",
            "lz160",
            "--output-dir",
            str(tmp_path),
            "--xsize-km",
            "20",
            "--nx",
            "81",
            "--wf-km",
            "12",
            "--x-inner-km",
            "6",
            "--y-inner-km",
            "8",
            "--x-inner-points",
            "61",
            "--y-inner-points",
            "81",
            "--fixed-deep-depth-km",
            "10",
        ],
    )

    runner.main()

    expected = {
        "lz160_traction_audit.npz",
        "lz160_summary.json",
        "summary.json",
        "diagnostic_summary.md",
    }
    assert expected.issubset(path.name for path in tmp_path.iterdir())
    payload = json.loads((tmp_path / "summary.json").read_text())
    case = payload["cases"]["lz160"]
    assert case["dof_count"] > 0
    assert np.isfinite(case["current_face_jump_relative_l2"])
    assert np.isfinite(case["direct_trace_face_jump_relative_l2"])

    with np.load(tmp_path / "lz160_traction_audit.npz") as data:
        assert data["tau_production"].shape == data["tau_direct_left"].shape
        np.testing.assert_allclose(
            data["tau_production"],
            0.5 * (data["tau_current_left"] + data["tau_current_right"]),
            rtol=2e-12,
            atol=1e-8,
        )
