"""End-to-end smoke test for the static BP3 deep-corner runner."""

import json
import sys

from examples import run_bp3_deep_corner_diagnostic as runner


def test_deep_corner_runner_writes_profiles_and_incremental_summary(
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
            "run_bp3_deep_corner_diagnostic.py",
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
        "lz160_profiles.npz",
        "lz160_summary.json",
        "summary.json",
        "summary.md",
    }
    assert expected.issubset(path.name for path in tmp_path.iterdir())
    payload = json.loads((tmp_path / "summary.json").read_text())
    assert payload["cases"]["lz160"]["dof_count"] > 0
    assert (
        payload["cases"]["lz160"]["source_metrics"]["bottom_endpoint"]
        ["equivalent_width_km"]
        == 1.0
    )
