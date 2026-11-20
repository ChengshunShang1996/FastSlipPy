"""Check target-stress acceptance and exhaustion on a single, small Kratos RVE."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from verify_single_rve import prepare_small_case

from fastslippy.pre_processing.model_parameters import DEMCouplingParameters
from fastslippy.solver.kratos_dem_backend import (
    KratosDEMBackendFactory,
    NormalConsolidationError,
)


def check_case(directory, settings, target, expected_status):
    factory = KratosDEMBackendFactory(settings, directory)
    try:
        rve = factory(0, target, 0.0)
    except NormalConsolidationError:
        if expected_status != "max_steps":
            raise
    else:
        try:
            if expected_status != "converged":
                raise AssertionError(
                    "The factory returned a usable, unconsolidated RVE."
                )
            for wall in (rve.stage.top_wall, rve.stage.bottom_wall):
                np.testing.assert_allclose(
                    [n.X - n.X0 for n in wall.Nodes], 0.0, atol=1e-14, rtol=0
                )
            np.testing.assert_equal(rve.stage.shear_displacement, 0.0)
        finally:
            rve.close()

    path = directory / "rve_00000" / "normal_consolidation.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    stress_tol = (
        settings.consolidation_stress_atol + target * settings.consolidation_stress_rtol
    )
    streak = 0
    for index, row in enumerate(history, start=1):
        stress_errors = [abs(float(row[k]) - target) for k in ("top_Pa", "bottom_Pa")]
        speeds = [
            float(row[k])
            for k in ("wall_speed_m_s", "grain_speed_m_s", "rim_speed_m_s")
        ]
        passed = (
            max(stress_errors + [float(row["imbalance_Pa"])]) <= stress_tol
            and max(speeds) <= settings.consolidation_speed_tol
        )
        streak = streak + 1 if passed else 0
        np.testing.assert_equal(int(row["stable_steps"]), streak)
        if index < len(history):
            np.testing.assert_equal(row["status"], "monitoring")
            np.testing.assert_array_less(streak, settings.consolidation_window_steps)
    last = history[-1]
    np.testing.assert_equal(last["status"], expected_status)
    if expected_status == "converged":
        np.testing.assert_equal(streak, settings.consolidation_window_steps)
    else:
        np.testing.assert_equal(len(history), settings.consolidation_max_steps)
    print(
        f"{expected_status}: target {target:g} Pa, top {float(last['top_Pa']):.4f} Pa, "
        f"bottom {float(last['bottom_Pa']):.4f} Pa after {len(history)} steps"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("output/normal-consolidation")
    )
    out = parser.parse_args().output.resolve()
    width = prepare_small_case(out / "input")
    settings = DEMCouplingParameters(
        case_directory=str(out / "input"), layer_length=width
    )
    check_case(out / "5kPa", settings, 5000.0, "converged")
    check_case(out / "15kPa", settings, 15000.0, "converged")
    settings.consolidation_max_steps = 120
    check_case(out / "step-limit", settings, 15000.0, "max_steps")


if __name__ == "__main__":
    main()
