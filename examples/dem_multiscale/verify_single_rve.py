"""Displacement-controlled check of one real 2-D Kratos RVE (SI units)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
from prepare_case import CASE_DIRECTORY, COARSE_RADIUS, write_packing

from fastslippy.pre_processing.model_parameters import DEMCouplingParameters
from fastslippy.solver.kratos_dem_backend import KratosDEMRVE


def prepare_small_case(directory):
    # Four staggered, initially touching rows. Keep the periodic seam away
    # from the wall particles so coordinates also measure their displacement.
    radius = COARSE_RADIUS
    width = 8 * radius
    rows = [
        [
            (
                (2 * column + 0.25 + row % 2) * radius,
                row * math.sqrt(3) * radius,
                radius,
            )
            for column in range(4)
        ]
        for row in range(4)
    ]
    write_packing(directory, rows[1] + rows[2], rows[0], rows[3], [])
    shutil.copyfile(
        CASE_DIRECTORY / "MaterialsDEM.json", directory / "MaterialsDEM.json"
    )
    parameters = json.loads((CASE_DIRECTORY / "ProjectParametersDEM.json").read_text())
    parameters["BoundingBoxMaxX"] = width
    (directory / "ProjectParametersDEM.json").write_text(
        json.dumps(parameters, indent=2)
    )
    return width


def wall_state(wall, km):
    positions = np.array([[n.X, n.Y] for n in wall.Nodes])
    displacement = positions - [[n.X0, n.Y0] for n in wall.Nodes]
    velocities = np.array(
        [list(n.GetSolutionStepValue(km.VELOCITY))[:2] for n in wall.Nodes]
    )
    force = np.sum(
        [list(n.GetSolutionStepValue(km.TOTAL_FORCES))[:2] for n in wall.Nodes], axis=0
    )
    return displacement, velocities, force, positions[:, 1].mean()


class ObservedRVE(KratosDEMRVE):
    """Sample the production subcycling path immediately after every solve."""

    def _single_step(self):
        km = self.KM
        stage = self.stage
        old_time = stage.time
        super()._single_step()
        info = stage.spheres_model_part.ProcessInfo
        dt = info[km.DELTA_TIME]
        disp, vel, force, top_y = wall_state(stage.top_wall, km)
        bottom_disp, bottom_vel, bottom_force, bottom_y = wall_state(
            stage.bottom_wall, km
        )
        self.integral += vel.mean(axis=0) * dt
        target = stage.prescribed_wall_velocity
        self.records.append(
            dict(
                phase=self.phase,
                time_s=info[km.TIME],
                dt_s=dt,
                clock_error_s=max(
                    abs(stage.time - info[km.TIME]), abs(stage.time - old_time - dt)
                ),
                top_ux_m=disp[:, 0].mean(),
                top_uy_m=disp[:, 1].mean(),
                bottom_ux_m=bottom_disp[:, 0].mean(),
                bottom_uy_m=bottom_disp[:, 1].mean(),
                top_vx_m_s=vel[:, 0].mean(),
                top_vy_m_s=vel[:, 1].mean(),
                bottom_vx_m_s=bottom_vel[:, 0].mean(),
                bottom_vy_m_s=bottom_vel[:, 1].mean(),
                wall_velocity_error_m_s=max(
                    abs(vel - target).max(), abs(bottom_vel).max()
                ),
                wall_displacement_error_m=max(
                    abs(disp - self.integral).max(), abs(bottom_disp).max()
                ),
                height_m=top_y - bottom_y,
                top_fx_N=force[0],
                top_fy_N=force[1],
                bottom_fx_N=bottom_force[0],
                bottom_fy_N=bottom_force[1],
                sigma_Pa=stage.measured_normal_traction,
                tau_Pa=stage.measured_shear_traction,
            )
        )


def verify(rve, width):
    rve.records = []
    rve.integral = np.zeros(2)
    dt = float(rve.stage._GetSolver().dt)
    phases = [
        ("compress", (0.0, -0.03), 200),
        ("hold_compression", (0.0, 0.0), 1200),
        # The moving wall makes the fractional step observable in its position.
        ("shear", (0.01, 0.0), 100.5),
        ("hold_shear", (0.0, 0.0), 1200),
    ]
    end_time = 0.0
    for name, velocity, steps in phases:
        rve.phase = name
        rve.stage.prescribed_wall_velocity = velocity
        start = len(rve.records)
        result = rve.advance(
            driving_shear_stress=0,
            effective_normal_stress=0,
            radiation_damping=0,
            duration=steps * dt,
            initial_velocity=0,
        )
        end_time += steps * dt
        samples = rve.records[start:]
        expected_steps = [dt] * int(steps)
        if steps % 1:
            expected_steps.append((steps % 1) * dt)
        np.testing.assert_allclose(
            [s["dt_s"] for s in samples], expected_steps, rtol=1e-10, atol=1e-17
        )
        np.testing.assert_allclose(samples[-1]["time_s"], end_time, rtol=0, atol=1e-13)
        np.testing.assert_allclose(
            result.slip_increment, velocity[0] * steps * dt, rtol=0, atol=1e-13
        )
        if name == "shear" and samples[-1]["top_fx_N"] >= 0:
            raise AssertionError(
                "Top-wall reaction must oppose the imposed positive shear motion."
            )

    for key, limit in (
        ("clock_error_s", 1e-13),
        ("wall_velocity_error_m_s", 1e-12),
        ("wall_displacement_error_m", 1e-12),
    ):
        np.testing.assert_array_less([s[key] for s in rve.records], limit)
    for traction, force, sign in (
        ("sigma_Pa", "top_fy_N", 1),
        ("tau_Pa", "top_fx_N", -1),
    ):
        np.testing.assert_allclose(
            [s[traction] for s in rve.records],
            [sign * s[force] / width for s in rve.records],
            atol=1e-8,
        )

    for phase in ("hold_compression", "hold_shear"):
        tail = [s for s in rve.records if s["phase"] == phase][-100:]
        for s in tail:
            if s["sigma_Pa"] <= 0 or (phase == "hold_shear" and s["tau_Pa"] <= 0):
                raise AssertionError(f"Unexpected wall reaction during {phase}: {s}")
            if phase == "hold_compression":
                np.testing.assert_allclose(s["tau_Pa"], 0.0, atol=1e-6)
            for component in ("fx_N", "fy_N"):
                np.testing.assert_allclose(
                    s["top_" + component],
                    -s["bottom_" + component],
                    rtol=1e-3,
                    atol=1e-6,
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/single-rve"))
    out = parser.parse_args().output.resolve()
    width = prepare_small_case(out / "input")
    settings = DEMCouplingParameters(
        case_directory=str(out / "input"), layer_length=width
    )
    # This displacement-controlled calibration starts from the raw packing.
    rve = ObservedRVE(0, None, 0.0, settings, out)
    try:
        verify(rve, width)
    finally:
        rve.close()
        if rve.records:
            with (out / "history.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=rve.records[0])
                writer.writeheader()
                writer.writerows(rve.records)
    print(
        f"Single-RVE checks passed ({len(rve.records)} steps). "
        f"History: {out / 'history.csv'}"
    )


if __name__ == "__main__":
    main()
