"""Verify a near-static shear preload and handoff on a single Kratos RVE."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from verify_single_rve import prepare_small_case, wall_state

from fastslippy.pre_processing.model_parameters import DEMCouplingParameters
from fastslippy.solver.kratos_dem_backend import KratosDEMRVE, ShearPreloadError


def prepare(settings, directory, normal, drive, eta):
    return KratosDEMRVE(
        0,
        normal,
        0.0,
        settings,
        directory,
        initial_driving_shear_stress=drive,
        radiation_damping=eta,
    )


def hold(rve, normal, drive, eta):
    """Keep the preload tractions unchanged through 200 production substeps."""
    stage, km, cfg = rve.stage, rve.KM, rve.settings
    dt = float(stage._GetSolver().dt)
    np.testing.assert_equal([rve.loading_time, rve.loading_slip], [0.0, 0.0])
    normal_tol = cfg.consolidation_stress_atol + cfg.consolidation_stress_rtol * normal
    shear_tol = cfg.consolidation_stress_atol + cfg.consolidation_stress_rtol * abs(
        drive
    )
    samples = []
    accumulated = 0.0
    for step in range(201):
        if step:
            result = rve.advance(
                driving_shear_stress=drive,
                effective_normal_stress=normal,
                radiation_damping=eta,
                duration=dt,
                initial_velocity=0.0,
            )
            accumulated += result.slip_increment
        disp, vel, force, _ = wall_state(stage.top_wall, km)
        bottom_disp, _, bottom_force, _ = wall_state(stage.bottom_wall, km)
        shear = -force[0] / cfg.layer_length
        sigma = force[1] / cfg.layer_length
        vx = vel[:, 0].mean()
        measured_slip = disp[:, 0].mean() - bottom_disp[:, 0].mean()
        np.testing.assert_allclose(
            measured_slip, rve.loading_slip_origin + accumulated, rtol=0, atol=1e-13
        )
        np.testing.assert_allclose(rve.loading_slip, accumulated, rtol=0, atol=1e-14)
        np.testing.assert_allclose(rve.loading_time, step * dt, rtol=0, atol=1e-13)
        np.testing.assert_allclose(
            [sigma, -bottom_force[1] / cfg.layer_length],
            normal,
            rtol=0,
            atol=normal_tol,
        )
        np.testing.assert_allclose(
            [shear, bottom_force[0] / cfg.layer_length],
            drive - eta * vx,
            rtol=0,
            atol=shear_tol,
        )
        np.testing.assert_array_less(np.abs(vel), cfg.consolidation_speed_tol)
        if step == 1:
            for value, key, tol in (
                (sigma, "normal_Pa", normal_tol),
                (shear, "contact_Pa", shear_tol),
                (vx, "vx_m_s", cfg.consolidation_speed_tol),
            ):
                np.testing.assert_allclose(value, samples[0][key], rtol=0, atol=tol)
        samples.append(
            dict(
                step=step,
                loading_time_s=rve.loading_time,
                normal_Pa=sigma,
                contact_Pa=shear,
                driving_Pa=drive,
                vx_m_s=vx,
                wall_slip_m=measured_slip,
                loading_slip_m=accumulated,
            )
        )
    if drive and rve.loading_slip_origin * np.sign(drive) <= 1e-9:
        raise AssertionError("Expected a signed, measurable preload displacement.")
    with (rve.work_directory / "handoff.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=samples[0])
        writer.writeheader()
        writer.writerows(samples)
    return samples


def check_preload_window(trace, cfg, normal, drive, eta):
    normal_tol = cfg.consolidation_stress_atol + cfg.consolidation_stress_rtol * normal
    shear_tol = cfg.consolidation_stress_atol + cfg.consolidation_stress_rtol * abs(
        drive
    )
    tail = trace[-cfg.consolidation_window_steps :]
    np.testing.assert_equal(tail["stable_steps"], np.arange(1, len(tail) + 1))
    np.testing.assert_equal(tail["applied_drive_Pa"], drive)
    for field in ("top_Pa", "bottom_Pa"):
        np.testing.assert_allclose(tail[field], normal, atol=normal_tol, rtol=0)
    np.testing.assert_allclose(tail["imbalance_Pa"], 0, atol=normal_tol, rtol=0)
    for field in ("top_shear_Pa", "bottom_shear_Pa"):
        np.testing.assert_allclose(
            tail[field] + eta * tail["top_vx_m_s"], drive, atol=shear_tol, rtol=0
        )
    for field in ("wall_speed_m_s", "top_vx_m_s", "grain_speed_m_s", "rim_speed_m_s"):
        np.testing.assert_allclose(
            tail[field], 0, atol=cfg.consolidation_speed_tol, rtol=0
        )
    np.testing.assert_allclose(eta * tail["top_vx_m_s"], 0, atol=shear_tol, rtol=0)


def plot_handoff(axes, trace, samples, target):
    times = (trace["time_s"] - trace["time_s"][-1]) * 1e3
    held = np.array([r["loading_time_s"] for r in samples]) * 1e3
    for axis, field, held_field, factor in zip(
        axes,
        ("top_Pa", "top_shear_Pa", "top_vx_m_s", "preload_slip_m"),
        ("normal_Pa", "contact_Pa", "vx_m_s", "wall_slip_m"),
        (1e-3, 1e-3, 1e3, 1e6),
    ):
        (line,) = axis.plot(
            times, trace[field] * factor, label=f"{target / 1000:g} kPa"
        )
        axis.plot(
            held, [r[held_field] * factor for r in samples], color=line.get_color()
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/shear-preload"))
    out = parser.parse_args().output.resolve()
    width = prepare_small_case(out / "input")
    settings = DEMCouplingParameters(
        case_directory=str(out / "input"), layer_length=width, wall_mass_per_area=50.0
    )
    normal, eta = 10000.0, 50000.0
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(8, 8), constrained_layout=True)
    for name, target in (("zero", 0.0), ("positive", 1000.0), ("negative", -1000.0)):
        rve = prepare(settings, out / name, normal, target, eta)
        try:
            samples = hold(rve, normal, target, eta)
            trace = np.genfromtxt(
                rve.work_directory / "shear_preload.csv",
                delimiter=",",
                names=True,
                dtype=None,
                encoding="utf-8",
            )
            assert trace["status"][-1] == "converged"
            check_preload_window(trace, settings, normal, target, eta)
            plot_handoff(axes, trace, samples, target)
            print(
                f"{name}: preload {len(trace)} steps, first contact change "
                f"{samples[1]['contact_Pa'] - samples[0]['contact_Pa']:.6g} Pa, "
                f"hold slip {rve.loading_slip:.6g} m"
            )
        finally:
            rve.close()
    for axis, label in zip(
        axes,
        ("Normal (kPa)", "Contact shear (kPa)", "Wall speed (mm/s)", "Wall slip (µm)"),
    ):
        axis.axvline(0, color="black", linestyle="--", linewidth=0.8)
        axis.set_ylabel(label)
    axes[1].legend(title="Target drive")
    axes[-1].set_xlabel("Time relative to handoff (ms)")
    fig.savefig(out / "handoff.png", dpi=160)
    plt.close(fig)

    directory = out / "overload"
    try:
        rve = prepare(
            replace(settings, consolidation_max_steps=2500),
            directory,
            normal,
            20000.0,
            eta,
        )
    except ShearPreloadError as error:
        with (directory / "rve_00000" / "shear_preload.csv").open() as stream:
            history = list(csv.DictReader(stream))
        tail = history[-settings.consolidation_window_steps :]
        assert history[-1]["status"] == "sustained_sliding", str(error)
        assert all(
            float(r["applied_drive_Pa"]) == 20000.0
            and float(r["top_vx_m_s"]) > settings.consolidation_speed_tol
            for r in tail
        )
        print(error)
    else:
        rve.close()
        raise AssertionError("Overload unexpectedly reached a static state.")


if __name__ == "__main__":
    main()
