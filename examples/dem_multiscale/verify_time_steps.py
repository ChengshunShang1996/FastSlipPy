"""One-factor DEM and Euler time-step refinement of the full-cycle lab case."""

import argparse
import csv
from dataclasses import replace
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from run_case_dem_multiscale import run_case

from fastslippy.solver.kratos_dem_backend import KratosDEMRVE

# The finest DEM run doubles as the coarsest FDM run. All ratios are integral.
CASES = [(25, 5), (25, 2.5), (25, 1.25), (12.5, 1.25), (6.25, 1.25)]
FIELDS = ("slip_m", "velocity_m_s", "shear_Pa", "normal_Pa", "height_m")


class MeasuredRVE(KratosDEMRVE):
    def __init__(self, *args, dem_dt, **kwargs):
        self.samples = []
        super().__init__(*args, **kwargs)
        # Every run prepares exactly the same state at the original step.
        solver = self.stage._GetSolver()
        # Preserve the normal servo's relaxation time when its sampling rate changes.
        r = self.settings.servo_relaxation
        self.stage.settings = replace(
            self.settings, servo_relaxation=1 - (1 - r) ** (dem_dt / solver.dt)
        )
        solver.dt = dem_dt
        self.residual = 0.0
        self.record()

    def _single_step(self):
        if not self.samples:  # Normal and shear preload
            return super()._single_step()
        stage = self.stage
        dt = stage._GetSolver().dt
        old_v = stage.wall_velocity[0]
        force = stage.driving_shear_stress - stage.measured_shear_traction
        force -= stage.radiation_damping * old_v
        super()._single_step()
        dv = stage.wall_velocity[0] - old_v
        self.residual = max(
            self.residual, abs(force - self.settings.wall_mass_per_area * dv / dt)
        )

    def record(self):
        state = self.wall_state
        height = np.mean([n.Y for n in self.stage.top_wall.Nodes]) - np.mean(
            [n.Y for n in self.stage.bottom_wall.Nodes]
        )
        self.samples.append(
            (
                state.loading_time,
                state.loading_slip,
                state.velocity,
                state.shear_traction,
                state.normal_traction,
                height,
                self.residual,
            )
        )

    def advance(self, **kwargs):
        self.residual = 0.0
        result = super().advance(**kwargs)
        self.record()
        return result

    def close(self):
        if self.samples:
            np.savetxt(
                self.work_directory / "response.csv",
                self.samples,
                delimiter=",",
                header="time_s," + ",".join(FIELDS) + ",momentum_residual_Pa",
                comments="",
            )
        super().close()


def factory(dem_dt, settings, output):
    return partial(MeasuredRVE, settings=settings, output_root=output, dem_dt=dem_dt)


def compare(out):
    traces = []
    for macro, micro in CASES:
        directory = out / f"fdm{macro}_dem{micro}"
        data = np.stack(
            [
                np.loadtxt(path, delimiter=",", skiprows=1)
                for path in sorted(directory.glob("dem_rves/rve_*/response.csv"))
            ]
        )
        assert data.shape == (5, 1 + round(1200 / macro), 7)
        assert np.isfinite(data).all()
        history = np.genfromtxt(
            directory / "dem_history.csv", names=True, delimiter=","
        )
        for i, node in enumerate(data):
            accepted = history[history["node"] == i]
            np.testing.assert_equal(
                np.diff(accepted["dem_steps"]), round(macro / micro)
            )
            np.testing.assert_allclose(
                accepted["fdm_slip_m"], node[:, 1], rtol=0, atol=1e-16
            )
        np.testing.assert_allclose(
            data[:, :, 0],
            np.tile(np.arange(data.shape[1]) * macro * 1e-6, (5, 1)),
            rtol=0,
            atol=1e-13,
        )
        np.testing.assert_allclose(data[:, :, -1], 0, atol=1e-8)
        if traces:
            np.testing.assert_array_equal(data[:, 0], traces[0][:, 0])
        traces.append(data)

    with (out / "sensitivity.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        coarse_rms = {}
        writer.writerow(
            (
                "refinement",
                "coarse_step_us",
                "fine_step_us",
                "field",
                "max_abs_difference",
                "rms_difference",
                "relative_rms",
            )
        )
        for i in range(4):
            is_dem = i < 2
            # Use the original 25 µs sampling grid for every error norm.
            a = traces[i][:, :: round(25 / CASES[i][0])]
            b = traces[i + 1][:, :: round(25 / CASES[i + 1][0])]
            np.testing.assert_allclose(a[:, :, 0], b[:, :, 0], rtol=0, atol=1e-13)
            for j, field in enumerate(FIELDS, 1):
                delta = a[:, :, j] - b[:, :, j]
                rms = np.sqrt(np.mean(delta**2))
                if i in (0, 2):
                    coarse_rms[field] = rms
                elif rms >= coarse_rms[field]:
                    raise AssertionError(
                        f"Refinement did not reduce RMS error for {field}: "
                        f"{coarse_rms[field]:g} -> {rms:g}"
                    )
                scale = np.sqrt(np.mean((b[:, :, j] - b[:, 0:1, j]) ** 2))
                writer.writerow(
                    (
                        "DEM" if is_dem else "FDM",
                        CASES[i][int(is_dem)],
                        CASES[i + 1][int(is_dem)],
                        field,
                        abs(delta).max(),
                        rms,
                        rms / scale if scale else np.nan,
                    )
                )

    with (out / "phases.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "macro_us",
                "micro_us",
                "phase",
                "max_abs_net_slip_m",
                "peak_abs_speed_m_s",
                "max_abs_shear_change_Pa",
                "max_momentum_residual_Pa",
            )
        )
        for (macro, micro), data in zip(CASES, traces):
            n = (data.shape[1] - 1) // 4
            for i, phase in enumerate(("initial_hold", "load", "quiet_hold", "reload")):
                part = data[:, i * n : (i + 1) * n + 1]
                writer.writerow(
                    (
                        macro,
                        micro,
                        phase,
                        abs(part[:, -1, 1] - part[:, 0, 1]).max(),
                        abs(part[:, :, 2]).max(),
                        abs(part[:, :, 3] - part[:, 0:1, 3]).max(),
                        abs(part[:, 1:, -1]).max(),
                    )
                )

    fig, axes = plt.subplots(
        5, 1, sharex=True, figsize=(8, 10), constrained_layout=True
    )
    labels = (
        "Slip (nm)",
        "Slip rate (µm/s)",
        "Shear change (Pa)",
        "Normal change (Pa)",
        "Wall separation change (nm)",
    )
    for (macro, micro), data in zip(CASES, traces):
        node = data[2]
        for j, (ax, factor) in enumerate(zip(axes, (1e9, 1e6, 1, 1, 1e9)), 1):
            origin = node[0, j] if j >= 3 else 0
            ax.plot(
                node[:, 0] * 1e3,
                (node[:, j] - origin) * factor,
                label=f"{macro:g} / {micro:g}",
            )
            ax.set_ylabel(labels[j - 1])
    for ax in axes:
        for t in (0.3, 0.6, 0.9):
            ax.axvline(t, color="gray", linewidth=0.5)
    axes[0].legend(title="FDM / DEM (µs)", ncol=3)
    axes[-1].set_xlabel("Loading time (ms), central fault node")
    fig.savefig(out / "responses.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("output/time-step-baseline")
    )
    out = parser.parse_args().output.resolve()
    for macro, micro in CASES:
        run_case(
            out / f"fdm{macro}_dem{micro}",
            1,
            round(1200 / macro),
            dt=macro * 1e-6,
            factory_class=partial(factory, micro * 1e-6),
        )
        plt.close("all")
    compare(out)
