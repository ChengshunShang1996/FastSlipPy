"""Full-cycle DEM/FDM laboratory example with hold, loading and reloading."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from verify_single_rve import prepare_small_case

from fastslippy import FastSlipPy
from fastslippy.pre_processing.model_parameters import (
    DEMCouplingParameters,
    FrictionLaw,
    ModelParameters,
)


class DEMExample(FastSlipPy):
    def set_lab_case_velocity_bc(self, p, t):
        # The imposed loading history is fixed in physical time under refinement.
        phase = round(t / p.dt_init) // round(0.0003 / p.dt_init)
        velocity = 0.0 if phase in (0, 2) else 0.05
        for edge in (p.bc.right, p.bc.top, p.bc.bottom):
            edge.uy.set_velocity(velocity)


def plot_dem_diagnostics(output: Path) -> None:
    """Compare wall, bulk and controller measurements for the central RVE."""
    import matplotlib.pyplot as plt

    path = output / "dem_history.csv"
    history = np.genfromtxt(path, delimiter=",", names=True)
    required = {
        "internal_slip_m",
        "bulk_shear_Pa",
        "bulk_normal_Pa",
        "target_normal_Pa",
        "target_driving_shear_Pa",
        "max_contact_friction_ratio",
    }
    if not required.issubset(history.dtype.names or ()):
        return
    node_id = int(np.median(np.unique(history["node"])))
    node = history[history["node"] == node_id]
    time_ms = 1.0e3 * node["fdm_time_s"]

    fig, axes = plt.subplots(
        3, 2, figsize=(11, 9), sharex=True, constrained_layout=True
    )
    ax = axes[0, 0]
    for field, label in (
        ("target_normal_Pa", "target"),
        ("top_normal_Pa", "top wall"),
        ("bottom_normal_Pa", "bottom wall"),
        ("bulk_normal_Pa", "bulk"),
    ):
        ax.plot(time_ms, 1.0e-3 * node[field], label=label)
    ax.set_ylabel("Normal stress (kPa)")
    ax.legend()

    ax = axes[0, 1]
    for field, label in (
        ("target_driving_shear_Pa", "FDM drive"),
        ("top_shear_Pa", "top wall"),
        ("bottom_shear_Pa", "bottom wall"),
        ("bulk_shear_Pa", "bulk"),
    ):
        ax.plot(time_ms, 1.0e-3 * node[field], label=label)
    ax.set_ylabel("Shear stress (kPa)")
    ax.legend()

    axes[1, 0].plot(time_ms, 1.0e3 * node["dem_slip_m"], label="top wall")
    axes[1, 0].plot(
        time_ms, 1.0e3 * node["internal_slip_m"], label="upper-lower grains"
    )
    axes[1, 0].set_ylabel("Accumulated slip (mm)")
    axes[1, 0].legend()

    axes[1, 1].plot(
        time_ms, node["end_velocity_m_s"], label="top-wall velocity"
    )
    axes[1, 1].plot(
        time_ms, node["internal_velocity_m_s"], label="internal velocity"
    )
    axes[1, 1].set_ylabel("Velocity (m/s)")
    axes[1, 1].legend()

    axes[2, 0].plot(
        time_ms,
        node["max_contact_friction_ratio"],
        label=r"max $|F_t|/|F_n|$",
    )
    axes[2, 0].plot(
        time_ms, node["contact_friction_limit"], "--", label="Coulomb limit"
    )
    axes[2, 0].plot(
        time_ms, node["sliding_contact_fraction"], label="sliding fraction"
    )
    axes[2, 0].set_ylabel("Contact friction diagnostic")
    axes[2, 0].legend()

    axes[2, 1].plot(
        time_ms, node["normal_wall_velocity_m_s"], label="normal wall velocity"
    )
    saturated = node["normal_servo_saturated"] > 0.5
    axes[2, 1].scatter(
        time_ms[saturated],
        node["normal_wall_velocity_m_s"][saturated],
        s=8,
        color="red",
        label="servo saturated",
    )
    axes[2, 1].set_ylabel("Normal velocity (m/s)")
    axes[2, 1].legend()
    for ax in axes[-1]:
        ax.set_xlabel("FDM time (ms)")
    fig.suptitle(f"DEM coupling diagnostics, node {node_id}")
    fig.savefig(output / "dem_diagnostics.png", dpi=180)
    plt.close(fig)


def run_case(
    output: Path,
    output_interval: int,
    checkpoint_interval: int,
    *,
    dt=2.5e-5,
    factory_class=None,
):
    if not np.isfinite(dt) or dt <= 0 or not np.isclose(
        round(0.0003 / dt) * dt, 0.0003, rtol=1e-12, atol=0
    ):
        raise ValueError("The macro step must divide the 0.3 ms loading phase.")
    width = prepare_small_case(output / "input")
    my_sim_time = 0.1
    params = ModelParameters(
        case_type="lab",
        friction_law=FrictionLaw.DEM_MULTISCALE,
        dem=DEMCouplingParameters(
            case_directory=str(output / "input"),
            layer_length=width,
            initial_driving_shear_stress=None,
            wall_mass_per_area=50.0,
            max_macro_dt=dt,
            output_enabled = True,
            consolidation_max_steps = 60000,
            normal_servo_velocity = 0.05,
        ),
        alpha=90.0,
        xsize=1.0,
        ysize=1.0,
        Nx=5,
        Ny=5,
        Nt=round(my_sim_time / dt),
        tfinal=my_sim_time,
        dt_init=dt,
        dt_max=dt,
        dt_growth=1.0,
        output_interval=output_interval,
        checkpoint_interval=checkpoint_interval,
        output_vtk_option=False,
        E=1e8,
        sigma0=1e5,
    )
    params.bc.left.set_fixed()
    for edge in (params.bc.right, params.bc.top, params.bc.bottom):
        edge.ux.set_fixed()
        edge.uy.set_velocity(0.0)
    factory = (
        None
        if factory_class is None
        else factory_class(params.dem, output / "dem_rves")
    )
    model = DEMExample(
        params=params, output_dir=str(output), dem_backend_factory=factory
    )
    model.run()
    plot_dem_diagnostics(output)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/dem-coupled"))
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Repeat with sparse output and compare the accepted trajectories.",
    )
    args = parser.parse_args()
    model = run_case(args.output.resolve(), 10, 100)
    if args.verify:
        repeat = args.output.resolve() / "sparse-output"
        sparse = run_case(repeat, 8, 16)
        history = np.genfromtxt(
            args.output / "dem_history.csv", delimiter=",", names=True
        )
        other = np.genfromtxt(repeat / "dem_history.csv", delimiter=",", names=True)
        assert len(history) == model.p.Ny * (model.p.Nt + 1)
        for i in range(model.p.Ny):
            node = history[history["node"] == i]
            np.testing.assert_allclose(
                node["fdm_time_s"], node["dem_time_s"], rtol=0, atol=1e-14
            )
            np.testing.assert_allclose(
                node["fdm_slip_m"], node["dem_slip_m"], rtol=0, atol=1e-16
            )
            np.testing.assert_equal(np.diff(node["dem_steps"]), 5)
            np.testing.assert_equal(node["fdm_time_s"][-1], model.output.tm[-1])
            selected = other[other["node"] == i]
            for field in history.dtype.names:
                np.testing.assert_array_equal(node[field][::8], selected[field])
        for field in ("tauqs", "sigmaqs", "ux", "uy"):
            np.testing.assert_array_equal(getattr(model, field), getattr(sparse, field))
        print(
            "Verified RVE clocks, slips, quiet-phase substeps "
            "and output-frequency independence."
        )


if __name__ == "__main__":
    main()
