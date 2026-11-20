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
        # Each of the four phases occupies twelve Euler intervals.
        phase = round(t / p.dt_init) // 12
        velocity = 0.0 if phase in (0, 2) else 0.002
        for edge in (p.bc.right, p.bc.top, p.bc.bottom):
            edge.uy.set_velocity(velocity)


def run_case(output: Path, output_interval: int, checkpoint_interval: int):
    width = prepare_small_case(output / "input")
    dt = 2.5e-5
    params = ModelParameters(
        case_type="lab",
        friction_law=FrictionLaw.DEM_MULTISCALE,
        dem=DEMCouplingParameters(
            case_directory=str(output / "input"),
            layer_length=width,
            initial_driving_shear_stress=500.0,
            wall_mass_per_area=50.0,
            max_macro_dt=dt,
        ),
        alpha=90.0,
        xsize=1.0,
        ysize=1.0,
        Nx=5,
        Ny=5,
        Nt=48,
        tfinal=48 * dt,
        dt_init=dt,
        dt_max=dt,
        dt_growth=1.0,
        output_interval=output_interval,
        checkpoint_interval=checkpoint_interval,
        output_vtk_option=False,
        E=1e8,
        sigma0=1e4,
    )
    params.bc.left.set_fixed()
    for edge in (params.bc.right, params.bc.top, params.bc.bottom):
        edge.ux.set_fixed()
        edge.uy.set_velocity(0.0)
    model = DEMExample(params=params, output_dir=str(output))
    model.run()
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
    model = run_case(args.output.resolve(), 1, 24)
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
