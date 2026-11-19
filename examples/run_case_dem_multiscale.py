"""Run the optional fast-slip Kratos DEM coupling.

The referenced DEM case must contain ``dem_shearDEM.mdpa``,
``dem_shearDEM_FEM_boundary.mdpa``, ``MaterialsDEM.json`` and
``ProjectParametersDEM.json``.  Stress, length and time units in that case
must be consistent with FastSlipPy before using this as a physical model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fastslippy import FastSlipPy
from fastslippy.pre_processing.model_parameters import (
    DEMCouplingParameters,
    FrictionLaw,
    ModelParameters,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dem_case_directory",
        type=Path,
        help="directory containing the prepared Kratos DEM shear case",
    )
    parser.add_argument("--output", type=Path, default=Path("output/dem-coupled"))
    args = parser.parse_args()

    dem = DEMCouplingParameters(
        case_directory=str(args.dem_case_directory.resolve()),
        activation_velocity=5.0e-5,
        deactivation_velocity=2.0e-5,
        deactivation_steps=10,
        max_macro_dt=1.0e-4,
        max_slip_increment=1.0e-6,
        layer_length=0.12,
        wall_mass_per_area=100.0,
        normal_servo_velocity=0.1,
        servo_relaxation=0.15,
        maximum_wall_speed=0.5,
        consolidation_time=1.0e-3,
        output_enabled=False,
    )
    params = ModelParameters(
        case_type="lab",
        friction_law=FrictionLaw.DEM_MULTISCALE,
        dem=dem,
        alpha=90.0,
        xsize=1.0,
        ysize=1.0,
        Nx=11,
        Ny=11,
        Nt=1000,
        tfinal=0.1,
        dt_init=1.0e-5,
        dt_max=1.0e-3,
        output_interval=10,
        checkpoint_interval=100,
        output_vtk_option=False,
        E=1.0e8,
        sigma0=1.0e4,
        Vi=1.0e-40,
    )
    params.bc.left.set_fixed()
    params.bc.right.ux.set_fixed()
    params.bc.right.uy.set_velocity(1.0e-4)
    params.bc.top.ux.set_fixed()
    params.bc.top.uy.set_velocity(1.0e-4)
    params.bc.bottom.ux.set_fixed()
    params.bc.bottom.uy.set_velocity(1.0e-4)

    FastSlipPy(params=params, output_dir=str(args.output)).run()


if __name__ == "__main__":
    main()

