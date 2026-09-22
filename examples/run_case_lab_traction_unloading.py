#/////////////////////////////////////////////////
__author__      = "Chengshun Shang (Utrecht University)"
__copyright__   = "Copyright (C) 2026-present by Chengshun Shang"
__version__     = "0.1.3"
__maintainer__  = "Chengshun Shang"
__email__       = "c.shang@uu.nl"
__status__      = "development"
__date__        = "September 22, 2026"
__license__     = "MIT License"
#/////////////////////////////////////////////////

import numpy as np

from fastslippy import FastSlipPy
from fastslippy.pre_processing.model_parameters import ModelParameters


NORMAL_STRESS_DROP = 5.0e6       # Pa; lab background is 15 MPa compression
UNLOADING_DURATION = 40.0         # s


class RunFastSlipPy(FastSlipPy):
    """Lab case with shear loading and a decreasing fault-normal stress."""

    def set_lab_case_velocity_bc(self, p: ModelParameters, t: float):
        # Retain the shear-loading history from run_case_lab.py.
        shear_velocity = 1.0e-4
        # if t <= 4.0:
        #     shear_velocity = 1.0e-5
        # elif t <= 6.0:
        #     shear_velocity = 1.0e-4
        # else:
        #     shear_velocity = 1.0e-5
        p.bc.right.uy.value = shear_velocity
        p.bc.top.uy.value = shear_velocity
        p.bc.bottom.uy.value = shear_velocity

        # Smoothly reduce compression by NORMAL_STRESS_DROP:
        #   Delta sigma(t) = drop/2 * (1 - cos(pi*t/T)).
        # A positive traction on the right boundary is tensile, so the
        # resulting positive sigma_xx perturbation reduces the effective
        # fault-normal compression: sigma_eff = sigma_initial - sigma_xx.
        if 0.0 < t < UNLOADING_DURATION:
            traction_rate = (
                0.5
                * NORMAL_STRESS_DROP
                * np.pi
                / UNLOADING_DURATION
                * np.sin(np.pi * t / UNLOADING_DURATION)
            )
        else:
            traction_rate = 0.0
        p.bc.right.ux.set_traction(traction_rate)


def build_parameters() -> ModelParameters:
    params = ModelParameters(
        case_type="lab",
        alpha=90.0,
        xsize=1.0,
        ysize=1.0,
        Nx=41,
        Ny=41,
        Nt=100000,
        tfinal=50.0,
        output_interval=10,
        checkpoint_interval=1000,
        vtk_interval=10000,
        Vi=1.0e-40,
        dt_init=1.0e-4,
        dt_max=0.01,
        mu0=0.72,
        nu=0.25,
        E=0.55e10,#1.0e12,
        V0=1.0e-6,
        a0=0.012,
        b0=0.0135,
        flash_heating_option=False,
    )

    # Fix the left side as the horizontal reference and apply a tensile
    # traction rate on the right to unload the initial 15 MPa compression.
    params.bc.left.ux.set_fixed()
    params.bc.right.ux.set_traction(0.0)
    params.bc.top.ux.set_traction_free()
    params.bc.bottom.ux.set_traction_free()

    params.bc.left.uy.set_fixed()
    params.bc.right.uy.set_velocity(1.0e-4)
    params.bc.top.uy.set_velocity(1.0e-4)
    params.bc.bottom.uy.set_velocity(1.0e-4)

    params.layers.set_homogeneous(
        top=1,
        bottom=2,
        a=params.a0,
        b=params.b0,
    )
    return params


if __name__ == "__main__":
    model = RunFastSlipPy(
        params=build_parameters(),
        output_dir="output/lab_traction_unloading",
    )
    model.run()
