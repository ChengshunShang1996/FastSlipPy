#/////////////////////////////////////////////////
__author__      = "Chengshun Shang (Utrecht University)"
__copyright__   = "Copyright (C) 2026-present by Chengshun Shang"
__version__     = "0.0.1"
__maintainer__  = "Chengshun Shang"
__email__       = "c.shang@uu.nl"
__status__      = "development"
__date__        = "Aug 6, 2026"
__license__     = "MIT License"
#/////////////////////////////////////////////////

from fastslippy import FastSlipPy
from fastslippy.pre_processing.model_parameters import ModelParameters


class RunFastSlipPy(FastSlipPy):
    """
    Example run with stretched mesh in x direction (fault-normal) only.
    Keep y uniform for direct visual comparison against uniform-mesh runs.
    """

    def run(self):
        super().run()


if __name__ == "__main__":
    params = ModelParameters(
        case_type="lab",
        alpha=90.0,
        xsize=1.0,
        ysize=1.0,
        Nx=51,
        Ny=41,
        Nt=10000,
        output_interval=50,
        checkpoint_interval=10000,
        dt_init=1e-4,
        dt_max=0.01,
        Vi=1e-40,
        mu0=0.72,
        nu=0.25,
        E=0.55e10,
        V0=1e-6,
        a0=0.012,
        b0=0.0135,
        flash_heating_option=False,
        x_stretch_enabled=True,
        y_stretch_enabled=False,
        x_stretch_inner_size=0.5,
        y_stretch_inner_size=0.5,
        x_stretch_inner_points=31,
        y_stretch_inner_points=31,
        x_stretch_power=2,
        y_stretch_power=2,
        x_stretch_max_cell_size=0.05,
        y_stretch_max_cell_size=0.05,
        allow_nonuniform_solver=True,
    )

    params.bc.left.ux.set_fixed()
    params.bc.left.uy.set_fixed()
    params.bc.right.ux.set_fixed()
    params.bc.right.uy.set_velocity(1e-5)
    params.bc.top.ux.set_fixed()
    params.bc.top.uy.set_velocity(1e-5)
    params.bc.bottom.ux.set_fixed()
    params.bc.bottom.uy.set_velocity(1e-5)

    params.layers.set_homogeneous(top=1, bottom=2, a=params.a0, b=params.b0)

    model = RunFastSlipPy(params=params, output_dir="output")
    model.run()
