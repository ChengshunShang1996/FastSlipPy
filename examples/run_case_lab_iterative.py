#/////////////////////////////////////////////////
__author__      = "Chengshun Shang (Utrecht University)"
__copyright__   = "Copyright (C) 2026-present by Chengshun Shang"
__version__     = "0.0.1"
__maintainer__  = "Chengshun Shang"
__email__       = "c.shang@uu.nl"
__status__      = "development"
__date__        = "Aug 9, 2026"
__license__     = "MIT License"
#/////////////////////////////////////////////////

from fastslippy import FastSlipPy
from fastslippy.pre_processing.model_parameters import ModelParameters


class RunFastSlipPy(FastSlipPy):
    """
    This can be customized for specific runs.
    """

    def run(self):
        super().run()


if __name__ == "__main__":
    # Lab benchmark example using iterative linear solver to reduce memory use.
    params = ModelParameters(
        case_type="lab",
        alpha=90.0,
        xsize=1.0,
        ysize=1.0,
        Nx=41,
        Ny=41,
        Nt=10000,
        output_interval=10,
        checkpoint_interval=10000,
        Vi=1e-40,
        dt_init=0.0001,
        dt_max=0.01,
        mu0=0.72,
        nu=0.25,
        E=0.55e10,
        V0=1e-6,
        a0=0.012,
        b0=0.0135,
        flash_heating_option=False,
        linear_solver="iterative",
        iterative_method="gmres",
        iterative_rtol=1e-6,
        iterative_atol=0.0,
        iterative_maxiter=200,
        ilu_drop_tol=1e-2,
        ilu_fill_factor=8.0,
    )

    params.bc.left.ux.set_fixed()
    params.bc.left.uy.set_fixed()
    params.bc.right.ux.set_fixed()
    params.bc.right.uy.set_velocity(1e-5)
    params.bc.bottom.ux.set_fixed()
    params.bc.bottom.uy.set_velocity(1e-5)  # only acts on right half of bottom boundary
    params.bc.top.ux.set_fixed()
    params.bc.top.uy.set_velocity(1e-5)  # only acts on right half of top boundary

    params.layers.set_homogeneous(top=1, bottom=2, a=params.a0, b=params.b0)

    model = RunFastSlipPy(params=params, output_dir="output")
    model.run()
