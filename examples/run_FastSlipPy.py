#/////////////////////////////////////////////////
__author__      = "Chengshun Shang (Utrecht University)"
__copyright__   = "Copyright (C) 2026-present by Chengshun Shang"
__version__     = "0.0.1"
__maintainer__  = "Chengshun Shang"
__email__       = "c.shang@uu.nl"
__status__      = "development"
__date__        = "May 5, 2026"
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
    # Customise parameters here or leave all defaults

    params = ModelParameters(
        alpha = 90.0,
        xsize = 1.0,
        ysize = 1.0,
        Nx=21, Ny=21,
        Nt=15000,
        output_interval=1,
        checkpoint_interval=5000,
        Vi=1e-40,
        dt_init=0.0001,
        dt_max = 0.01,
        mu0=0.72,
        nu=0.25,
        E=0.55e10, #according to k_critical = sigam * (b-a) / d_c, E = 1e10
        V0 = 1e-6,
        a0 = 0.012,
        b0 = 0.0135
    )

    #params = ModelParameters()
    model = RunFastSlipPy(params=params, output_dir="output")
    model.run()
