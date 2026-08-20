#/////////////////////////////////////////////////
__author__      = "Chengshun Shang (Utrecht University)"
__copyright__   = "Copyright (C) 2026-present by Chengshun Shang"
__version__     = "0.0.1"
__maintainer__  = "Chengshun Shang"
__email__       = "c.shang@uu.nl"
__status__      = "development"
__date__        = "June 28, 2026"
__license__     = "MIT License"
#/////////////////////////////////////////////////

import numpy as np

from fastslippy import FastSlipPy
from fastslippy.pre_processing.model_parameters import ModelParameters
from fastslippy.pre_processing.grid import Grid

class RunFastSlipPy(FastSlipPy):
    """
    This can be customized for specific runs.
    """
    def run(self):
        super().run()

        self.grid.plot_grid()
        self.grid.plot_mesh()

if __name__ == "__main__":
    # Customise parameters here or leave all defaults

    params = ModelParameters(
        case_type = "california",
        alpha = 60.0,
        xsize = 80e3,
        ysize = 80e3,
        Nx = 361, Ny = 361,
        #Nx = 641, Ny = 161, #500 m 
        #Nx = 141, Ny = 31,
        Nt = 100000,
        output_interval = 10,
        checkpoint_interval = 1000,
        rho = 2670.0,
        cs = 3464,
        mu0 = 0.6,
        nu = 0.25,
        #E=0.55e10, #according to k_critical = sigam * (b-a) / d_c, E = 1e10
        V0 = 1e-6,
        a0 = 0.01,
        a_max = 0.025,
        b0 = 0.015,
        L = 0.008,
        dt_init = 1.0,
        dt_max = 1e6,
        output_vtk_option = False,
        Vi = 1e-9,
        flash_heating_option = False,
        H = 15e3,
        h = 3e3,
        W_f = 40e3,
        x_stretch_enabled=True,
        y_stretch_enabled=True,
        x_stretch_inner_size=20e3,
        y_stretch_inner_size=20e3,
        x_stretch_inner_points=201,
        y_stretch_inner_points=201,
        x_stretch_power=2,
        y_stretch_power=2,
        allow_nonuniform_solver=True,
    )

    yr = 365 * 24 * 3600.0
    params.loading.tload = 0.0 * yr
    params.loading.dPdt_pre = 0.0
    params.loading.dPdt_post = 0.0
    params.loading.V_p = 1e-9
    params.loading.V_L = 1e-9
    
    #velocity_x = params.loading.V_p * 0.5 * np.sin(params.alpha * 3.1415926 / 180.0)
    #velocity_y = params.loading.V_p * 0.5 * np.cos(params.alpha * 3.1415926 / 180.0)
    #velocity_x = params.loading.V_p * 0.5
    velocity_y = params.loading.V_p * 0.5

    params.bc.left.ux.set_fixed()
    params.bc.left.uy.set_velocity(-1 * velocity_y)
    params.bc.right.ux.set_fixed()
    params.bc.right.uy.set_velocity(velocity_y)
    params.bc.top.set_traction_free()
    params.bc.bottom.ux.set_fixed()
    params.bc.bottom.uy.set_velocity(velocity_y)

    top_of_layer = params.ysize
    bottom_of_layer = params.ysize * 2
    params.layers.set_homogeneous(top = top_of_layer, bottom = bottom_of_layer, a=params.a0, b=params.b0)

    model = RunFastSlipPy(params=params, output_dir="output")
    model.run()
