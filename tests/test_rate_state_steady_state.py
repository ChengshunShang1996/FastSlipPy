#/////////////////////////////////////////////////
__author__      = "Chengshun Shang (Utrecht University)"
__copyright__   = "Copyright (C) 2026-present by Chengshun Shang"
__version__     = "0.0.1"
__maintainer__  = "Chengshun Shang"
__email__       = "c.shang@uu.nl"
__status__      = "development"
__date__        = "May 22, 2026"
__license__     = "MIT License"
#/////////////////////////////////////////////////

import numpy as np

from fastslippy.pre_processing.model_parameters import ModelParameters

def test_rate_state_steady_state():

    """
    Benchmark 8:
    rate-and-state steady state test

    Verify:

        theta_ss = L / V

    and:

        f_ss = f0 + (a-b)*ln(V/V0)
    """

    # --------------------------------------------------
    # 1. Parameters
    # --------------------------------------------------

    a = 0.01
    b = 0.015

    f0 = 0.6

    V0 = 1e-6

    L = 1e-5

    sigma_n = 50e6

    # test velocities
    V_values = np.logspace(-9, -3, 50)

    # --------------------------------------------------
    # 2. Numerical steady-state friction
    # --------------------------------------------------

    f_numerical = np.zeros_like(V_values)

    theta_ss = L / V_values

    for i, V in enumerate(V_values):

        f_numerical[i] = (
            f0
            + a * np.log(V / V0)
            + b * np.log(V0 * theta_ss[i] / L)
        )

    # --------------------------------------------------
    # 3. Exact steady-state solution
    # --------------------------------------------------

    f_exact = f0 + (a - b) * np.log(V_values / V0)

    # --------------------------------------------------
    # 4. Error
    # --------------------------------------------------

    error = np.max(np.abs(f_numerical - f_exact))
    
    assert error < 1e-12