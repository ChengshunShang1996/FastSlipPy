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

def test_radiation_damping():

    """
    Benchmark 7:
    radiation damping decay test

    Solve:

        eta * V + k * U = 0

        dU/dt = V

    Exact solution:

        U(t) = U0 * exp(-k/eta * t)

    Expected:
        exponential decay
        monotonic energy dissipation
    """

    # --------------------------------------------------
    # 1. Parameters
    # --------------------------------------------------

    params = ModelParameters()

    eta = params.eta

    k = 1e6

    U0 = 1e-3

    dt = 0.01

    Nt = 2000

    # --------------------------------------------------
    # 2. Arrays
    # --------------------------------------------------

    U = np.zeros(Nt)

    V = np.zeros(Nt)

    t = np.arange(Nt) * dt

    # initial condition
    U[0] = U0

    # --------------------------------------------------
    # 3. Time integration
    # --------------------------------------------------

    for n in range(Nt - 1):

        # damping relation
        V[n] = -k / eta * U[n]

        # forward Euler
        U[n+1] = U[n] + dt * V[n]

    # final velocity
    V[-1] = -k / eta * U[-1]

    # --------------------------------------------------
    # 4. Exact solution
    # --------------------------------------------------

    U_exact = U0 * np.exp(-k / eta * t)

    # --------------------------------------------------
    # 5. Error diagnostics
    # --------------------------------------------------

    max_error = np.max(np.abs(U - U_exact))
    monotonic = np.all(np.diff(U) <= 0)
    tol = 1e-6

    assert max_error < tol
    assert monotonic