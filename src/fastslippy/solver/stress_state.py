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

import matplotlib.pyplot as plt
import numpy as np

from fastslippy.pre_processing.model_parameters import  ModelParameters
from fastslippy.utilities.math_util import MathUtil

class StressState:
    """
    Computes and stores the initial lithostatic/hydrostatic stress state
    and tracks evolving pore pressures on both sides of the fault.
    """

    def __init__(self, p: ModelParameters, y: np.ndarray):
        self.p = p
        self.y = y
        self.sigman0, self.tau0, self.Pl0, self.Pr0 = self._initial_stress()
        self.Pl = self.Pl0.copy()
        self.Pr = self.Pr0.copy()
        self.P  = np.where(y < 1000, self.Pl, self.Pr)

    # TODO: This should be replaced by a more flexible function 
    def _initial_stress(self):
        p, y = self.p, self.y

        if p.case_type == "groningen":

            sigmatop = p.rho * p.g * 2000 - 4.70e6
            Ptop     = p.rhof * p.g * 2000
            sigmav = p.rho * p.g * y + sigmatop
            Pl0 = (p.rhof * p.g * y
                + (p.rhof - p.rhog) * p.g * (1000 - y) * (y > 800) * (y <= 1000)
                + Ptop + 1.16e6 * (y > 800))
            Pr0 = (p.rhof * p.g * y
                + (p.rhof - p.rhog) * p.g * (1000 - y) * (y > 850) * (y <= 1050)
                + Ptop + 1.16e6 * (y > 850))
            sigman0 = ((1 + p.K0) / 2 * sigmav
                    + (1 - p.K0) / 2 * MathUtil.cosd(2 * p.alpha) * sigmav
                    - np.where(y < 1000, Pl0, Pr0))
            tau0 = (1 - p.K0) / 2 * MathUtil.sind(2 * p.alpha) * sigmav

        elif p.case_type == "lab":

            sigman0 = 0.0 * y + 15e6
            tau0 = 0.0 * y  + 1e-30
            Pl0 = 0.0 * y + 1e-30
            Pr0 = 0.0 * y + 1e-30

        elif p.case_type == "california":

            sigman0 = 0.0 * y + 50e6
            tau0 = sigman0 * p.a_max * np.arcsinh(p.Vi / (2 * p.V0) * np.exp((p.mu0 + p.b0 * np.log(p.V0 / np.abs(p.Vi))) / p.a_max))
            Pl0 = 0.0 * y + 1e-30
            Pr0 = 0.0 * y + 1e-30

        return sigman0, tau0, Pl0, Pr0

    def update_pressure(self, dt: float, dPdt: float):
        """Advance pore pressures by one time step."""
        y = self.y
        self.Pl += dPdt * dt * (y > 800) * (y <= 1000)
        self.Pr += dPdt * dt * (y > 850) * (y <= 1050)
        self.P   = np.where(y < 1000, self.Pl, self.Pr)