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

from fastslippy.pre_processing.model_parameters import ModelParameters
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

        #TODO: will be removed after testing        
        '''
        fig, axes = plt.subplots(1, 4, figsize=(14, 5))
        lw = 2
        # --------------------------------------------------
        # Effective normal stress
        # --------------------------------------------------
        axes[0].plot(self.sigman0 / 1e6, y, linewidth=lw)
        axes[0].invert_yaxis()
        axes[0].set_xlabel(r'$\sigma_n$ (MPa)')
        axes[0].set_ylabel('Depth (m)')
        axes[0].set_title('Effective Normal Stress')
        axes[0].set_xlim(10, 25)
        axes[0].set_ylim(2000, 0)
        axes[0].grid(True)
        # --------------------------------------------------
        # Shear stress
        # --------------------------------------------------
        axes[1].plot(self.tau0 / 1e6, y, linewidth=lw)
        axes[1].invert_yaxis()
        axes[1].set_xlabel(r'$\tau$ (MPa)')
        axes[1].set_title('Shear Stress')
        axes[1].set_ylim(2000, 0)
        axes[1].grid(True)
        # --------------------------------------------------
        # Left pore pressure
        # --------------------------------------------------
        axes[2].plot(self.Pl0 / 1e6, y, linewidth=lw)
        axes[2].invert_yaxis()
        axes[2].set_xlabel(r'$P_l$ (MPa)')
        axes[2].set_title('Left Pore Pressure')
        axes[2].set_ylim(2000, 0)
        axes[2].set_xlim(20, 50)
        axes[2].grid(True)
        # --------------------------------------------------
        # Right pore pressure
        # --------------------------------------------------
        axes[3].plot(self.Pr0 / 1e6, y, linewidth=lw)
        axes[3].invert_yaxis()
        axes[3].set_xlabel(r'$P_r$ (MPa)')
        axes[3].set_title('Right Pore Pressure')
        axes[3].set_ylim(2000, 0)
        axes[3].set_xlim(20, 50)
        axes[3].grid(True)
        # --------------------------------------------------
        plt.suptitle('Initial Stress and Pressure Profiles')
        plt.tight_layout()
        #plt.show()
        '''

    # TODO: This should be replaced by a more flexible function 
    def _initial_stress(self):
        p, y = self.p, self.y
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

        #Lab-test version (for testing only, will be removed)
        sigman0 = 15e6
        tau0 = 0.0
        return sigman0, tau0, Pl0, Pr0

    def update_pressure(self, dt: float, dPdt: float):
        """Advance pore pressures by one time step."""
        y = self.y
        self.Pl += dPdt * dt * (y > 800) * (y <= 1000)
        self.Pr += dPdt * dt * (y > 850) * (y <= 1050)
        self.P   = np.where(y < 1000, self.Pl, self.Pr)