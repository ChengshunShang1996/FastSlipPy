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

import numpy as np
from scipy.optimize import brentq

from fastslippy.pre_processing.model_parameters import ModelParameters
from fastslippy.pre_processing.frictional_zones import FrictionalZones
from fastslippy.solver.stress_state import StressState
from fastslippy.utilities.math_util import MathUtil

class FaultState:
    """
    Holds and evolves the on-fault variables:
      U      – cumulative slip [m]
      V      – slip rate [m/s]
      theta  – state variable [s]
      sigma  – effective normal stress [Pa]
      tau    – shear stress [Pa]
    """

    def __init__(self, p: ModelParameters, stress: StressState,
                 fric: FrictionalZones):
        self.p = p
        Ny = p.Ny
        self.U     = np.zeros(Ny)
        self.V     = np.full(Ny, p.Vi)
        logarg = 2 * p.V0 / p.Vi * np.sinh((stress.tau0 - p.eta * p.Vi) / fric.a / stress.sigman0)
        self.theta = (p.L / p.V0 * np.exp(fric.a / fric.b * np.emath.log(logarg)- p.mu0 / fric.b))
        self.sigma = stress.sigman0.copy()
        self.tau   = stress.tau0 - p.eta * self.V

    def solve_slip_rate_1(self, tauqs_col: np.ndarray, stress: StressState,
                        fric: FrictionalZones):
        """
        Solve for V at each fault node using the rate-and-state friction law
        (with flash heating):

            σ · a · asinh[ V/(2V₀) · exp((μ₀ + b·ln(V₀θ/L))/a) ]
                / (1 + L/(Vw·θ))   +   η·V   =   τ_qs + τ₀
        """
        p = self.p
        for iy in range(p.Ny):
            rhs = tauqs_col[iy] + stress.tau0[iy]
            a_i = fric.a[iy];  b_i = fric.b[iy]
            th  = self.theta[iy]
            sig = self.sigma[iy]

            def equation(VV):
                arg = (p.mu0 + b_i * np.log(p.V0 * th / p.L)) / a_i
                friction = sig * a_i * np.arcsinh(VV / (2 * p.V0) * np.exp(arg))
                flash    = 1 + p.L / p.Vw / th
                return friction / flash + p.eta * VV - rhs

            # Guard against sign errors in the bracket
            try:
                self.V[iy] = brentq(equation, 1e-40, 1e10, xtol=1e-12)
            except ValueError:
                pass   # keep previous V if bracketing fails

        self.V = np.maximum(self.V, 1e-40)
        self.V[0]  = self.V[1]
        self.V[-1] = self.V[-2]
    
    # ------------------------------------------------------------------
    def advance(self, dt: float, tauqs_col: np.ndarray, stress: StressState):
        """Update theta, U, tau after the velocity solve."""
        p = self.p
        
        self.theta = self.theta + dt * (1 - self.V * self.theta / p.L)
        # TODO: this one is better
        # x = self.V * dt / p.L
        # expo = x > 1e-6
        # theta_new = np.empty_like(self.theta)
        # theta_new[expo] = (
        #     p.L / self.V[expo] * (1.0 - np.exp(-x[expo]))
        #     + self.theta[expo] * np.exp(-x[expo]))
        # theta_new[~expo] = (self.theta[~expo]
        #     + dt * (1.0 - self.V[~expo] * self.theta[~expo] / p.L))
        # self.theta = theta_new

        self.tau   = tauqs_col + stress.tau0 - p.eta * self.V
        self.U     = self.U + dt * self.V

    # Tip: The following two methods are alternative implementations of the slip rate solver. 
    # The first one uses a custom bisection method, while the second one uses scipy's brentq. 
    # Currently, they are not used
    def solve_slip_rate(self,
                    tauqs_col: np.ndarray,
                    stress: StressState,
                    fric: FrictionalZones):

        p = self.p

        for iy in range(p.Ny):

            rhs = tauqs_col[iy] + stress.tau0[iy]
            a_i = fric.a[iy]
            b_i = fric.b[iy]
            th  = self.theta[iy]
            sig = self.sigma[iy]

            arg = (p.mu0 + b_i * np.log(p.V0 * th / p.L)) / a_i

            flash_denom = 1.0 + p.L/(p.Vw*th)

            exp_arg = np.exp(arg)

            def equation(VV):
                friction = (sig * a_i * np.arcsinh(VV/(2.0*p.V0) * exp_arg))
                return friction/flash_denom + p.eta*VV - rhs

            # guaranteed bracket
            lo = 1e-40

            #hi = max(2.0 * rhs / p.eta, float(np.max(self.V))) 
            #hi = np.max(self.V)*2
            #hi = max(1e-20, np.max(self.V)*2)
            hi = self.V[iy]*2

            try:

                x, fx, flag = MathUtil.bisection(
                    equation,
                    lo,
                    hi,
                    target=0.0,
                    tolX=0.0,
                    tolFun=5,
                    maxiter=100)

                if np.isfinite(x):
                    self.V[iy] = x

            except Exception as e:

                print(
                    f"[iy={iy}] solver failed: {e} "
                    f"f(lo)={equation(lo):.3e} "
                    f"f(hi)={equation(hi):.3e}"
                )
        
        self.V = np.maximum(self.V, 1e-40)

        self.V[0]  = self.V[1]
        self.V[-1] = self.V[-2]