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
        if p.case_type == "groningen" or p.case_type == "california":
            logarg = 2 * p.V0 / p.Vi * np.sinh((stress.tau0 - p.eta * p.Vi) / fric.a / stress.sigman0)
            safe_logarg = np.maximum(logarg, 1e-30)
            if np.any(logarg <= 0):
                print(logarg)
                print("Warning: logarg has non-positive values, which may cause issues in the solver.")
            self.theta = (p.L / p.V0 * np.exp(fric.a / fric.b * np.log(safe_logarg)- p.mu0 / fric.b))
            #self.theta = (p.L / p.V0 * np.exp(fric.a / fric.b * np.emath.log(logarg)- p.mu0 / fric.b))
        elif p.case_type == "lab":
            self.theta = np.full(p.Ny, p.L / p.V0)
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

            if p.flash_heating_option:
                flash_denom = 1.0 + p.L/(p.Vw*th)
            else:
                flash_denom = 1.0

            exp_arg = np.exp(arg)

            def equation(VV):
                friction = (sig * a_i * np.arcsinh(VV/(2.0*p.V0) * exp_arg))
                return friction/flash_denom + p.eta*VV - rhs
            
            if rhs == 0 or rhs < 0:
                self.V[iy] = 1e-40
                continue

            #print(f"Solving for V at iy={iy} with theta={th:.3e}, arg={arg:.3e}, exp_arg={exp_arg:.3e}, sigma={sig:.3e}")
            
            # guaranteed bracket
            lo = 1e-40

            #hi = max(2.0 * rhs / p.eta, float(np.max(self.V))) 
            #hi = np.max(self.V)*2
            hi = max(1e-20, np.max(self.V)*2)
            #hi = self.V[iy]*2

            try:

                x, fx, flag = MathUtil.bisection(
                    equation,
                    lo,
                    hi,
                    target=0.0,
                    tolX=0.0, #1e-14,
                    tolFun=5,
                    maxiter=1000)

                # if np.isfinite(x):
                #     self.V[iy] = x
                if flag > 0:
                    self.V[iy] = x
                else:              
                    if p.run_mode == "debug":
                        print(
                                iy,
                                f"theta={th:.3e}",
                                f"arg={arg:.3e}",
                                f"exp_arg={exp_arg:.3e}",
                                f"rhs={rhs:.3e}",
                                f"sig={sig:.3e}",
                                f"f(lo)={equation(lo):.3e}",
                                f"f(hi)={equation(hi):.3e}",
                                f"eta = {p.eta:.3e}",
                                f"flag={flag}"
                            )
                        print(
                                "tauqs =", tauqs_col[iy],
                                "tau0  =", stress.tau0[iy],
                                "rhs   =", rhs
                            ) 
                        print("root failed", iy, flag)

            except Exception as e:

                if p.run_mode == "debug":
                    print(
                        f"[iy={iy}] solver failed: {e} "
                        f"f(lo)={equation(lo):.3e} "
                        f"f(hi)={equation(hi):.3e}"
                    )
        
        if p.case_type == "california":
            start_idx = int(p.W_f // (p.ysize / (p.Ny - 1)) + 1)
            self.V[start_idx : p.Ny] = -1 * p.loading.V_L 

        self.V = np.maximum(self.V, 1e-40)

        self.V[0]  = self.V[1]
        self.V[-1] = self.V[-2]

    def solve_slip_rate_newton(self,
                        tauqs_col: np.ndarray,
                        stress: StressState,
                        fric: FrictionalZones):
        """
        Solves for the fault slip velocity V (Slip Rate) under Rate-and-State friction laws.
        
        Uses a Newton-Raphson scheme in log-space (y = ln V) equipped with 
        Step Clipping and Safeguarded Bounds for absolute convergence and speed.
        """
        p = self.p

        for iy in range(p.Ny):
            # 1. Compute total driving shear stress (Right-Hand Side)
            rhs = tauqs_col[iy] + stress.tau0[iy]
            
            # Physical constraint: If driving stress is non-positive, fault cannot slip forward.
            if rhs <= 0.0:
                self.V[iy] = 1e-40
                continue

            a_i = fric.a[iy]
            b_i = fric.b[iy]
            th  = self.theta[iy]
            sig = self.sigma[iy]

            # 2. Compute state-dependent parameters
            arg = (p.mu0 + b_i * np.log(p.V0 * th / p.L)) / a_i
            exp_arg = np.exp(arg)

            # Flash heating modification factor
            if p.flash_heating_option:
                flash_denom = 1.0 + p.L / (p.Vw * th)
            else:
                flash_denom = 1.0

            # Simplified coefficients for: C * arcsinh(B * V) + eta * V = rhs
            C = (sig * a_i) / flash_denom
            B = exp_arg / (2.0 * p.V0)

            # 3. Newton-Raphson iteration in log-space (y = ln V)
            # Clamp initial guess logV within safe numerical bounds [-92.1, 4.6] (V in 1e-40 to 100 m/s)
            v_init = np.clip(self.V[iy], 1e-40, 100.0)
            logV = np.log(v_init)
            
            converged = False
            max_iters = 30
            
            for iteration in range(max_iters):
                # Guard against overflow: bound logV to prevent np.exp overflow
                logV = np.clip(logV, -92.1, 11.5)  # exp(-92.1) ~ 1e-40, exp(11.5) ~ 1e5
                V = np.exp(logV)
                BV = B * V
                
                # Compute residual: f(y) = C * arcsinh(B * V) + eta * V - rhs
                arcsinh_val = np.arcsinh(BV)
                f = C * arcsinh_val + p.eta * V - rhs
                
                # Derivative: df/dy = V * (df/dV)
                # where d/dV [arcsinh(B*V)] = B / sqrt(1 + (B*V)^2)
                df_dV = C * (B / np.sqrt(1.0 + BV**2)) + p.eta
                df_dy = V * df_dV
                
                # Prevent division by zero or underflow
                if abs(df_dy) < 1e-30 or np.isnan(df_dy):
                    break

                # Key Fix: Step Clipping
                # Restrict max single-step update to delta logV = 5.0 (velocity change <= e^5 approx. 148x)
                dy = - f / df_dy
                dy = np.clip(dy, -5.0, 5.0)
                
                logV += dy
                
                # Convergence criteria
                if abs(dy) < 1e-12 or abs(f) < 1e-10:
                    self.V[iy] = np.exp(logV)
                    converged = True
                    break

            # 4. Fallback solver (In rare non-convergence edge cases, use Brent's method)
            if not converged:
                def backup_eq(VV):
                    return C * np.arcsinh(B * VV) + p.eta * VV - rhs

                try:
                    from scipy.optimize import brentq
                    # Use a safe physical velocity bracket [1e-40, 100.0 m/s]
                    self.V[iy] = brentq(backup_eq, 1e-40, 100.0, xtol=1e-13, rtol=1e-13)
                except Exception:
                    # If backup fails in extreme cases, retain clipped value
                    self.V[iy] = np.exp(np.clip(logV, -92.1, 11.5))

        # 5. California benchmark loading boundary condition
        if p.case_type == "california":
            start_idx = int(p.W_f // (p.ysize / (p.Ny - 1)) + 1)
            self.V[start_idx : p.Ny] =  p.loading.V_L

        # 6. Apply floor truncation and update ghost cells
        self.V = np.maximum(self.V, 1e-40)
        self.V[0]  = self.V[1]
        self.V[-1] = self.V[-2]
    
    def solve_slip_rate_3(self,
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
            
            lo = 1e-40
            hi = self.V[iy]*2

            if rhs/sig < p.mu0:
                self.V[iy] = lo
            else:
                if hi > 1e-5:
                    hi = 1e-5
                self.V[iy] = hi
        
        self.V = np.maximum(self.V, 1e-40)

        self.V[0]  = self.V[1]
        self.V[-1] = self.V[-2]