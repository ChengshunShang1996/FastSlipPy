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
from typing import Optional
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
                 fric: FrictionalZones, fault_y: Optional[np.ndarray] = None):
        self.p = p
        Ny = p.Ny
        self._fault_y = None if fault_y is None else np.asarray(fault_y, dtype=float)
        self.U     = np.zeros(Ny)
        self.V     = np.full(Ny, p.Vi)
        if p.case_type == "groningen" or p.case_type == "california":
            logarg = 2 * p.V0 / p.Vi * np.sinh((stress.tau0 - p.eta * p.Vi) / fric.a / stress.sigman0)
            if np.any(logarg <= 0):
                raise ValueError(
                    "BP3 initial-state inversion produced a non-positive "
                    "logarithm argument."
                )
            self.theta = (
                p.L / p.V0
                * np.exp(fric.a / fric.b * np.log(logarg) - p.mu0 / fric.b)
            )
        elif p.case_type == "lab":
            self.theta = np.full(p.Ny, p.L / p.V0)
        self.sigma = stress.sigman0.copy()
        self.tau   = stress.tau0 - p.eta * self.V
        if p.case_type == "california":
            start_idx = self.california_loading_start_idx()
            self.V[start_idx:] = p.loading.V_L

    def california_loading_start_idx(self) -> int:
        """
        Return the first fault-node index where y >= W_f.
        """
        p = self.p
        if self._fault_y is not None:
            idx = int(np.searchsorted(self._fault_y, p.W_f, side="left"))
            return int(np.clip(idx, 0, p.Ny))
        uniform_y = np.linspace(0.0, p.ysize, p.Ny)
        return int(np.searchsorted(uniform_y, p.W_f, side="left"))

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

    def solve_slip_rate_newton_v2(
                        self,
                        tauqs_col: np.ndarray,
                        stress: StressState,
                        fric: FrictionalZones):
        """Solve the signed rate-state equation with safeguarded Newton steps.

        Unlike :meth:`solve_slip_rate_newton`, this version solves for
        ``log(abs(V))`` on either velocity branch. A monotonic root bracket is
        maintained throughout the iteration, so an unsafe Newton step can be
        replaced by a bracketed step without losing the root. The original
        Newton solver is intentionally retained for comparison.
        """
        p = self.p
        tauqs_col = np.asarray(tauqs_col, dtype=float)
        if tauqs_col.shape != (p.Ny,):
            raise ValueError(
                f"tauqs_col must have shape ({p.Ny},), got {tauqs_col.shape}."
            )

        creep_start = (
            self.california_loading_start_idx()
            if p.case_type == "california"
            else p.Ny
        )
        solved = self.V.copy()

        for iy in range(creep_start):
            rhs = float(tauqs_col[iy] + stress.tau0[iy])
            if not np.isfinite(rhs):
                raise RuntimeError(
                    f"Newton v2 received a non-finite driving stress at node {iy}."
                )
            if rhs == 0.0:
                solved[iy] = 0.0
                continue

            a_i = float(fric.a[iy])
            b_i = float(fric.b[iy])
            theta_i = float(self.theta[iy])
            sigma_i = float(self.sigma[iy])
            if a_i <= 0.0 or theta_i <= 0.0 or sigma_i <= 0.0:
                raise RuntimeError(
                    "Newton v2 requires positive a, theta, and effective "
                    f"normal stress (fault node {iy})."
                )

            exponent = (
                p.mu0 + b_i * np.log(p.V0 * theta_i / p.L)
            ) / a_i
            flash = (
                1.0 + p.L / (p.Vw * theta_i)
                if p.flash_heating_option
                else 1.0
            )
            friction_scale = sigma_i * a_i / flash
            log_multiplier = exponent - np.log(2.0 * p.V0)
            target = abs(rhs)
            velocity_sign = 1.0 if rhs > 0.0 else -1.0

            if not np.isfinite(log_multiplier) or friction_scale <= 0.0:
                raise RuntimeError(
                    f"Newton v2 received invalid friction data at node {iy}."
                )

            def residual_and_log_derivative(speed):
                """Return F(speed) and dF/d(log(speed)), avoiding overflow."""
                log_speed = np.log(speed)
                log_q = log_speed + log_multiplier
                if log_q > 20.0:
                    asinh_q = log_q + np.log1p(
                        np.sqrt(1.0 + np.exp(-2.0 * log_q))
                    )
                    q_over_hypot = 1.0 / np.sqrt(
                        1.0 + np.exp(-2.0 * log_q)
                    )
                else:
                    q = np.exp(log_q)
                    asinh_q = np.arcsinh(q)
                    q_over_hypot = q / np.hypot(1.0, q)

                residual = (
                    friction_scale * asinh_q + p.eta * speed - target
                )
                derivative = (
                    friction_scale * q_over_hypot + p.eta * speed
                )
                return residual, derivative

            # F is strictly increasing. Radiation damping therefore gives the
            # finite analytical upper bound ``target / eta``.
            guaranteed_upper = target / p.eta
            previous_speed = abs(float(self.V[iy]))
            if not np.isfinite(previous_speed):
                previous_speed = abs(float(p.Vi))
            smallest_speed = np.nextafter(0.0, 1.0)
            upper = min(
                max(2.0 * previous_speed, smallest_speed),
                guaranteed_upper,
            )
            f_upper, _ = residual_and_log_derivative(upper)

            # Normally the previous solution gives a tight bracket. If stress
            # changed sharply, expand it and finally use the analytical bound.
            for _ in range(64):
                if f_upper >= 0.0:
                    break
                next_upper = min(2.0 * upper, guaranteed_upper)
                if next_upper <= upper:
                    break
                upper = next_upper
                f_upper, _ = residual_and_log_derivative(upper)
            if f_upper < 0.0:
                upper = guaranteed_upper
                f_upper, _ = residual_and_log_derivative(upper)
            if not np.isfinite(f_upper) or f_upper < 0.0:
                raise RuntimeError(
                    f"Newton v2 could not bracket the friction root at node {iy}."
                )

            lower = 0.0
            speed = min(max(previous_speed, smallest_speed), upper)
            converged = False
            for _ in range(100):
                residual, derivative = residual_and_log_derivative(speed)
                if not np.isfinite(residual) or not np.isfinite(derivative):
                    raise RuntimeError(
                        f"Newton v2 became non-finite at fault node {iy}."
                    )
                if abs(residual) <= p.friction_tolerance:
                    converged = True
                    break

                if residual > 0.0:
                    upper = speed
                else:
                    lower = speed

                if derivative > 0.0:
                    newton_log_speed = np.log(speed) - residual / derivative
                    log_lower = np.log(lower) if lower > 0.0 else -np.inf
                    log_upper = np.log(upper)
                    newton_speed = (
                        np.exp(newton_log_speed)
                        if np.isfinite(newton_log_speed)
                        and log_lower < newton_log_speed < log_upper
                        else np.nan
                    )
                else:
                    newton_speed = np.nan

                if not np.isfinite(newton_speed) or not (
                    lower < newton_speed < upper
                ):
                    newton_speed = (
                        np.sqrt(lower * upper)
                        if lower > 0.0
                        else 0.5 * upper
                    )

                if newton_speed == speed:
                    # No further representable progress is possible.
                    converged = abs(residual) <= max(
                        p.friction_tolerance,
                        16.0 * np.finfo(float).eps * target,
                    )
                    break
                speed = newton_speed

            if not converged:
                residual, _ = residual_and_log_derivative(speed)
                raise RuntimeError(
                    "Newton v2 did not converge at fault node "
                    f"{iy} (friction residual={residual:g} Pa)."
                )
            solved[iy] = velocity_sign * speed

        self.V[:] = solved
        if p.case_type == "california":
            self.V[creep_start:] = p.loading.V_L
    
    # ------------------------------------------------------------------
    def advance(self, dt: float, tauqs_col: np.ndarray, stress: StressState):
        """Update theta, U, tau after the velocity solve."""
        p = self.p
        
        # self.theta = self.theta + dt * (1 - self.V * self.theta / p.L)
        # TODO: this one is better
        speed = np.abs(self.V)
        x = speed * dt / p.L
        expo = x > 1e-6
        theta_new = np.empty_like(self.theta)
        theta_new[expo] = (
            p.L / speed[expo] * (1.0 - np.exp(-x[expo]))
            + self.theta[expo] * np.exp(-x[expo]))
        theta_new[~expo] = (self.theta[~expo]
            + dt * (1.0 - speed[~expo] * self.theta[~expo] / p.L))
        self.theta = theta_new

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
            start_idx = self.california_loading_start_idx()
            self.V[start_idx : p.Ny] = -1 * p.loading.V_L 

        self.V = np.maximum(self.V, 1e-40)

        self.V[0]  = self.V[1]
        self.V[-1] = self.V[-2]

    def solve_slip_rate_matlab(
                        self,
                        tauqs_col: np.ndarray,
                        stress: StressState,
                        fric: FrictionalZones):
        """Solve the signed BP3 friction equation as in the MATLAB reference."""

        p = self.p
        if p.case_type == "california":
            creep_start = self.california_loading_start_idx()
        else:
            creep_start = p.Ny

        active = np.arange(creep_start, dtype=int)
        if active.size == 0:
            self.V[:] = p.loading.V_L
            return

        bracket = 2.0 * float(np.max(np.abs(self.V[active])))
        bracket = max(bracket, np.finfo(float).tiny)
        positive_branch = p.loading.V_p > 0.0
        lower, upper = ((0.0, bracket) if positive_branch else (-bracket, 0.0))

        solved = self.V.copy()
        for iy in active:
            rhs = tauqs_col[iy] + stress.tau0[iy]
            a_i = fric.a[iy]
            b_i = fric.b[iy]
            theta_i = self.theta[iy]
            sigma_i = self.sigma[iy]
            exponent = (
                p.mu0 + b_i * np.log(p.V0 * theta_i / p.L)
            ) / a_i
            exp_exponent = np.exp(exponent)
            flash = (
                1.0 + p.L / (p.Vw * theta_i)
                if p.flash_heating_option
                else 1.0
            )

            def residual(velocity):
                friction = sigma_i * a_i * np.arcsinh(
                    velocity / (2.0 * p.V0) * exp_exponent
                )
                return friction / flash + p.eta * velocity - rhs

            lo = lower
            hi = upper
            f_lo = residual(lo)
            f_hi = residual(hi)
            if abs(f_lo) <= p.friction_tolerance:
                solved[iy] = lo
                continue
            if abs(f_hi) <= p.friction_tolerance:
                solved[iy] = hi
                continue
            if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_lo * f_hi > 0:
                raise RuntimeError(
                    "BP3 friction solve failed: the root left the dynamic "
                    f"bracket at fault node {iy} (half-width={bracket:g} m/s)."
                )

            for _ in range(1000):
                midpoint = 0.5 * (lo + hi)
                f_mid = residual(midpoint)
                if not np.isfinite(f_mid):
                    raise RuntimeError(
                        f"BP3 friction residual became non-finite at node {iy}."
                    )
                if abs(f_mid) <= p.friction_tolerance:
                    solved[iy] = midpoint
                    break
                if np.signbit(f_mid) == np.signbit(f_hi):
                    hi = midpoint
                    f_hi = f_mid
                else:
                    lo = midpoint
                    f_lo = f_mid
            else:
                raise RuntimeError(
                    f"BP3 friction bisection did not converge at node {iy}."
                )

        self.V[:] = solved
        if p.case_type == "california":
            self.V[creep_start:] = p.loading.V_L

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
            start_idx = self.california_loading_start_idx()
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
