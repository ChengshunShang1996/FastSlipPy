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

class MathUtil:
    """
    A collection of math utility functions.
    """

    @staticmethod
    def cosd(angle_deg, tol=1e-15):

        angle = angle_deg % 360

        if np.isclose(angle, 0, atol=tol):
            return 1.0

        if np.isclose(angle, 90, atol=tol):
            return 0.0

        if np.isclose(angle, 180, atol=tol):
            return -1.0

        if np.isclose(angle, 270, atol=tol):
            return 0.0

        return np.cos(np.deg2rad(angle))


    @staticmethod
    def sind(angle_deg, tol=1e-15):

        angle = angle_deg % 360

        if np.isclose(angle, 0, atol=tol):
            return 0.0

        if np.isclose(angle, 90, atol=tol):
            return 1.0

        if np.isclose(angle, 180, atol=tol):
            return 0.0

        if np.isclose(angle, 270, atol=tol):
            return -1.0

        return np.sin(np.deg2rad(angle))
    
    def log_bisection(func, lo, hi, tol_log=1e-14, tol_f=1e-12, maxiter=200):
        """
        Robust log-space bisection solver.

        Solves:
            func(V) = 0

        using:
            x = log(V)

        Extremely stable for:
            V ~ 1e-40 ... 1e+10
        """

        f_lo = func(lo)
        f_hi = func(hi)

        # root must be bracketed
        if f_lo * f_hi > 0:
            return np.nan

        log_lo = np.log(lo)
        log_hi = np.log(hi)

        for _ in range(maxiter):

            log_mid = 0.5 * (log_lo + log_hi)

            mid = np.exp(log_mid)

            f_mid = func(mid)

            # convergence tests
            if abs(log_hi - log_lo) < tol_log:
                return mid

            if abs(f_mid) < tol_f:
                return mid

            # keep bracket
            if f_lo * f_mid < 0:
                log_hi = log_mid
                f_hi = f_mid
            else:
                log_lo = log_mid
                f_lo = f_mid

        return np.exp(0.5 * (log_lo + log_hi))

    def bisection(f, lb, ub, target=0.0, tolX=1e-6, tolFun=0.0, maxiter=1000):

        # shift function by target
        def g(x):
            return f(x) - target

        flb = g(lb)
        fub = g(ub)

        # if flb == 0:
        #     return lb, target, 3
        # if fub == 0:
        #     return ub, target, 3
        if abs(flb) <= tolFun:
            return lb, flb + target, 3
        if abs(fub) <= tolFun:
            return ub, fub + target, 3

        # root must be bracketed
        # if flb * fub > 0:
        #     return np.nan, np.nan, -2
        #     #raise ValueError("Root is not bracketed: f(lb) and f(ub) must have opposite signs.")
        
        # -------------------------------------------------
        # adaptive bracket expansion
        # -------------------------------------------------

        expand_iter = 0
        max_expand = 30

        while flb * fub > 0:
            ub *= 10.0
            fub = g(ub)
            expand_iter += 1
            if expand_iter > max_expand:
                return np.nan, np.nan, -2
                #raise ValueError("Root is not bracketed: f(lb) and f(ub) must have opposite signs.")

        ubSign = np.sign(fub)

        for _ in range(maxiter):
            x = 0.5 * (lb + ub)
            fx = g(x)

            outsideTolX = abs(ub - x) > tolX
            outsideTolFun = abs(fx) > tolFun

            # convergence
            if not (outsideTolX and outsideTolFun):
                return x, fx + target, 3

            # if not outsideTolX:
            #     return x, fx + target, 1

            # if not outsideTolFun:
            #     return x, fx + target, 2

            # keep bracket
            if np.sign(fx) != ubSign:
                lb = x
                #flb = fx
            else:
                ub = x
                #fub = fx

        return x, fx + target, -1