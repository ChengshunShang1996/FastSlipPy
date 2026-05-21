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

"""
Classes
-------
ModelParameters   – all physical / numerical input parameters
Grid              – spatial discretisation and coordinate arrays
FrictionalZones   – depth-dependent rate-and-state a/b profiles
StressState       – initial stress, pressure, sigma, tau
FaultState        – fault slip variables (U, V, theta, sigma, tau)
MatrixBuilder     – builds LH (stiffness) and RH (forcing) sparse matrices
OutputManager     – in-memory storage + text-file logging + checkpointing
FaultSlipModel    – top-level driver that wires everything together

Usage
-----
    model = FaultSlipModel()
    model.run()
"""

import os
import json
import sys
import time
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from scipy import sparse
from scipy.sparse.linalg import factorized
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from scipy.optimize import root_scalar
from numba import njit

# ─────────────────────────────────────────────────────────────
# 1.  Model Parameters
# ─────────────────────────────────────────────────────────────

@dataclass
class ModelParameters:
    """
    All physical and numerical parameters for the fault-slip model.
    Edit the defaults here or pass keyword arguments to the constructor.
    """
    # --- Fault geometry ---
    alpha: float = 70.0          # Fault dip angle [degrees]

    # --- Grid ---
    xsize: float = 2000.0        # Horizontal model size [m]
    ysize: float = 2000.0        # Vertical model size [m]
    Nx: int = 201                # Horizontal grid points  (must be odd)
    Ny: int = 201                # Vertical grid points    (must be odd)

    # --- Material ---
    rho: float = 2400.0          # Rock density [kg/m³]
    rhof: float = 1150.0         # Fluid density [kg/m³]
    rhog: float = 200.0          # Gas density [kg/m³]
    Vp: float = 0.0              # Far-field loading rate [m/s]
    cs: float = 1645.0           # Shear-wave speed [m/s]
    nu: float = 0.15             # Poisson's ratio
    g: float = 9.81              # Gravitational acceleration [m/s²]
    K0: float = 0.75             # Ratio σ_min / σ_max

    # --- Rate-and-state defaults (used when heterogeneous profile is off) ---
    mu0: float = 0.3             # Reference friction coefficient
    V0: float = 1e-6             # Reference slip rate [m/s]
    a0: float = 0.01             # Direct effect (homogeneous fallback)
    b0: float = 0.015            # Evolution effect (homogeneous fallback)
    L: float = 0.5               # Characteristic slip distance [m]
    Vw: float = 1e90             # Dynamic weakening velocity [m/s]
    Vi: float = 1e-30            # Initial/background slip rate [m/s]

    # --- Time stepping ---
    Nt: int = 200               # Number of time steps
    dt_init: float = 1.0         # Initial time step [s]
    dt_max: float = 1e6          # Maximum time step [s]
    yr = 365 * 24 * 3600.0       # Seconds in a year
    tload: float = 10.0 * yr     # Time to apply pressure rate change [s]  #TODO: CHECK

    # --- Pressure rate ---
    dPdt_pre: float = 0.0       # Pressure rate before depletion [Pa/s]
    dPdt_post: float = -0.0127  # Pressure rate after depletion starts [Pa/s]

    # --- Output intervals ---
    output_interval: int = 10
    checkpoint_interval: int = 1000

    # --- Derived (computed in __post_init__) ---
    G: float = field(init=False)
    lam: float = field(init=False)   # First Lamé parameter (λ)
    eta: float = field(init=False)   # Radiation damping coefficient

    def __post_init__(self):
        
        self.G = self.rho * self.cs ** 2
        self.lam = 2 * self.G * (1 + self.nu) / 3 / (1 - 2 * self.nu) - 2 / 3 * self.G
        self.eta = self.G / 2 / self.cs
        #if self.tload is None:
        #    self.tload = 1000 * yr
        assert self.Nx % 2 == 1, "Nx must be odd (fault at centre column)."
        assert self.Ny % 2 == 1, "Ny must be odd."


# ─────────────────────────────────────────────────────────────
# 2.  Grid
# ─────────────────────────────────────────────────────────────

class Grid:
    """
    Builds and stores all spatial coordinate arrays needed by the model.
    """

    def __init__(self, p: ModelParameters):
        
        self.p = p
        self.sina = sind(p.alpha)
        self.cosa = cosd(p.alpha)

        Nx, Ny = p.Nx, p.Ny
        dx = p.xsize / (Nx - 1)
        dy = p.ysize / (Ny - 1)
        self.dx = dx
        self.dy = dy

        # Basic (τ / σ) nodes
        self.x = np.linspace(-p.xsize / 2, p.xsize / 2, Nx)          # (Nx,)
        self.y = np.linspace(0, p.ysize, Ny)                           # (Ny,)

        # Pressure / staggered nodes
        self.xp = np.linspace(-p.xsize / 2 - dx / 2,
                             p.xsize / 2 + dx / 2, Nx+1)       # (Nx+1,)
        self.yp = np.linspace(-dy / 2, p.ysize + dy / 2, Ny+1)                # (Ny+1,)

        #print(self.xp)
        #print(self.yp)

        # Rotated coordinate helpers (kept for post-processing / plotting)
        self.Xuy  = self.y[:, None] * self.cosa + self.xp[None, :]
        self.Yuy  = self.y[:, None] * self.sina + 0 * self.xp[None, :]
        self.Xux  = self.yp[:, None] * self.cosa + self.x[None, :]
        self.Yux  = self.yp[:, None] * self.sina + 0 * self.x[None, :]
        self.Xtau = self.y[:, None] * self.cosa + self.x[None, :]
        self.Ytau = self.y[:, None] * self.sina + 0 * self.x[None, :]
        self.Xsigma = self.yp[1:Ny, None] * self.cosa + self.xp[None, 1:Nx]
        self.Ysigma = self.yp[1:Ny, None] * self.sina + 0 * self.xp[None, 1:Nx]

        # Total DOF count
        self.N = (Nx + 1) * (Ny + 1) * 2

    # ------------------------------------------------------------------
    def plot_mesh(self):
        X0, Y0 = np.meshgrid(self.x, self.y+2000)
        X = Y0 * self.cosa + X0
        Y = Y0 * self.sina
        fig, ax = plt.subplots()
        ax.plot(X, Y, 'k', linewidth=0.4)
        ax.plot(X.T, Y.T, 'k', linewidth=0.4)
        #ax.set_xlim(-500, 2500)
        #ax.set_ylim(1800, 3800)

        y_line = self.y + 2000
        x_line = np.zeros_like(self.y)   # fault 在 x = 0
        X_fault = y_line * self.cosa + x_line
        Y_fault = y_line * self.sina
        ax.plot(X_fault, Y_fault, 'm--', linewidth=2, label='fault')

        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title('2-D Mesh Grid')
        plt.tight_layout()
        return fig
    
    def plot_grid(self):

        fig, ax = plt.subplots(figsize=(6, 6))

        # ─────────────────────────────
        # 1. τ / σ base grid (black)
        # ─────────────────────────────
        #X, Y = np.meshgrid(self.x, self.y)
        #ax.plot(X, Y, 'k-', linewidth=0.5)
        #ax.plot(X.T, Y.T, 'k-', linewidth=0.5)

        X0, Y0 = np.meshgrid(self.x, self.y)
        X = Y0 * self.cosa + X0
        Y = Y0 * self.sina
        ax.plot(X, Y, 'k-', linewidth=0.5)
        ax.plot(X.T, Y.T, 'k-', linewidth=0.5)

        # ─────────────────────────────
        # 2. uy points（blue）
        # ─────────────────────────────
        ax.scatter(self.Xuy, self.Yuy, marker='^', c='blue', label='uy nodes')

        # ─────────────────────────────
        # 3. ux points（red）
        # ─────────────────────────────
        ax.scatter(self.Xux, self.Yux, marker='^', c='red', label='ux nodes')

        # ─────────────────────────────
        # 4. sigma points（green)
        # ─────────────────────────────

        ax.scatter(self.Xsigma, self.Ysigma, marker='s', c='green', label='sigma nodes')

        ax.scatter(self.Xtau, self.Ytau, s=5, c='black', label='tau nodes')

        XP0, YP0 = np.meshgrid(self.xp, self.yp)
        XP = YP0 * self.cosa + XP0
        YP = YP0 * self.sina
        ax.scatter(XP, YP, s=5, c='orange', label='pressure nodes')

        # ─────────────────────────────
        # 5. fault position（middle column）
        # ─────────────────────────────
        mid = len(self.x) // 2
        ax.axvline(self.x[mid], color='magenta', linestyle='--', label='fault')

        # ─────────────────────────────
        # 6. settings
        # ─────────────────────────────
        ax.set_aspect('equal')
        ax.set_title("Grid + Staggered Layout")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc='upper right', fontsize=8)

        plt.tight_layout()
        plt.show()


# ─────────────────────────────────────────────────────────────
# 3.  Frictional Zones  (rate-and-state a, b profiles)
# ─────────────────────────────────────────────────────────────

class FrictionalZones:
    """
    Assigns depth-dependent rate-and-state parameters a(y) and b(y).

    The heterogeneous stratigraphy matches the Groningen / Slochteren
    reservoir setting.  You can subclass or replace `build()` to supply
    any profile you like.
    """

    # Layer depths relative to surface (positive downward convention)
    # These are absolute depths [m from surface].  ysize is subtracted to
    # convert to the model's coordinate system where y=0 is the top.
    LAYERS = {
        "Rocksalt":   {"top": 2000, "bot": 2730, "a": 0.00447,  "b": -0.00590}, # Zechstein rocksalt (halite)
        "BasalZech":  {"top": 2730, "bot": 2780, "a": 0.06895,  "b":  0.07209}, # Basal zechstein
        "TenBoer":    {"top": 2780, "bot": 2850, "a": 0.00305,  "b": -0.00093}, # Ten Boer
        "Sandstone":  {"top": 2850, "bot": 3050, "a": 0.04065,  "b":  0.03796}, # Slochteren Sandstone
        "Carbonif":   {"top": 3050, "bot": 4000, "a": 0.02538,  "b":  0.02347}, # Carboniferous member
    }

    # LAYERS = {
    #     "Sandstone":  {"top": 2000, "bot": 4000, "a": 0.04,  "b":  0.03} # Slochteren Sandstone
    # }

    def __init__(self, p: ModelParameters, y: np.ndarray):
        self.p = p
        self.y = y
        self.a, self.b = self.build()

    # ------------------------------------------------------------------
    def build(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (a, b) arrays of shape (Ny,) using the layer definitions above.
        Override this method to supply a custom depth profile.
        """
        p = self.p
        y = self.y
        a = np.zeros_like(y)
        b = np.zeros_like(y)

        layers = list(self.LAYERS.items())

        for i, (name, layer) in enumerate(layers):

            # Convert absolute depth [m] to model y-coordinate
            top_y = layer["top"] - p.ysize
            bot_y = layer["bot"] - p.ysize

            # First layer
            if i == 0:
                mask = y <= bot_y

            # Last layer
            elif i == len(layers) - 1:
                mask = y > top_y

            # Middle layers
            else:
                mask = (y > top_y) & (y <= bot_y)

            a[mask] = layer["a"]
            b[mask] = layer["b"]

        # Fill any unassigned nodes with homogeneous defaults
        # THIS SHOULD NOT HAPPEN IF LAYERS COVER THE ENTIRE DEPTH RANGE
        #a[a == 0] = p.a0
        #b[b == 0] = p.b0

        return a, b


# ─────────────────────────────────────────────────────────────
# 4.  Stress State
# ─────────────────────────────────────────────────────────────

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

    # ------------------------------------------------------------------
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
                   + (1 - p.K0) / 2 * cosd(2 * p.alpha) * sigmav
                   - np.where(y < 1000, Pl0, Pr0))
        tau0 = (1 - p.K0) / 2 * sind(2 * p.alpha) * sigmav
        return sigman0, tau0, Pl0, Pr0

    # ------------------------------------------------------------------
    def update_pressure(self, dt: float, dPdt: float):
        """Advance pore pressures by one time step."""
        y = self.y
        self.Pl += dPdt * dt * (y > 800) * (y <= 1000)
        self.Pr += dPdt * dt * (y > 850) * (y <= 1050)
        self.P   = np.where(y < 1000, self.Pl, self.Pr)


# ─────────────────────────────────────────────────────────────
# 5.  Fault State
# ─────────────────────────────────────────────────────────────

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

    # ------------------------------------------------------------------
    def solve_slip_rate2(self,
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

            # arg = np.clip(
            #     (p.mu0 + b_i*np.log(p.V0*th/p.L))/a_i,
            #     -500.0,
            #     500.0
            # )

            arg = (p.mu0 + b_i * np.log(p.V0 * th / p.L)) / a_i

            flash_denom = 1.0 + p.L/(p.Vw*th)

            exp_arg = np.exp(arg)

            # -----------------------------------------
            # monotone friction equation
            # -----------------------------------------

            def equation(VV):
                friction = (sig * a_i * np.arcsinh(VV/(2.0*p.V0) * exp_arg))
                return friction/flash_denom + p.eta*VV - rhs

            # guaranteed bracket
            lo = 1e-40

            #hi = max(2.0 * rhs / p.eta, float(np.max(self.V))) 
            hi = np.max(self.V)*2

            try:

                # sol = log_bisection(
                #     equation,
                #     lo,
                #     hi,
                #     tol_log=1e-14,
                #     tol_f=5,
                #     maxiter=200
                # )

                x, fx, flag = bisection(
                    equation,
                    lo,
                    hi,
                    target=0.0,
                    tolX=0.0,
                    tolFun=5,
                    maxiter=1e3)

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

    def solve_slip_rate1(self, tauqs_col: np.ndarray, stress: StressState,
                    fric: FrictionalZones):
        """
        Solve for V at each fault node using the rate-and-state friction law
        with flash heating:

            σ · a · arcsinh[ V/(2V₀) · exp((μ₀ + b·ln(V₀θ/L))/a) ]
                / (1 + L/(Vw·θ))   +   η·V   =   τ_qs + τ₀

        Exact equivalent of MATLAB:
            V(iy) = fzero(@(VV) ..., V(iy))

        Strategy
        --------
        The equation is STRICTLY MONOTONE INCREASING in V (both the arcsinh
        friction term and η·V are non-decreasing, with η·V strictly increasing).
        Therefore there is EXACTLY ONE root, and the bracket

            lo = 1e-40,   hi = 2 · rhs / η

        is GUARANTEED valid for all physically meaningful inputs:
        • f(lo) ≈ –rhs < 0  (arcsinh(0) = 0,  η·0 = 0)
        • f(hi) ≥ η·(2·rhs/η) – rhs = rhs > 0  (friction term ≥ 0)

        brentq on this bracket is the closest Python equivalent to MATLAB's
        fzero, which also uses Brent's method after locating a bracket from x0.
        """
        p = self.p

        for iy in range(p.Ny):
            rhs = tauqs_col[iy] + stress.tau0[iy]

            # rhs must be positive for a physical solution to exist.
            # (Both tauqs and tau0 are positive driving stresses.)
            if rhs <= 0:
                # No positive-V root exists; keep previous value.
                continue

            a_i  = fric.a[iy]
            b_i  = fric.b[iy]
            th   = self.theta[iy]
            sig  = self.sigma[iy]

            # Precompute constants for this node (same for every function eval)
            arg         = np.clip((p.mu0 + b_i * np.log(p.V0 * th / p.L)) / a_i,
                                -500.0, 500.0)
            flash_denom = 1.0 + p.L / (p.Vw * th)

            #print(f"[iy={iy}] tauqs={tauqs_col[iy]:.6e} tau0={stress.tau0[iy]:.6e}  a={a_i:.6e}  b={b_i:.6e}  th={th:.6e}  sig={sig:.6e} mu={p.mu0:.6e}  V0={p.V0:.6e} L={p.L:.6e}  Vw={p.Vw:.6e} eta={p.eta:.6e}")

            def equation(VV):
                friction = sig * a_i * np.arcsinh(VV / (2.0 * p.V0) * np.exp(arg))
                return friction / flash_denom + p.eta * VV - rhs

            # Analytically guaranteed bracket — no search loop needed.
            lo = 1e-40
            hi = 2.0 * rhs / p.eta   # f(hi) >= rhs > 0  always
            #hi = max(self.V)

            #print(f"[iy={iy}] V={self.V[iy]:.6e}")
            
            try:
                #sol = brentq(equation, lo, hi, xtol=1e-14, rtol=1e-12, maxiter=200)
                sol = brentq(
                            equation,
                            lo,
                            hi,
                            xtol=1e-300,
                            rtol=1e-15,
                            maxiter=1000
                        )
                self.V[iy] = sol
            except Exception as e:
                print(f"[iy={iy}] brentq failed: {e}  "
                    f"f(lo)={equation(lo):.3e}  f(hi)={equation(hi):.3e}")
                # Keep previous V[iy]

            #print(f"[iy={iy}] V={self.V[iy]:.6e}")
            #print(f"*************")

        self.V = np.maximum(self.V, 1e-40)
        self.V[0]  = self.V[1]
        self.V[-1] = self.V[-2]

    def solve_slip_rate(self, tauqs_col: np.ndarray, stress: StressState,
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
        #print(f"self.U=[{', '.join(f'{x:.6e}' for x in self.U)}]")
        #print('*************')


# ─────────────────────────────────────────────────────────────
# 6.  Adaptive time-step stabiliser  (ksi)
# ─────────────────────────────────────────────────────────────

def build_ksi(p: ModelParameters, fric: FrictionalZones,
              sigman0: np.ndarray, dy: float) -> np.ndarray:
    """
    Stability factor ksi used for adaptive time stepping:
    """
    a = fric.a
    b = fric.b

    k1 = (np.pi / 4.0) * p.G / dy * p.L / a / sigman0
    k2 = (b - a) / a
    k3 = (k1 - k2)**2 / 4.0 - k1
    k4 = np.minimum(1.0 / (k1 - k2), 0.2)
    k5 = np.minimum(1.0 - k2 / k1, 0.2)
    ksi = np.where(k3 > 0, k4, k5)

    return ksi


# ─────────────────────────────────────────────────────────────
# 7.  Matrix Builder  (LH, RH)
# ─────────────────────────────────────────────────────────────

class MatrixBuilder:
    """
    Assembles the sparse stiffness matrix LH and right-hand-side vector RH
    for the quasi-static elastic equilibrium problem on the staggered grid.

    The stencil follows the original MATLAB build_LH / build_RH logic exactly.
    """

    def __init__(self, p: ModelParameters, grid: Grid):
        self.p    = p
        self.grid = grid

    # ------------------------------------------------------------------
    # Helper: global DOF indices
    # ------------------------------------------------------------------
    @staticmethod
    def _dofs(ix: int, iy: int, Ny: int):
        """Return (kux, kuy) 0-based DOF indices for node (ix, iy)."""
        kux = ((ix) * (Ny + 1) + iy) * 2        # ux DOF (0-based)
        kuy = kux + 1                            # uy DOF (0-based)
        return kux, kuy

    # ------------------------------------------------------------------
    def build_LH(self) -> sparse.csr_matrix:
        p, g = self.p, self.grid
        Nx, Ny, N = p.Nx, p.Ny, g.N
        dx, dy   = g.dx, g.dy
        lam, G   = p.lam, p.G
        sina, cosa = g.sina, g.cosa

        rows, cols, vals = [], [], []

        def add(r, c, v):
            rows.append(r); cols.append(c); vals.append(v)

        for ix in range(Nx+1):           # 0 … Nx  (MATLAB 1 … Nx+1)
            for iy in range(Ny+1):       # 0 … Ny

                kux, kuy = self._dofs(ix, iy, Ny)
                mid = (Nx) // 2            # fault column index (0-based)

                # ── uy equation (iy < Ny) ──────────────────────────────
                if iy < Ny:
                    if ix == 0: # Neumann BC
                        add(kuy, kuy, 1);  add(kuy, kuy + (Ny+1)*2, -1)
                    elif ix == Nx:
                        add(kuy, kuy, 1);  add(kuy, kuy - (Ny+1)*2, -1)
                    elif iy == 0:
                        add(kuy, kuy, 1)
                    elif iy == Ny - 1:
                        add(kuy, kuy, 1)
                    elif ix == mid:
                        # Fault left side
                        add(kuy, kuy, -1); add(kuy, kuy + (Ny+1)*2, 1)
                    elif ix == mid + 1:
                        # Fault right side
                        kux_n, kuy_n = self._dofs(ix, iy, Ny)
                        add(kuy, kuy - 2*(Ny+1)*2, 1)
                        add(kuy, kuy - (Ny+1)*2,  -1)
                        add(kuy, kuy,              -1)
                        add(kuy, kuy + (Ny+1)*2,   1)
                        # Cross-coupling terms with ux
                        add(kuy, kux + (Ny+1)*2,       cosa/4)
                        add(kuy, kux + (Ny+1)*2 + 2,   cosa/4)
                        add(kuy, kux - (Ny+1)*2,       -cosa/2)
                        add(kuy, kux - (Ny+1)*2 + 2,   -cosa/2)
                        add(kuy, kux - 3*(Ny+1)*2,     cosa/4)
                        add(kuy, kux - 3*(Ny+1)*2 + 2, cosa/4)
                        add(kuy, kuy + (Ny+1)*2 - 2,   cosa/4/dy*dx)
                        add(kuy, kuy + (Ny+1)*2 + 2,  -cosa/4/dy*dx)
                        add(kuy, kuy - 2,               cosa/4/dy*dx)
                        add(kuy, kuy + 2,              -cosa/4/dy*dx)
                        add(kuy, kuy - (Ny+1)*2 - 2,  -cosa/4/dy*dx)
                        add(kuy, kuy - (Ny+1)*2 + 2,   cosa/4/dy*dx)
                        add(kuy, kuy - 2*(Ny+1)*2 - 2,  -cosa/4/dy*dx)
                        add(kuy, kuy - 2*(Ny+1)*2 + 2,   cosa/4/dy*dx)
                    else:
                        # Interior bulk
                        r2 = dx*dx / dy/dy * (lam + 2*G) / G
                        add(kuy, kuy, -2 - 2*r2)
                        add(kuy, kuy - (Ny+1)*2, 1)
                        add(kuy, kuy + (Ny+1)*2, 1)
                        add(kuy, kuy - 2,  r2)
                        add(kuy, kuy + 2,  r2)
                        c_val = cosa/dy*dx*(lam + 3*G)/G/4
                        add(kuy, kuy + (Ny+1)*2 - 2,   c_val)
                        add(kuy, kuy + (Ny+1)*2 + 2,  -c_val)
                        add(kuy, kuy - (Ny+1)*2 - 2,  -c_val)
                        add(kuy, kuy - (Ny+1)*2 + 2,   c_val)
                        #kux, _ = self._dofs(ix, iy, Ny)
                        fac = 1/dy*dx*(lam + G)/G
                        if ix == 1 or ix == Nx - 1:
                            add(kuy, kux - (Ny+1)*2,      fac)
                            add(kuy, kux - (Ny+1)*2 + 2, -fac)
                            add(kuy, kux,                 -fac)
                            add(kuy, kux + 2,              fac)
                        else:
                            cf = cosa*(lam + G)/G/4
                            add(kuy, kux - (Ny+1)*2,      fac + cf)
                            add(kuy, kux - (Ny+1)*2 + 2, -fac + cf)
                            add(kuy, kux,                 -fac + cf)
                            add(kuy, kux + 2,              fac + cf)
                            add(kuy, kux - 2*(Ny+1)*2,    -cf)
                            add(kuy, kux - 2*(Ny+1)*2+2,  -cf)
                            add(kuy, kux + (Ny+1)*2,      -cf)
                            add(kuy, kux + (Ny+1)*2 + 2,  -cf)
                else:
                    add(kuy, kuy, 1)

                # ── ux equation (ix < Nx) ──────────────────────────────
                if ix < Nx:
                    # _, kuy_n = self._dofs(ix, iy, Ny)
                    #mid_ix = mid
                    r2 = dx*dx / dy/dy
                    r_lam = (lam + 2*G) / G
                    if iy == 0:
                        add(kux, kux, 1); add(kux, kux + 2, -1)
                    elif iy == Ny:
                        add(kux, kux, 1); add(kux, kux - 2, -1)
                    elif ix == 0:
                        add(kux, kux, 1)
                    elif ix == Nx - 1:
                        add(kux, kux, 1)
                    elif ix == mid:
                        # Fault column – normal stress jump condition
                        add(kux, kux,              -2*r_lam)
                        add(kux, kux + (Ny+1)*2,   r_lam)
                        add(kux, kux - (Ny+1)*2,   r_lam)
                        fac = lam/G/dy*dx
                        add(kux, kuy,                    -fac)
                        add(kux, kuy + (Ny+1)*2,          fac)
                        add(kux, kuy - 2,                 fac)
                        add(kux, kuy + (Ny+1)*2 - 2,     -fac)
                    else:
                        # Interior bulk
                        add(kux, kux, -2*r_lam - 2*r2)
                        add(kux, kux - (Ny+1)*2, r_lam)
                        add(kux, kux + (Ny+1)*2, r_lam)
                        add(kux, kux - 2, r2)
                        add(kux, kux + 2, r2)
                        c_val = cosa/dy*dx*(lam + 3*G)/G/4
                        add(kux, kux + (Ny+1)*2 - 2,   c_val)
                        add(kux, kux + (Ny+1)*2 + 2,  -c_val)
                        add(kux, kux - (Ny+1)*2 - 2,  -c_val)
                        add(kux, kux - (Ny+1)*2 + 2,   c_val)
                        fac = 1/dy*dx*(lam + G)/G
                        if iy == 1 or iy == Ny - 1:
                            add(kux, kuy + (Ny+1)*2,      fac)
                            add(kux, kuy + (Ny+1)*2 - 2, -fac)
                            add(kux, kuy,                 -fac)
                            add(kux, kuy - 2,              fac)
                        else:
                            cf = cosa/dy/dy*dx*dx*(lam + G)/G/4
                            add(kux, kuy + (Ny+1)*2,        fac + cf)
                            add(kux, kuy + (Ny+1)*2 - 2,   -fac + cf)
                            add(kux, kuy,                   -fac + cf)
                            add(kux, kuy - 2,                fac + cf)
                            add(kux, kuy + (Ny+1)*2 + 2,   -cf)
                            add(kux, kuy + (Ny+1)*2 - 4,   -cf)
                            add(kux, kuy + 2,               -cf)
                            add(kux, kuy - 4,               -cf)
                else:
                    add(kux, kux, 1)

        LH = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))

        #from scipy.io import loadmat
        #LH_mat = loadmat('LH.mat')['LH']
        #print(np.max(np.abs(LH - LH_mat)))
        return LH

    # ------------------------------------------------------------------
    def build_RH(self, dPdt: float, V: np.ndarray) -> np.ndarray:
        p, g = self.p, self.grid
        Nx, Ny, N = p.Nx, p.Ny, g.N
        dx, dy   = g.dx, g.dy
        G        = p.G
        sina, cosa = g.sina, g.cosa
        y        = g.y
        mid      = Nx // 2    # fault column 0-based

        RH = np.zeros(N)

        for ix in range(Nx+1):
            for iy in range(Ny+1):
                kux, kuy = self._dofs(ix, iy, Ny)

                # ── uy block ──
                if iy < Ny:
                    #if ix not in (0, Nx) and iy not in (0, Ny - 1):
                    if ix == 0:
                        pass
                    elif ix == Nx:
                        pass
                    elif iy == 0:
                        #RH[kuy] = 0.0
                        pass
                    elif iy == Ny - 1:
                        pass
                    elif ix == mid:
                        RH[kuy] = V[iy]
                    elif ix == mid + 1:
                        #RH[kuy] = -1 * V[iy]
                        pass  # velocity BC handled above
                    else:
                        yv = y[iy]
                        if yv == 850 and ix >= mid + 1:
                            RH[kuy] =  dPdt / dy * dx*dx / G * sina
                        if yv == 1050 and ix >= mid + 1:
                            RH[kuy] = -dPdt / dy * dx*dx / G * sina
                        if yv == 800 and ix <= mid:
                            RH[kuy] =  dPdt / dy * dx*dx / G * sina
                        if yv == 1000 and ix <= mid:
                            RH[kuy] = -dPdt / dy * dx*dx / G * sina
                        # if yv == 860 and ix >= mid + 1:
                        #     RH[kuy] =  dPdt / dy * dx*dx / G * sina
                        # if yv == 1060 and ix >= mid + 1:
                        #     RH[kuy] = -dPdt / dy * dx*dx / G * sina
                        # if yv == 820 and ix <= mid:
                        #     RH[kuy] =  dPdt / dy * dx*dx / G * sina
                        # if yv == 1020 and ix <= mid:
                        #     RH[kuy] = -dPdt / dy * dx*dx / G * sina

                # ── ux block ──
                if ix < Nx:
                    #if iy not in (0, Ny) and ix not in (0, Nx - 1):
                    if iy == 0:
                        pass
                    elif iy == Ny:
                        pass
                    elif ix == 0:
                        pass
                    elif ix == Nx - 1:
                        pass
                    elif ix == mid:
                        yv = y[iy]
                        if 800 < yv <= 850:
                            RH[kux] = -dPdt * dx / G
                        if 1000 < yv <= 1050:
                            RH[kux] =  dPdt * dx / G
                        # if 820 < yv <= 860:
                        #     RH[kux] = -dPdt * dx / G
                        # if 1020 < yv <= 1060:
                        #     RH[kux] =  dPdt * dx / G
                    else:
                        yv = y[iy]
                        if yv == 1050 and ix > mid + 1:
                            RH[kux] =  dPdt / dy * dx*dx / G * sina * cosa
                        if yv == 1000 and ix < mid + 1:
                            RH[kux] =  dPdt / dy * dx*dx / G * sina * cosa
                        if yv == 850 and ix > mid + 1:
                            RH[kux] = -dPdt / dy * dx*dx / G * sina * cosa
                        if yv == 800 and ix < mid + 1:
                            RH[kux] = -dPdt / dy * dx*dx / G * sina * cosa
                        # if yv == 1060 and ix > mid + 1:
                        #     RH[kux] =  dPdt / dy * dx*dx / G * sina * cosa
                        # if yv == 1020 and ix < mid + 1:
                        #     RH[kux] =  dPdt / dy * dx*dx / G * sina * cosa
                        # if yv == 860 and ix > mid + 1:
                        #     RH[kux] = -dPdt / dy * dx*dx / G * sina * cosa
                        # if yv == 820 and ix < mid + 1:
                        #     RH[kux] = -dPdt / dy * dx*dx / G * sina * cosa
        return RH

# ─────────────────────────────────────────────────────────────
# 8.  Output Manager
# ─────────────────────────────────────────────────────────────

class OutputManager:
    """
    Handles:
      - in-memory snapshot arrays (like MATLAB globals Um, Vm, …)
      - ASCII log file (output.txt)
      - NumPy checkpoints
    """

    def __init__(self, p: ModelParameters, output_dir: Path = Path(".")):
        self.p   = p
        self.out = output_dir
        self.out.mkdir(parents=True, exist_ok=True)
        Ny, Nx = p.Ny, p.Nx
        n = p.Nt // p.output_interval

        self.Um      = np.zeros((Ny, n))
        self.Vm      = np.zeros((Ny, n))
        self.taum    = np.zeros((Ny, n))
        self.sigmam  = np.zeros((Ny, n))
        self.Pm      = np.zeros((Ny, n))
        self.thetam  = np.zeros((Ny, n))
        self.dtm     = np.zeros(n)
        self.tm      = np.zeros(n)
        self.taumall    = np.zeros((Ny, Nx, n))
        self.sigmamall  = np.zeros((Ny-1, Nx-1, n))
        self.uymall     = np.zeros((Ny, Nx+1, n))
        self.vymall     = np.zeros((Ny, Nx+1, n))
        self.uxmall     = np.zeros((Ny+1, Nx, n))
        self.vxmall     = np.zeros((Ny+1, Nx, n))

        self._logfile = open(self.out / "output.txt", "w")

    # ------------------------------------------------------------------
    def log(self, it: int, t: float, dt: float, V: np.ndarray, U: np.ndarray,
            checkpointer: int = 0):
        yr = 365 * 24 * 3600
        line = (f"it={checkpointer+it}, t={t/yr:.6f} yr, dt={dt:.3e}, "
                f"maxV={V.max():.3e}, minV={V.min():.3e}, maxU={U.max():.6f}\n")
        self._logfile.write(line)
        self._logfile.flush()

    # ------------------------------------------------------------------
    def write_memory(self, it: int,
                     U, V, tau, sigma, P, theta, dt, t,
                     tauqs, sigmaqs, uy, vy, ux, vx):
        idx = it // self.p.output_interval - 1
        self.Um[:, idx]     = U
        self.Vm[:, idx]     = V
        self.taum[:, idx]   = tau
        self.sigmam[:, idx] = sigma
        self.Pm[:, idx]     = P
        self.thetam[:, idx] = theta
        self.dtm[idx]       = dt
        self.tm[idx]        = t
        self.taumall[:, :, idx]   = tauqs
        self.sigmamall[:, :, idx] = sigmaqs
        self.uymall[:, :, idx]    = uy
        self.vymall[:, :, idx]    = vy
        self.uxmall[:, :, idx]    = ux
        self.vxmall[:, :, idx]    = vx

    # ------------------------------------------------------------------
    def save_checkpoint(self, it: int, checkpointer: int,
                        fault: "FaultState", tauqs, sigmaqs,
                        uy, vy, ux, vx, dt: float, t: float):
        fname = self.out / f"data_{checkpointer + it}.npz"
        np.savez(fname,
                 U=fault.U, V=fault.V, tau=fault.tau, sigma=fault.sigma,
                 P=np.zeros_like(fault.U),  # placeholder; update as needed
                 theta=fault.theta, dt=dt, t=t,
                 tauqs=tauqs, sigmaqs=sigmaqs,
                 uy=uy, vy=vy, ux=ux, vx=vx)

    # ------------------------------------------------------------------
    def save_all(self):
        fname = self.out / "dataall.npz"
        np.savez(fname,
                 Um=self.Um, Vm=self.Vm, taum=self.taum,
                 sigmam=self.sigmam, Pm=self.Pm, thetam=self.thetam,
                 dtm=self.dtm, tm=self.tm)

    # ------------------------------------------------------------------
    def close(self):
        self._logfile.close()

    # ------------------------------------------------------------------
    def load_checkpoint(self, checkpointer: int) -> dict:
        fname = self.out / f"data_{checkpointer}.npz"
        return dict(np.load(fname))

# ─────────────────────────────────────────────────────────────
# 9.  Stress computation helpers
# ─────────────────────────────────────────────────────────────

def _movmean_discard(arr: np.ndarray, axis: int) -> np.ndarray:
    """Running mean of adjacent pairs along *axis*, discarding endpoints."""
    sl_a = [slice(None)] * arr.ndim
    sl_b = [slice(None)] * arr.ndim
    sl_a[axis] = slice(None, -1)
    sl_b[axis] = slice(1, None)
    return (arr[tuple(sl_a)] + arr[tuple(sl_b)]) / 2

def compute_stress_fields(uy, ux, dx, dy, lam, G, cosa, sina, Ny, Nx):
    """
    Compute tauqs (Ny × Nx) and sigmaqs (Ny-1 × Nx-1) from displacement fields.

    Grid shapes (matching MATLAB staggered layout):
        uy  : (Ny,   Nx+1)   – y-displacement on uy-nodes
        ux  : (Ny+1, Nx)     – x-displacement on ux-nodes

    MATLAB equivalents:
        tauqs   = G/sina*(diff(uy,1,2)/dx
                          + (1-2*cosa²)*diff(ux,1,1)/dy
                          + cosa*(movmean(duxdx,2,1,'discard')
                                  - movmean(duydy,2,2,'discard')))
        sigmaqs = (λ+2G)*diff(ux[2:Ny,:],1,2)/dx
                  + λ*diff(uy[:,2:Nx],1,1)/dy
                  - 2G*cosa*movmean(movmean(diff(ux,1,1)/dy,2,2,'discard'),2,1,'discard')
    """
    # ── Term 1: diff(uy, axis=1) / dx  →  shape (Ny, Nx) ──────────────
    # uy is (Ny, Nx+1), diff along columns → (Ny, Nx)  ✓
    term1 = np.diff(uy, axis=1) / dx                    # (Ny, Nx)

    # ── Term 2: (1-2cos²α) * diff(ux, axis=0) / dy  →  shape (Ny, Nx) ─
    # ux is (Ny+1, Nx), diff along rows → (Ny, Nx)  ✓
    term2 = (1 - 2 * cosa**2) * np.diff(ux, axis=0) / dy   # (Ny, Nx)

    # ── Term 3a: movmean(duxdx, 2, axis=0, 'discard')  →  (Ny, Nx) ────
    # duxdx = gradient(ux, dx, axis=1): ux is (Ny+1, Nx) → duxdx (Ny+1, Nx)
    # movmean of adjacent pairs along axis=0 discarding endpoints → (Ny, Nx)
    duxdx = np.gradient(ux, dx, axis=1)                 # (Ny+1, Nx)
    mm_duxdx = _movmean_discard(duxdx, axis=0)          # (Ny,   Nx)  ✓

    # ── Term 3b: movmean(duydy, 2, axis=1, 'discard')  →  (Ny, Nx) ────
    # duydy = gradient(uy, dy, axis=0): uy is (Ny, Nx+1) → duydy (Ny, Nx+1)
    # movmean of adjacent pairs along axis=1 discarding endpoints → (Ny, Nx)
    duydy = np.gradient(uy, dy, axis=0)                 # (Ny, Nx+1)
    mm_duydy = _movmean_discard(duydy, axis=1)          # (Ny, Nx)    ✓

    # ── Assemble tauqs ──────────────────────────────────────────────────
    tauqs = G / sina * (term1 + term2 + cosa * (mm_duxdx - mm_duydy))  # (Ny, Nx)

    # Interpolate across the fault column (fault sits at mid)
    mid = Nx // 2    # 0-based centre column index
    tauqs[:, mid] = (tauqs[:, mid - 1] + tauqs[:, mid + 1]) / 2

    # ══ sigmaqs  (Ny-1, Nx-1) ═══════════════════════════════════════════
    # MATLAB: diff(ux(2:Ny,:),1,2)/dx
    #   ux(2:Ny,:) in 1-based = ux[1:Ny, :] in 0-based → shape (Ny-1, Nx)
    #   diff along columns → (Ny-1, Nx-1)  ✓
    s_term1 = np.diff(ux[1:Ny, :], axis=1) / dx         # (Ny-1, Nx-1)

    # MATLAB: diff(uy(:,2:Nx),1,1)/dy
    #   uy(:,2:Nx) in 1-based = uy[:, 1:Nx] in 0-based → shape (Ny, Nx-1)
    #   diff along rows → (Ny-1, Nx-1)  ✓
    s_term2 = np.diff(uy[:, 1:Nx], axis=0) / dy         # (Ny-1, Nx-1)

    # MATLAB: movmean(movmean(diff(ux,1,1)/dy, 2,2,'discard'), 2,1,'discard')
    #   diff(ux,1,1)/dy: ux (Ny+1,Nx), diff rows → (Ny, Nx)
    #   inner movmean along axis=1 'discard' → (Ny, Nx-1)
    #   outer movmean along axis=0 'discard' → (Ny-1, Nx-1)  ✓
    dux_dy     = np.diff(ux, axis=0) / dy                # (Ny,   Nx)
    mm_inner   = _movmean_discard(dux_dy, axis=1)        # (Ny,   Nx-1)
    mm_outer   = _movmean_discard(mm_inner, axis=0)      # (Ny-1, Nx-1)

    sigmaqs = ((lam + 2*G) * s_term1
               + lam       * s_term2
               - 2*G*cosa  * mm_outer)                   # (Ny-1, Nx-1)

    return tauqs, sigmaqs

def log_bisection(func,
                  lo,
                  hi,
                  tol_log=1e-14,
                  tol_f=1e-12,
                  maxiter=200):
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

def bisection(f,
              lb,
              ub,
              target=0.0,
              tolX=1e-6,
              tolFun=0.0,
              maxiter=1000):

    # shift function by target
    def g(x):
        return f(x) - target

    flb = g(lb)
    fub = g(ub)

    if flb == 0:
        return lb, target, 3

    if fub == 0:
        return ub, target, 3

    # root must be bracketed
    if flb * fub > 0:
        return np.nan, np.nan, -2

    #for _ in range(maxiter):

    x = 0.5 * (lb + ub)

    fx = g(x)

    outsideTolX = abs(ub - x) > tolX
    outsideTolFun = abs(fx) > tolFun

    # convergence
    if (not outsideTolX) and (not outsideTolFun):
        return x, fx + target, 3

    if not outsideTolX:
        return x, fx + target, 1

    if not outsideTolFun:
        return x, fx + target, 2

    # keep bracket
    if np.sign(fx) != np.sign(fub):
        lb = x
        flb = fx
    else:
        ub = x
        fub = fx

    return x, fx + target, -1


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
# ─────────────────────────────────────────────────────────────
# 10.  Top-level Model Driver
# ─────────────────────────────────────────────────────────────

class FaultSlipModel:
    """
    Top-level driver.  Instantiate with a ModelParameters object (or use
    defaults), then call .run().

    Example
    -------
        params = ModelParameters(Nx=51, Ny=51, Nt=200)
        model  = FaultSlipModel(params)
        model.run()
    """

    def __init__(self, params: Optional[ModelParameters] = None,
                 output_dir: str = "output",
                 checkpointer: int = 0):
        self.p            = params or ModelParameters()
        self.checkpointer = checkpointer
        self.output       = OutputManager(self.p, Path(output_dir))

        # Build grid
        self.grid  = Grid(self.p)
        # Friction profile
        self.fric  = FrictionalZones(self.p, self.grid.y)
        # Initial stress
        self.stress = StressState(self.p, self.grid.y)
        # Fault state
        self.fault  = FaultState(self.p, self.stress, self.fric)
        # ksi for adaptive dt
        self.ksi    = build_ksi(self.p, self.fric, self.stress.sigman0, self.grid.dy)

        # Displacement / velocity fields
        p  = self.p
        Nx, Ny = p.Nx, p.Ny
        self.ux = np.zeros((Ny + 1, Nx))
        self.uy = np.zeros((Ny, Nx + 1))
        self.vx = np.zeros((Ny + 1, Nx))
        self.vy = np.zeros((Ny, Nx + 1))
        self.tauqs   = np.zeros((Ny, Nx))
        self.sigmaqs = np.zeros((Ny - 1, Nx - 1))

    # ------------------------------------------------------------------
    def _build_and_factor_LH(self, dPdt: float):
        builder = MatrixBuilder(self.p, self.grid)
        LH = builder.build_LH()
        #print(LH)
        self.RH_builder = builder
        self.dPdt = dPdt
        self._solve = factorized(LH.tocsc())   # sparse LU decomposition

    # ------------------------------------------------------------------
    def run(self):
        t0_wall = time.perf_counter()
        p = self.p
        Nx, Ny = p.Nx, p.Ny
        yr = 365 * 24 * 3600

        # ── initialise / load checkpoint ──
        if not self.checkpointer:
            dPdt = p.dPdt_pre
            self._build_and_factor_LH(dPdt)
        else:
            ckpt = self.output.load_checkpoint(self.checkpointer)
            self.fault.U     = ckpt["U"]
            self.fault.V     = ckpt["V"]
            self.fault.tau   = ckpt["tau"]
            self.fault.sigma = ckpt["sigma"]
            self.fault.theta = ckpt["theta"]
            self.tauqs   = ckpt["tauqs"]
            self.sigmaqs = ckpt["sigmaqs"]
            self.uy = ckpt["uy"];  self.vy = ckpt["vy"]
            self.ux = ckpt["ux"];  self.vx = ckpt["vx"]
            dPdt = p.dPdt_pre
            self._build_and_factor_LH(dPdt)

        dt     = p.dt_init
        dt_max = p.dt_max
        t      = 0.0
        t2     = 0.0
        phase  = 0        # 0 = pre-depletion, 1 = transition, 2 = post-depletion

        print(f"Setup complete in {time.perf_counter()-t0_wall:.1f}s.  "
              f"Starting {p.Nt} time steps …")

        # ── time loop ────────────────────────────────────────────────
        for it in range(1, p.Nt + 1):

            # Phase transition: pre → post depletion
            if phase == 1:
                dPdt  = p.dPdt_post
                self._build_and_factor_LH(dPdt)
                dt    = p.dt_init
                dt_max = p.dt_max
                t2    = 0.0
                phase = 2

            # ── velocity solve (rate-and-state) ──
            mid = Nx // 2
            self.fault.solve_slip_rate(self.tauqs[:, mid],
                                       self.stress, self.fric)
            
            #print(self.fault.V)

            # ── adaptive time step ──
            V_inner = self.fault.V[1: Ny - 1]
            ksi_inner = self.ksi[1: Ny - 1]
            dt_cand = np.min(ksi_inner * p.L / V_inner)
            dt_cand = max(dt_cand, 1e-150)
            dt      = min(min(1.2 * dt, dt_cand), dt_max)

            # Clamp dt so we hit tload exactly
            if phase == 0 and t + dt >= p.tload:
                dt    = p.tload - t
                phase = 1

            # ── aging law + fault advance ──
            self.fault.advance(dt, self.tauqs[:, mid], self.stress)

            # ── update RH with current slip velocities ──
            RH = self.RH_builder.build_RH(dPdt, self.fault.V)
            # Inject velocity BC at fault column
            fault_rows = (np.arange(1, Ny - 1) + (Nx // 2) * (Ny + 1)) * 2 + 1
            RH[fault_rows] = self.fault.V[1: Ny - 1]
            #print(RH[fault_rows])

            # ── elastic solve ──
            S   = self._solve(RH)
            #vpx = S[0::2].reshape(Ny + 1, Nx + 1)
            #vpy = S[1::2].reshape(Ny + 1, Nx + 1)
            vpx = np.reshape(S[0::2], (p.Nx+1, p.Ny+1), order='C').T
            vpy = np.reshape(S[1::2], (p.Ny+1, p.Nx+1), order='C').T
            self.vy = vpy[:Ny, :]
            self.vx = vpx[:, :Nx]

            # ── integrate displacements ──
            self.uy += self.vy * dt
            self.ux += self.vx * dt

            # ── compute stress ──
            self.tauqs, self.sigmaqs = compute_stress_fields(
                self.uy, self.ux, self.grid.dx, self.grid.dy,
                p.lam, p.G, self.grid.cosa, self.grid.sina, Ny, Nx)

            # Update effective normal stress from sigmaqs
            mid_l = (Nx - 1) // 2 - 1
            mid_r = (Nx - 1) // 2
            sigmal = np.concatenate([[self.sigmaqs[0, mid_l]],
                                     _movmean_discard(self.sigmaqs[:, mid_l], 0),
                                     [self.sigmaqs[-1, mid_l]]])
            sigmar = np.concatenate([[self.sigmaqs[0, mid_r]],
                                     _movmean_discard(self.sigmaqs[:, mid_r], 0),
                                     [self.sigmaqs[-1, mid_r]]])
            self.fault.sigma = self.stress.sigman0 - np.minimum(sigmal, sigmar)
            #self.fault.sigma = self.stress.sigman0 - 0.5 * (sigmal + sigmar)

            # ── pressure update ──
            self.stress.update_pressure(dt, dPdt)

            # ── logging ──
            self.output.log(it, t2 if phase == 2 else t, dt,
                            self.fault.V, self.fault.U, self.checkpointer)

            if it % p.output_interval == 0:
                self.output.write_memory(
                    it, self.fault.U, self.fault.V, self.fault.tau,
                    self.fault.sigma, self.stress.P, self.fault.theta,
                    dt, t, self.tauqs, self.sigmaqs,
                    self.uy, self.vy, self.ux, self.vx)

            if it % p.checkpoint_interval == 0:
                self.output.save_checkpoint(
                    it, self.checkpointer, self.fault,
                    self.tauqs, self.sigmaqs,
                    self.uy, self.vy, self.ux, self.vx, dt, t)
                self.output.save_all()
                print(f"  Checkpoint it={it}, elapsed {time.perf_counter()-t0_wall:.1f}s")

            t += dt
            if phase == 2:
                t2 += dt

            # print(t)
            # print(t2)

        # ── wrap up ──
        self.output.save_all()
        self.output.close()
        print(f"Done.  Total wall time: {time.perf_counter()-t0_wall:.1f}s")

    # ------------------------------------------------------------------
    def plot_results_v0(self):
        """Quick diagnostic plots after run()."""
        yr    = 365 * 24 * 3600
        om    = self.output
        tm_yr = om.tm / yr

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Slip velocity
        ax = axes[0, 0]
        ax.contourf(tm_yr, self.grid.y, np.log10(np.abs(om.Vm) + 1e-40))
        ax.set_xlabel("Time [yr]");  ax.set_ylabel("Depth [m]")
        ax.set_title("log₁₀ Slip velocity [m/s]")

        # Cumulative slip
        ax = axes[0, 1]
        ax.contourf(tm_yr, self.grid.y, om.Um)
        ax.set_xlabel("Time [yr]");  ax.set_ylabel("Depth [m]")
        ax.set_title("Cumulative slip U [m]")

        # Shear stress
        ax = axes[1, 0]
        ax.contourf(tm_yr, self.grid.y, om.taum / 1e6)
        ax.set_xlabel("Time [yr]");ax.set_ylabel("Depth [m]")
        ax.set_title("Shear stress τ [MPa]")

        # Normal stress
        ax = axes[1, 1]
        ax.contourf(tm_yr, self.grid.y, om.sigmam / 1e6)
        ax.set_xlabel("Time [yr]");  ax.set_ylabel("Depth [m]")
        ax.set_title("Normal stress σ [MPa]")

        plt.tight_layout()
        fig.savefig(self.output.out / "results.png", dpi=150)
        plt.show()
        return fig

    def plot_results(self, it: int = -1):
        """
        Enhanced plotting:
        - τ/σ vs depth
        - 2D fields (ux, uy, vx, vy, tauqs, sigmaqs)
        
        Parameters
        ----------
        it : int
            Time index (default = last snapshot)
        """

        om = self.output
        grid = self.grid
        yr = 365 * 24 * 3600

        if it < 0:
            it = om.Um.shape[1] - 1  # last stored snapshot

        # ─────────────────────────────────────────────
        # 1. Ratio τ/σ vs depth
        # ─────────────────────────────────────────────
        tau = om.taum[:, it]
        sigma = om.sigmam[:, it]
        ratio = tau / (sigma + 1e-12)

        plt.figure(figsize=(8, 5))
        plt.plot(grid.y+2000, ratio, 'o-', lw=1)
        #plt.gca().invert_yaxis()
        plt.ylabel(r"$\tau / \sigma_n$")
        plt.xlabel("Depth [m]")
        #plt.ylim(0.3, 0.35)
        #plt.xlim(2000, 4000)
        plt.title("Ratio shear / normal stress")
        plt.grid(True)
        plt.tight_layout()

        plt.figure(figsize=(8, 5))
        plt.plot(grid.y+2000, om.taum[:, it] / 1e6, 'o-', lw=1)
        plt.ylabel(r"Shear stress $\tau$ [MPa]")
        plt.xlabel("Depth [m]")
        plt.grid(True)
        plt.tight_layout()

        plt.figure(figsize=(8, 5))
        plt.plot(grid.y+2000, om.sigmam[:, it] / 1e6, 'o-', lw=1)
        plt.ylabel(r"Normal stress $\sigma_n$ [MPa]")
        plt.xlabel("Depth [m]")
        plt.grid(True)
        plt.tight_layout()

        plt.figure(figsize=(8, 5))
        #plt.plot(grid.y+2000, om.Vm, 'o-', lw=1)
        plt.plot(grid.y+2000, om.Vm[:, it], 'o-', lw=1)
        plt.ylabel(r"Slip velocity $V$ [m/s]")
        plt.xlabel("Depth [m]")
        plt.grid(True)
        plt.tight_layout()

        # ─────────────────────────────────────────────
        # 2. Extract 2D fields
        # ─────────────────────────────────────────────
        ux = om.uxmall[:, :, it]
        uy = om.uymall[:, :, it]
        vx = om.vxmall[:, :, it]
        vy = om.vymall[:, :, it]
        tauqs = om.taumall[:, :, it]
        sigmaqs = om.sigmamall[:, :, it]

        # ─────────────────────────────────────────────
        # 3. 2D plots (match your layout)
        # ─────────────────────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # --- ux / vx ---
        im = axes[0, 0].pcolormesh(
            grid.Xux, grid.Yux, vx,
            shading='auto'
        )
        axes[0, 0].set_title("Vx")
        fig.colorbar(im, ax=axes[0, 0])

        # --- tauqs ---
        im = axes[0, 1].pcolormesh(
            grid.Xtau, grid.Ytau, tauqs,
            shading='auto'
        )
        axes[0, 1].set_title(r"Shear stress $\tau_{qs}$")
        fig.colorbar(im, ax=axes[0, 1])

        # --- uy ---
        im = axes[1, 0].pcolormesh(
            grid.Xuy, grid.Yuy, vy,
            shading='auto'
        )
        axes[1, 0].set_title("Vy")
        fig.colorbar(im, ax=axes[1, 0])

        # --- sigmaqs ---
        im = axes[1, 1].pcolormesh(
            grid.Xsigma, grid.Ysigma, sigmaqs,
            shading='auto'
        )
        axes[1, 1].set_title(r"Normal stress $\sigma_{qs}$")
        fig.colorbar(im, ax=axes[1, 1])

        for ax in axes.flat:
            ax.set_aspect('equal')
            ax.invert_yaxis()

        plt.tight_layout()

        plt.show()

        #return fig

def test_rigid_translation():

    """
    Benchmark 1:
    rigid body translation

    ux = constant
    uy = constant

    Expected:
        tauqs   = 0
        sigmaqs = 0
    """

    # --------------------------------------------------
    # 1. Build minimal model
    # --------------------------------------------------

    params = ModelParameters(
        Nx=51,
        Ny=51
    )

    grid = Grid(params)

    # --------------------------------------------------
    # 2. Constant displacement field
    # --------------------------------------------------

    ux_const = 1.2345
    uy_const = -2.3456

    # ux shape = (Ny+1, Nx)
    ux = np.ones((params.Ny + 1, params.Nx)) * ux_const

    # uy shape = (Ny, Nx+1)
    uy = np.ones((params.Ny, params.Nx + 1)) * uy_const

    # --------------------------------------------------
    # 3. Compute stresses
    # --------------------------------------------------

    tauqs, sigmaqs = compute_stress_fields(
        uy=uy,
        ux=ux,
        dx=grid.dx,
        dy=grid.dy,
        lam=params.lam,
        G=params.G,
        cosa=grid.cosa,
        sina=grid.sina,
        Ny=params.Ny,
        Nx=params.Nx
    )

    # --------------------------------------------------
    # 4. Compute errors
    # --------------------------------------------------

    max_tau = np.max(np.abs(tauqs))
    max_sigma = np.max(np.abs(sigmaqs))

    print("\n========== RIGID TRANSLATION TEST ==========")

    print(f"max |tauqs|   = {max_tau:.3e}")
    print(f"max |sigmaqs| = {max_sigma:.3e}")

    # --------------------------------------------------
    # 5. Pass/fail
    # --------------------------------------------------

    tol = 1e-12

    if max_tau < tol and max_sigma < tol:
        print("PASS")
    else:
        print("FAIL")

    # --------------------------------------------------
    # 6. Optional visualization
    # --------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axes[0].imshow(tauqs)
    axes[0].set_title("tauqs")

    im1 = axes[1].imshow(sigmaqs)
    axes[1].set_title("sigmaqs")

    plt.colorbar(im0, ax=axes[0])
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.show()

def test_rigid_rotation():

    """
    Benchmark 2:
    rigid body rotation

    ux = -omega * y
    uy =  omega * x

    Expected:
        tauqs   = 0
        sigmaqs = 0
    """

    # --------------------------------------------------
    # 1. Build model/grid
    # --------------------------------------------------

    params = ModelParameters(
        Nx=51,
        Ny=51
    )

    grid = Grid(params)

    omega = 1e-6

    # --------------------------------------------------
    # 2. Build coordinate arrays
    # --------------------------------------------------

    # ux nodes: shape (Ny+1, Nx)
    Xux = grid.Xux
    Yux = grid.Yux

    # uy nodes: shape (Ny, Nx+1)
    Xuy = grid.Xuy
    Yuy = grid.Yuy

    # --------------------------------------------------
    # 3. Define rigid rotation field
    # --------------------------------------------------

    ux = -omega * Yux
    uy =  omega * Xuy

    # --------------------------------------------------
    # 4. Compute stresses
    # --------------------------------------------------

    tauqs, sigmaqs = compute_stress_fields(
        uy=uy,
        ux=ux,
        dx=grid.dx,
        dy=grid.dy,
        lam=params.lam,
        G=params.G,
        cosa=grid.cosa,
        sina=grid.sina,
        Ny=params.Ny,
        Nx=params.Nx
    )

    # --------------------------------------------------
    # 5. Errors
    # --------------------------------------------------

    max_tau = np.max(np.abs(tauqs))
    max_sigma = np.max(np.abs(sigmaqs))

    print("\n========== RIGID ROTATION TEST ==========")

    print(f"max |tauqs|   = {max_tau:.3e}")
    print(f"max |sigmaqs| = {max_sigma:.3e}")

    tol = 1e-10

    if max_tau < tol and max_sigma < tol:
        print("PASS")
    else:
        print("FAIL")

    # --------------------------------------------------
    # 6. Visualization
    # --------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axes[0].imshow(tauqs)
    axes[0].set_title("tauqs")

    im1 = axes[1].imshow(sigmaqs)
    axes[1].set_title("sigmaqs")

    plt.colorbar(im0, ax=axes[0])
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.show()

def test_uniaxial_extension():

    """
    Benchmark 3:
    uniaxial extension

    ux = a * x
    uy = 0

    Expected:
        tauqs ≈ 0
        sigmaqs = constant
    """

    # --------------------------------------------------
    # 1. Build model/grid
    # --------------------------------------------------

    params = ModelParameters(
        Nx=51,
        Ny=51
    )

    grid = Grid(params)

    # prescribed strain
    a = 1e-6

    # --------------------------------------------------
    # 2. Coordinates
    # --------------------------------------------------

    Xux = grid.Xux
    Xuy = grid.Xuy

    # --------------------------------------------------
    # 3. Displacement field
    # --------------------------------------------------

    # ux shape = (Ny+1, Nx)
    ux = a * Xux

    # uy shape = (Ny, Nx+1)
    uy = np.zeros_like(Xuy)

    # --------------------------------------------------
    # 4. Compute stresses
    # --------------------------------------------------

    tauqs, sigmaqs = compute_stress_fields(
        uy=uy,
        ux=ux,
        dx=grid.dx,
        dy=grid.dy,
        lam=params.lam,
        G=params.G,
        cosa=grid.cosa,
        sina=grid.sina,
        Ny=params.Ny,
        Nx=params.Nx
    )

    # --------------------------------------------------
    # 5. Diagnostics
    # --------------------------------------------------

    max_tau = np.max(np.abs(tauqs))

    sigma_mean = np.mean(sigmaqs)
    sigma_std = np.std(sigmaqs)

    print("\n========== UNIAXIAL EXTENSION TEST ==========")

    print(f"max |tauqs|      = {max_tau:.3e}")

    print(f"mean(sigmaqs)    = {sigma_mean:.3e}")

    print(f"std(sigmaqs)     = {sigma_std:.3e}")

    # --------------------------------------------------
    # 6. Pass / fail
    # --------------------------------------------------

    tau_tol = 1e-10
    sigma_tol = 1e-10

    if max_tau < tau_tol and sigma_std < sigma_tol:
        print("PASS")
    else:
        print("FAIL")

    # --------------------------------------------------
    # 7. Visualization
    # --------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axes[0].imshow(tauqs)
    axes[0].set_title("tauqs")

    im1 = axes[1].imshow(sigmaqs)
    axes[1].set_title("sigmaqs")

    plt.colorbar(im0, ax=axes[0])
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.show()

def test_constant_strain_equilibrium():

    """
    Benchmark 4:
    Constant strain equilibrium test

    ux = a * x
    uy = 0

    Since stress is constant:

        div(sigma) = 0

    therefore:

        LH @ U = 0

    should hold to machine precision.
    """

    # --------------------------------------------------
    # 1. Build model/grid/matrix
    # --------------------------------------------------

    params = ModelParameters(
        Nx=51,
        Ny=51
    )

    grid = Grid(params)

    builder = MatrixBuilder(params, grid)

    LH = builder.build_LH()

    # --------------------------------------------------
    # 2. Prescribed displacement field
    # --------------------------------------------------

    a = 1e-6

    # staggered fields
    ux = a * grid.Xux
    uy = np.zeros_like(grid.Xuy)

    # --------------------------------------------------
    # 3. Pack into global vector U
    # --------------------------------------------------

    U = np.zeros(grid.N)

    for ix in range(params.Nx + 1):
        for iy in range(params.Ny + 1):

            kux, kuy = builder._dofs(ix, iy, params.Ny)

            # ux nodes exist for ix < Nx
            if ix < params.Nx:
                U[kux] = ux[iy, ix]

            # uy nodes exist for iy < Ny
            if iy < params.Ny:
                U[kuy] = uy[iy, ix]

    # --------------------------------------------------
    # 4. RHS = 0
    # --------------------------------------------------

    RH = np.zeros(grid.N)

    # --------------------------------------------------
    # 5. Residual
    # --------------------------------------------------

    residual = LH @ U - RH

    max_residual = np.max(np.abs(residual))
    rms_residual = np.sqrt(np.mean(residual**2))

    print("\n========== CONSTANT STRAIN EQUILIBRIUM TEST ==========")

    print(f"max residual = {max_residual:.3e}")
    print(f"rms residual = {rms_residual:.3e}")

    # --------------------------------------------------
    # 6. Pass/fail
    # --------------------------------------------------

    tol = 1e-10

    if max_residual < tol:
        print("PASS")
    else:
        print("FAIL")

    # --------------------------------------------------
    # 7. Visualize residual
    # --------------------------------------------------

    Rux = residual[0::2].reshape(params.Ny + 1, params.Nx + 1)
    Ruy = residual[1::2].reshape(params.Ny + 1, params.Nx + 1)

    # interior only
    interior_Rux = Rux[2:-2, 2:-2]
    interior_Ruy = Ruy[2:-2, 2:-2]

    print()
    print("INTERIOR RESIDUAL")
    print(np.max(np.abs(interior_Rux)))
    print(np.max(np.abs(interior_Ruy)))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axes[0].imshow(Rux)
    axes[0].set_title("Residual ux")

    im1 = axes[1].imshow(Ruy)
    axes[1].set_title("Residual uy")

    plt.colorbar(im0, ax=axes[0])
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.show()

def test_pure_shear():

    """
    Benchmark 5:
    pure shear test

    ux = gamma * y
    uy = 0

    Expected:
        tauqs   = constant
        sigmaqs = 0
    """

    # --------------------------------------------------
    # 1. Build model/grid
    # --------------------------------------------------

    params = ModelParameters(
        Nx=51,
        Ny=51
    )

    grid = Grid(params)

    gamma = 1e-6

    # --------------------------------------------------
    # 2. Coordinates
    # --------------------------------------------------

    Yux = grid.Yux
    Xuy = grid.Xuy

    # --------------------------------------------------
    # 3. Displacement field
    # --------------------------------------------------

    # ux shape = (Ny+1, Nx)
    ux = gamma * Yux

    # uy shape = (Ny, Nx+1)
    uy = np.zeros_like(Xuy)

    # --------------------------------------------------
    # 4. Compute stresses
    # --------------------------------------------------

    tauqs, sigmaqs = compute_stress_fields(
        uy=uy,
        ux=ux,
        dx=grid.dx,
        dy=grid.dy,
        lam=params.lam,
        G=params.G,
        cosa=grid.cosa,
        sina=grid.sina,
        Ny=params.Ny,
        Nx=params.Nx
    )

    # --------------------------------------------------
    # 5. Diagnostics
    # --------------------------------------------------

    tau_mean = np.mean(tauqs)
    tau_std  = np.std(tauqs)

    max_sigma = np.max(np.abs(sigmaqs))

    print("\n========== PURE SHEAR TEST ==========")

    print(f"mean(tauqs)      = {tau_mean:.3e}")

    print(f"std(tauqs)       = {tau_std:.3e}")

    print(f"max |sigmaqs|    = {max_sigma:.3e}")

    np.set_printoptions(precision=12, suppress=True)
    print(np.unique(np.round(tauqs, 12)))

    # --------------------------------------------------
    # 6. Pass / fail
    # --------------------------------------------------

    tau_tol = 1e-10
    sigma_tol = 1e-10

    if tau_std < tau_tol and max_sigma < sigma_tol:
        print("PASS")
    else:
        print("FAIL")

    # --------------------------------------------------
    # 7. Visualization
    # --------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axes[0].imshow(tauqs)
    axes[0].set_title("tauqs")

    im1 = axes[1].imshow(sigmaqs)
    axes[1].set_title("sigmaqs")

    plt.colorbar(im0, ax=axes[0])
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.show()

def test_fault_slip_symmetry():

    """
    Benchmark 6:
    smooth anti-symmetric fault slip

    ux = 0.5*D*tanh(x/w)
    uy = 0

    Expected:
        - tau localized near fault
        - left/right symmetry
        - sigma ≈ 0
        - no checkerboard
    """

    # --------------------------------------------------
    # 1. Build model/grid
    # --------------------------------------------------

    params = ModelParameters(
        Nx=101,
        Ny=101
    )

    grid = Grid(params)

    # --------------------------------------------------
    # 2. Slip parameters
    # --------------------------------------------------

    D = 1e-3

    w = 3 * grid.dx

    # --------------------------------------------------
    # 3. Coordinates
    # --------------------------------------------------

    Xux = grid.Xux
    Xuy = grid.Xuy

    # --------------------------------------------------
    # 4. Smooth fault slip field
    # --------------------------------------------------

    #This is compression
    #ux = 0.5 * D * np.tanh(Xux / w)
    #uy = np.zeros_like(Xuy)

    ux = np.zeros_like(Xux)
    uy = 0.5 * D * np.tanh(Xuy / w)

    # --------------------------------------------------
    # 5. Compute stresses
    # --------------------------------------------------

    tauqs, sigmaqs = compute_stress_fields(
        uy=uy,
        ux=ux,
        dx=grid.dx,
        dy=grid.dy,
        lam=params.lam,
        G=params.G,
        cosa=grid.cosa,
        sina=grid.sina,
        Ny=params.Ny,
        Nx=params.Nx
    )

    # --------------------------------------------------
    # 6. Symmetry diagnostics
    # --------------------------------------------------

    mid = params.Nx // 2

    tau_left  = tauqs[:, :mid]
    tau_right = np.flip(tauqs[:, mid+1:], axis=1)

    symmetry_error = np.max(np.abs(tau_left - tau_right))

    max_sigma = np.max(np.abs(sigmaqs))

    print("\n========== FAULT SLIP SYMMETRY TEST ==========")

    print(f"max symmetry error = {symmetry_error:.3e}")

    print(f"max |sigmaqs|      = {max_sigma:.3e}")

    # --------------------------------------------------
    # 7. Pass/fail
    # --------------------------------------------------

    tol = 1e-10

    if symmetry_error < tol:
        print("PASS")
    else:
        print("FAIL")

    # --------------------------------------------------
    # 8. Visualization
    # --------------------------------------------------

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    im0 = axes[0,0].imshow(ux)
    axes[0,0].set_title("ux")

    im1 = axes[0,1].imshow(uy)
    axes[0,1].set_title("uy")

    im2 = axes[1,0].imshow(tauqs)
    axes[1,0].set_title("tauqs")

    im3 = axes[1,1].imshow(sigmaqs)
    axes[1,1].set_title("sigmaqs")

    plt.colorbar(im0, ax=axes[0,0])
    plt.colorbar(im1, ax=axes[0,1])
    plt.colorbar(im2, ax=axes[1,0])
    plt.colorbar(im3, ax=axes[1,1])

    plt.tight_layout()
    plt.show()

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

    print("\n========== RADIATION DAMPING TEST ==========")

    print(f"max error       = {max_error:.3e}")

    print(f"monotonic decay = {monotonic}")

    # --------------------------------------------------
    # 6. Pass/fail
    # --------------------------------------------------

    if max_error < 1e-6 and monotonic:
        print("PASS")
    else:
        print("FAIL")

    # --------------------------------------------------
    # 7. Plot
    # --------------------------------------------------

    plt.figure(figsize=(6,4))

    plt.plot(t, U, label="numerical")

    plt.plot(t, U_exact, "--", label="exact")

    plt.xlabel("time")

    plt.ylabel("U")

    plt.title("Radiation Damping Decay")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.show()

    # --------------------------------------------------
    # 8. Energy decay
    # --------------------------------------------------

    energy = 0.5 * k * U**2

    plt.figure(figsize=(6,4))

    plt.semilogy(t, energy)

    plt.xlabel("time")

    plt.ylabel("energy")

    plt.title("Energy Dissipation")

    plt.grid(True)

    plt.tight_layout()

    plt.show()

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

    print("\n========== RATE-STATE STEADY-STATE TEST ==========")

    print(f"max error = {error:.3e}")

    # --------------------------------------------------
    # 5. Pass/fail
    # --------------------------------------------------

    if error < 1e-12:
        print("PASS")
    else:
        print("FAIL")

    # --------------------------------------------------
    # 6. Plot
    # --------------------------------------------------

    plt.figure(figsize=(6,4))

    plt.semilogx(V_values, f_numerical, label="numerical")

    plt.semilogx(V_values, f_exact, "--", label="exact")

    plt.xlabel("Slip velocity V")

    plt.ylabel("Steady-state friction")

    plt.title("Rate-State Steady State")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.show()

def test_rate_state_velocity_step():

    """
    Benchmark 9:
    rate-state velocity-step test

    Verify:

        1. direct effect
        2. state evolution
        3. steady-state relaxation
    """

    # --------------------------------------------------
    # 1. Parameters
    # --------------------------------------------------

    a = 0.01
    b = 0.015

    f0 = 0.6

    V0 = 1e-6

    L = 1e-5

    # velocity step
    V1 = 1e-6
    V2 = 1e-5

    # --------------------------------------------------
    # 2. Time setup
    # --------------------------------------------------

    dt = 0.01

    Nt = 4000

    t = np.arange(Nt) * dt

    t_step = 10.0

    # --------------------------------------------------
    # 3. Velocity history
    # --------------------------------------------------

    V = np.ones(Nt) * V1

    V[t >= t_step] = V2

    # --------------------------------------------------
    # 4. State evolution
    # --------------------------------------------------

    theta = np.zeros(Nt)

    # initial steady state
    theta[0] = L / V1

    # aging law integration
    for n in range(Nt - 1):

        dtheta = 1 - V[n] * theta[n] / L

        theta[n+1] = theta[n] + dt * dtheta

    # --------------------------------------------------
    # 5. Friction evolution
    # --------------------------------------------------

    f = (
        f0
        + a * np.log(V / V0)
        + b * np.log(V0 * theta / L)
    )

    # --------------------------------------------------
    # 6. Theoretical predictions
    # --------------------------------------------------

    # direct effect
    direct_theory = a * np.log(V2 / V1)

    # steady-state change
    steady_theory = (a - b) * np.log(V2 / V1)

    # measured values
    i_before = np.where(t < t_step)[0][-1]

    i_after = np.where(t >= t_step)[0][0]

    direct_numerical = f[i_after] - f[i_before]

    steady_numerical = f[-1] - f[0]

    # --------------------------------------------------
    # 7. Errors
    # --------------------------------------------------

    direct_error = abs(direct_numerical - direct_theory)

    steady_error = abs(steady_numerical - steady_theory)

    print("\n========== VELOCITY STEP TEST ==========")

    print(f"direct effect error      = {direct_error:.3e}")

    print(f"steady-state error       = {steady_error:.3e}")

    print()

    print(f"theoretical direct jump  = {direct_theory:.3e}")

    print(f"numerical direct jump    = {direct_numerical:.3e}")

    print()

    print(f"theoretical steady state = {steady_theory:.3e}")

    print(f"numerical steady state   = {steady_numerical:.3e}")

    # --------------------------------------------------
    # 8. Pass/fail
    # --------------------------------------------------

    tol = 1e-4

    if direct_error < tol and steady_error < tol:
        print("PASS")
    else:
        print("FAIL")

    # --------------------------------------------------
    # 9. Plot friction evolution
    # --------------------------------------------------

    plt.figure(figsize=(7,4))

    plt.plot(t, f)

    plt.axvline(t_step, color='k', linestyle='--')

    plt.xlabel("time")

    plt.ylabel("friction")

    plt.title("Rate-State Velocity Step")

    plt.grid(True)

    plt.tight_layout()

    plt.show()

    # --------------------------------------------------
    # 10. Plot state evolution
    # --------------------------------------------------

    plt.figure(figsize=(7,4))

    plt.semilogy(t, theta)

    plt.axvline(t_step, color='k', linestyle='--')

    plt.xlabel("time")

    plt.ylabel("state variable theta")

    plt.title("State Evolution")

    plt.grid(True)

    plt.tight_layout()

    plt.show()
    
def run_uniaxial_extension_test():

    print("\n========== UNIAXIAL EXTENSION (INCREMENTAL) ==========")

    # -------------------------------------------------
    # 1. parameters
    # -------------------------------------------------

    p = ModelParameters(
        Nx=51,
        Ny=51,
        xsize=2000.0,
        ysize=2000.0,
    )

    grid = Grid(p)

    # -------------------------------------------------
    # 2. build elastic matrix
    # -------------------------------------------------

    builder = MatrixBuilder(p, grid)

    LH = builder.build_LH()

    from scipy.sparse.linalg import factorized
    solve = factorized(LH.tocsc())

    # -------------------------------------------------
    # 3. fields
    # -------------------------------------------------

    ux = np.zeros((p.Ny + 1, p.Nx))
    uy = np.zeros((p.Ny, p.Nx + 1))

    vx = np.zeros_like(ux)
    vy = np.zeros_like(uy)

    # -------------------------------------------------
    # 4. loading
    # -------------------------------------------------

    Vpull = 1e-6       # m/s
    dt = 1.0
    Nt = 1000

    # -------------------------------------------------
    # 5. time stepping
    # -------------------------------------------------

    for it in range(Nt):

        RH = np.zeros(grid.N)

        # ---------------------------------------------
        # LEFT boundary: ux = 0
        # already enforced by LH
        # ---------------------------------------------

        # ---------------------------------------------
        # RIGHT boundary: vx = Vpull
        # ---------------------------------------------

        ix = p.Nx - 1

        for iy in range(1, p.Ny):

            kux, _ = builder._dofs(ix, iy, p.Ny)

            RH[kux] = Vpull

        # ---------------------------------------------
        # solve velocity system
        # ---------------------------------------------

        S = solve(RH)

        #vpx = S[0::2].reshape(p.Ny + 1, p.Nx + 1)
        #vpy = S[1::2].reshape(p.Ny + 1, p.Nx + 1)

        vpx = np.reshape(S[0::2], (p.Nx+1, p.Ny+1), order='C').T
        vpy = np.reshape(S[1::2], (p.Ny+1, p.Nx+1), order='C').T

        vx = vpx[:, :p.Nx]
        vy = vpy[:p.Ny, :]

        # ---------------------------------------------
        # integrate displacement
        # ---------------------------------------------

        ux += vx * dt
        uy += vy * dt

    # -------------------------------------------------
    # 6. compute stresses
    # -------------------------------------------------

    tauqs, sigmaqs = compute_stress_fields(
        uy,
        ux,
        grid.dx,
        grid.dy,
        p.lam,
        p.G,
        grid.cosa,
        grid.sina,
        p.Ny,
        p.Nx
    )

    # -------------------------------------------------
    # 7. diagnostics
    # -------------------------------------------------

    max_tau = np.max(np.abs(tauqs))

    # theoretical strain
    eps = Vpull * Nt * dt / p.xsize

    # theoretical stress
    sigma_theory = (p.lam + 2 * p.G) * eps

    sigma_mean = np.mean(sigmaqs)
    sigma_std  = np.std(sigmaqs)

    print(f"max |tauqs|        = {max_tau:.3e}")
    print(f"mean sigmaqs       = {sigma_mean:.3e}")
    print(f"std sigmaqs        = {sigma_std:.3e}")
    print(f"theoretical sigma  = {sigma_theory:.3e}")

    # -------------------------------------------------
    # 8. PASS/FAIL
    # -------------------------------------------------

    rel_error = abs(sigma_mean - sigma_theory) / abs(sigma_theory)

    if (
        max_tau < 1e-6 * abs(sigma_theory)
        and sigma_std < 1e-3 * abs(sigma_theory)
        and rel_error < 5e-2
    ):
        print("PASS")
    else:
        print("FAIL")

    # -------------------------------------------------
    # 9. plots
    # -------------------------------------------------

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # ux
    im = axes[0,0].pcolormesh(
        grid.Xux,
        grid.Yux,
        ux,
        shading='auto'
    )
    axes[0,0].set_title("ux")
    fig.colorbar(im, ax=axes[0, 0])

    # uy
    im = axes[0,1].pcolormesh(
        grid.Xuy,
        grid.Yuy,
        uy,
        shading='auto'
    )
    axes[0,1].set_title("uy")
    fig.colorbar(im, ax=axes[0, 1])

    # tau
    vmax = np.max(np.abs(tauqs))

    im = axes[1,0].pcolormesh(
        grid.Xtau,
        grid.Ytau,
        tauqs,
        shading='auto',
        vmin=-vmax,
        vmax=vmax
    )
    axes[1,0].set_title("tauqs")
    fig.colorbar(im, ax=axes[1, 0])

    # sigma
    im = axes[1,1].pcolormesh(
        grid.Xsigma,
        grid.Ysigma,
        sigmaqs,
        shading='auto'
    )
    axes[1,1].set_title("sigmaqs")
    fig.colorbar(im, ax=axes[1, 1])

    for ax in axes.flat:
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()

def run_fault_traction_transfer_test():

    print("\n========== FAULT TRACTION TRANSFER TEST ==========")

    # -------------------------------------------------
    # 1. parameters
    # -------------------------------------------------

    p = ModelParameters(
        Nx=201,
        Ny=201,
        xsize=2000.0,
        ysize=2000.0,
    )

    grid = Grid(p)

    builder = MatrixBuilder(p, grid)

    # -------------------------------------------------
    # 2. build matrix
    # -------------------------------------------------

    LH = builder.build_LH()

    from scipy.sparse.linalg import factorized

    solve = factorized(LH.tocsc())

    # -------------------------------------------------
    # 3. fields
    # -------------------------------------------------

    ux = np.zeros((p.Nx+1, p.Ny))
    uy = np.zeros((p.Nx, p.Ny+1))

    vx = np.zeros_like(ux)
    vy = np.zeros_like(uy)

    # -------------------------------------------------
    # 4. loading
    # -------------------------------------------------

    Vpl = 1e-6

    dt = 1.0

    Nt = 1000

    # -------------------------------------------------
    # 5. timestep loop
    # -------------------------------------------------

    for it in range(Nt):

        RH = np.zeros(grid.N)

        # ---------------------------------------------
        # RIGHT boundary loading
        # vy = Vpl
        # ---------------------------------------------

        ix = p.Nx

        for iy in range(1, p.Ny):

            _, kuy = builder._dofs(ix, iy, p.Ny)

            RH[kuy] = Vpl

        # ---------------------------------------------
        # LOCKED FAULT
        # uy+ - uy- = 0
        # already enforced by LH
        # RH remains zero there
        # ---------------------------------------------

        # ---------------------------------------------
        # solve
        # ---------------------------------------------

        S = solve(RH)

        vpx = S[0::2].reshape(
            p.Nx+1,
            p.Ny+1
        ).T

        vpy = S[1::2].reshape(
            p.Nx+1,
            p.Ny+1
        ).T

        vx = vpx[:, :p.Nx]
        vy = vpy[:p.Ny, :]

        # ---------------------------------------------
        # integrate
        # ---------------------------------------------

        ux += vx * dt
        uy += vy * dt

    # -------------------------------------------------
    # 6. stresses
    # -------------------------------------------------

    tauqs, sigmaqs = compute_stress_fields(
        uy,
        ux,
        grid.dx,
        grid.dy,
        p.lam,
        p.G,
        grid.cosa,
        grid.sina,
        p.Ny,
        p.Nx
    )

    # -------------------------------------------------
    # 7. fault diagnostics
    # -------------------------------------------------

    mid = p.Nx // 2

    # left/right traction
    tau_left  = tauqs[:, mid-1]
    tau_right = tauqs[:, mid]

    traction_jump = tau_right - tau_left

    max_jump = np.max(np.abs(traction_jump))

    # displacement jump
    uy_left  = uy[:, mid]
    uy_right = uy[:, mid+1]

    slip = uy_right - uy_left

    max_slip = np.max(np.abs(slip))

    print(f"max traction jump = {max_jump:.3e}")
    print(f"max slip          = {max_slip:.3e}")

    # -------------------------------------------------
    # 8. PASS/FAIL
    # -------------------------------------------------

    if (
        max_jump < 1e-8
        and
        max_slip < 1e-12
    ):
        print("PASS")
    else:
        print("FAIL")

    # -------------------------------------------------
    # 9. plots
    # -------------------------------------------------

    fig, axes = plt.subplots(2,2, figsize=(10,8))

    im = axes[0,0].pcolormesh(
        grid.Xuy,
        grid.Yuy,
        uy,
        shading='auto'
    )
    axes[0,0].set_title("uy")

    im = axes[0,1].pcolormesh(
        grid.Xtau,
        grid.Ytau,
        tauqs,
        shading='auto'
    )
    axes[0,1].set_title("tauqs")

    im = axes[1,0].plot(
        grid.y,
        traction_jump
    )

    axes[1,0].set_title("traction jump")

    im = axes[1,1].plot(
        grid.y,
        slip
    )

    axes[1,1].set_title("fault slip")

    plt.tight_layout()

    plt.show()

def run_constant_fault_slip_test():

    print("\n========== CONSTANT FAULT SLIP TEST ==========")

    # -------------------------------------------------
    # 1. setup
    # -------------------------------------------------

    p = ModelParameters(
        Nx=101,
        Ny=101,
        xsize=2000.0,
        ysize=2000.0,
    )

    grid = Grid(p)

    builder = MatrixBuilder(p, grid)

    LH = builder.build_LH()

    from scipy.sparse.linalg import spsolve

    # -------------------------------------------------
    # 2. prescribed constant slip
    # -------------------------------------------------

    delta0 = 1e-3

    RH = np.zeros(grid.N)

    mid = p.Nx // 2

    # impose constant tangential slip
    for iy in range(1, p.Ny):

        _, kuy = builder._dofs(mid, iy, p.Ny)

        RH[kuy] = delta0

    # -------------------------------------------------
    # 3. solve
    # -------------------------------------------------

    S = spsolve(LH.tocsc(), RH)

    vpx = S[0::2].reshape(
        p.Nx+1,
        p.Ny+1
    ).T

    vpy = S[1::2].reshape(
        p.Nx+1,
        p.Ny+1
    ).T

    ux = vpx[:, :p.Nx]
    uy = vpy[:p.Ny, :]

    #ux += vx * dt
    #uy += vy * dt

    # -------------------------------------------------
    # 4. stresses
    # -------------------------------------------------

    tauqs, sigmaqs = compute_stress_fields(
        uy,
        ux,
        grid.dx,
        grid.dy,
        p.lam,
        p.G,
        grid.cosa,
        grid.sina,
        p.Ny,
        p.Nx
    )

    # -------------------------------------------------
    # 5. diagnostics
    # -------------------------------------------------

    max_tau = np.max(np.abs(tauqs))
    max_sigma = np.max(np.abs(sigmaqs))

    # fault slip
    slip = uy[:, mid+1] - uy[:, mid]

    slip_error = np.max(np.abs(slip - delta0))

    # traction jump
    tau_left  = tauqs[:, mid-1]
    tau_right = tauqs[:, mid]

    traction_jump = tau_right - tau_left

    max_jump = np.max(np.abs(traction_jump))

    print(f"max |tauqs|         = {max_tau:.3e}")
    print(f"max |sigmaqs|       = {max_sigma:.3e}")
    print(f"max traction jump   = {max_jump:.3e}")
    print(f"max slip error      = {slip_error:.3e}")

    # -------------------------------------------------
    # 6. PASS / FAIL
    # -------------------------------------------------

    if (
        max_tau < 1e-8
        and
        max_sigma < 1e-8
        and
        max_jump < 1e-8
        and
        slip_error < 1e-12
    ):
        print("PASS")
    else:
        print("FAIL")

    # -------------------------------------------------
    # 7. plots
    # -------------------------------------------------

    fig, axes = plt.subplots(2,2, figsize=(10,8))

    im = axes[0,0].pcolormesh(
        grid.Xuy,
        grid.Yuy,
        uy,
        shading='auto'
    )
    axes[0,0].set_title("uy")

    im = axes[0,1].pcolormesh(
        grid.Xtau,
        grid.Ytau,
        tauqs,
        shading='auto'
    )
    axes[0,1].set_title("tauqs")

    axes[1,0].plot(grid.y, slip)
    axes[1,0].set_title("fault slip")

    axes[1,1].plot(grid.y, traction_jump)
    axes[1,1].set_title("traction jump")

    plt.tight_layout()
    plt.show()
# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Customise parameters here or leave all defaults
    '''
    params = ModelParameters(
        Nx=21, Ny=21,
        Nt=1000,
        output_interval=10,
        checkpoint_interval=1000,
    )'''

    params = ModelParameters()

    model = FaultSlipModel(params=params, output_dir="output")
    model.run()
    fig = model.grid.plot_mesh()
    fig.show()
    #fig = model.grid.plot_grid()
    model.plot_results()

    #Benchmarks
    #test_rigid_translation()
    #test_rigid_rotation()
    #test_uniaxial_extension()
    #test_constant_strain_equilibrium()
    #test_pure_shear()
    #test_fault_slip_symmetry()
    #test_radiation_damping()
    #test_rate_state_steady_state()
    #run_uniaxial_extension_test()
    #run_fault_traction_transfer_test()
    #run_constant_fault_slip_test()