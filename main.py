"""
fault_slip_model.py
===================
Python / OOP rewrite of Main_integrate.m
Seismo-thermomechanical numerical model with:
  - fluid pressure
  - reservoir setup + gravity
  - boundary conditions with total stress continuity
  - flash heating
  - adjustable dP/dt

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

import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


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
    Nx: int = 101                # Horizontal grid points  (must be odd)
    Ny: int = 101                # Vertical grid points    (must be odd)

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
    Nt: int = 1000               # Number of time steps
    dt_init: float = 1.0        # Initial time step [s]
    dt_max: float = 1e6         # Maximum time step [s]
    tload: Optional[float] = None  # Time to apply pressure rate change [s]
                                   # None → computed as 1000 yr

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
        yr = 365 * 24 * 3600
        self.G = self.rho * self.cs ** 2
        self.lam = 2 * self.G * (1 + self.nu) / 3 / (1 - 2 * self.nu) - 2 / 3 * self.G
        self.eta = self.G / 2 / self.cs
        if self.tload is None:
            self.tload = 1000 * yr
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
        self.sina = np.sin(np.deg2rad(p.alpha))
        self.cosa = np.cos(np.deg2rad(p.alpha))

        Nx, Ny = p.Nx, p.Ny
        dx = p.xsize / (Nx - 1)
        dy = p.ysize / (Ny - 1)
        self.dx = dx
        self.dy = dy

        # Basic (τ / σ) nodes
        self.x = np.linspace(-p.xsize / 2, p.xsize / 2, Nx)          # (Nx,)
        self.y = np.linspace(0, p.ysize, Ny)                           # (Ny,)

        # Pressure / staggered nodes
        self.xp = np.arange(-p.xsize / 2 - dx / 2,
                             p.xsize / 2 + dx / 2 + dx / 2, dx)       # (Nx+1,)
        self.yp = np.arange(-dy / 2, p.ysize + dy, dy)                # (Ny+1,)

        # Rotated coordinate helpers (kept for post-processing / plotting)
        self.Xuy  = self.y[:, None] * self.cosa + self.xp[None, :]
        #self.Yuy  = self.y[:, None] * self.sina
        self.Yuy  = np.repeat(self.y[:, None] * self.sina, len(self.xp), axis=1)
        self.Xux  = self.yp[:, None] * self.cosa + self.x[None, :]
        #self.Yux  = self.yp[:, None] * self.sina
        self.Yux  = self.yp[:, None] * self.sina + 0 * self.x[None, :]
        self.Xtau = self.y[:, None] * self.cosa + self.x[None, :]
        #self.Ytau = self.y[:, None] * self.sina
        self.Ytau = self.y[:, None] * self.sina + 0 * self.x[None, :]
        self.Xsigma = self.yp[1:Ny, None] * self.cosa + self.xp[None, 1:Nx]
        #self.Ysigma = self.yp[1:Ny, None] * self.sina
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
    
    def plot_grid(self, show_sigma=False):

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
        ax.scatter(self.Xuy, self.Yuy, s=5, c='blue', label='uy nodes')

        # ─────────────────────────────
        # 3. ux points（red）
        # ─────────────────────────────
        ax.scatter(self.Xux, self.Yux, s=5, c='red', label='ux nodes')

        # ─────────────────────────────
        # 4. sigma points（green, optional）
        # ─────────────────────────────
        if show_sigma:
            ax.scatter(self.Xsigma, self.Ysigma, s=5, c='green', label='sigma nodes')

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
        "Rocksalt":   {"top": 2000, "bot": 2730, "a": 0.00447,  "b": -0.00590},
        "BasalZech":  {"top": 2730, "bot": 2780, "a": 0.06895,  "b":  0.07209},
        "TenBoer":    {"top": 2780, "bot": 2850, "a": 0.00305,  "b": -0.00093},
        "Sandstone":  {"top": 2850, "bot": 3050, "a": 0.04065,  "b":  0.03796},
        "Carbonif":   {"top": 3050, "bot": 4000, "a": 0.02538,  "b":  0.02347},
    }

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

        for name, layer in self.LAYERS.items():
            # Convert absolute depth [m] to model y-coordinate
            top_y = layer["top"] - p.ysize
            bot_y = layer["bot"] - p.ysize
            mask = (y > top_y) & (y <= bot_y)
            a[mask] = layer["a"]
            b[mask] = layer["b"]

        # Fill any unassigned nodes with homogeneous defaults
        a[a == 0] = p.a0
        b[b == 0] = p.b0
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
                   + (1 - p.K0) / 2 * np.cos(np.deg2rad(2 * p.alpha)) * sigmav
                   - np.where(y < 1000, Pl0, Pr0))
        tau0 = (1 - p.K0) / 2 * np.sin(np.deg2rad(2 * p.alpha)) * sigmav
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
        self.theta = (p.L / p.V0
                      * np.exp(fric.a / fric.b
                               * np.log(2 * p.V0 / p.Vi
                                        * np.sinh((stress.tau0 - p.eta * p.Vi)
                                                  / fric.a / stress.sigman0))
                               - p.mu0 / fric.b))
        self.sigma = stress.sigman0.copy()
        self.tau   = stress.tau0 - p.eta * self.V

    # ------------------------------------------------------------------
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
        self.tau   = tauqs_col + stress.tau0 - p.eta * self.V
        self.U     = self.U + dt * self.V


# ─────────────────────────────────────────────────────────────
# 6.  Adaptive time-step stabiliser  (ksi)
# ─────────────────────────────────────────────────────────────

def build_ksi(p: ModelParameters, fric: FrictionalZones,
              sigman0: np.ndarray) -> np.ndarray:
    """
    Stability factor ksi used for adaptive time stepping:
        ksi_i = (b - a) * σ / (G/2cs * L)   (simplified form from MATLAB)
    Returns array of shape (Ny,).
    """
    a, b = fric.a, fric.b
    ksi = np.abs((b - a) * sigman0 / (p.G / 2 / p.cs)) / p.L
    # Clamp to [1e-150, inf] to avoid division by zero
    ksi = np.where(ksi < 1e-150, 1e-150, ksi)
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

        for ix in range(Nx + 1):           # 0 … Nx  (MATLAB 1 … Nx+1)
            for iy in range(Ny + 1):       # 0 … Ny

                kux, kuy = self._dofs(ix, iy, Ny)
                mid = (Nx) // 2            # fault column index (0-based)

                # ── uy equation (iy < Ny) ──────────────────────────────
                if iy < Ny:
                    if ix == 0:
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
                        for sign, col_off in [(1, (Ny+1)*2), (1, (Ny+1)*2+2),
                                              (-2, -(Ny+1)*2), (-2, -(Ny+1)*2+2),
                                              (1, -3*(Ny+1)*2), (1, -3*(Ny+1)*2+2)]:
                            pass  # These are in kux space; handled via kux
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
                        base_ux, _ = self._dofs(ix, iy, Ny)
                        fac = 1/dy*dx*(lam + G)/G
                        if ix == 1 or ix == Nx - 1:
                            add(kuy, base_ux - (Ny+1)*2,      fac)
                            add(kuy, base_ux - (Ny+1)*2 + 2, -fac)
                            add(kuy, base_ux,                 -fac)
                            add(kuy, base_ux + 2,              fac)
                        else:
                            cf = cosa*(lam + G)/G/4
                            add(kuy, base_ux - (Ny+1)*2,      fac + cf)
                            add(kuy, base_ux - (Ny+1)*2 + 2, -fac + cf)
                            add(kuy, base_ux,                 -fac + cf)
                            add(kuy, base_ux + 2,              fac + cf)
                            add(kuy, base_ux - 2*(Ny+1)*2,    -cf)
                            add(kuy, base_ux - 2*(Ny+1)*2+2,  -cf)
                            add(kuy, base_ux + (Ny+1)*2,      -cf)
                            add(kuy, base_ux + (Ny+1)*2 + 2,  -cf)
                else:
                    add(kuy, kuy, 1)

                # ── ux equation (ix < Nx) ──────────────────────────────
                if ix < Nx:
                    _, kuy_n = self._dofs(ix, iy, Ny)
                    mid_ix = mid
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
                    elif ix == mid_ix:
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

        for ix in range(Nx + 1):
            for iy in range(Ny + 1):
                kux, kuy = self._dofs(ix, iy, Ny)

                # ── uy block ──
                if iy < Ny:
                    if ix not in (0, Nx) and iy not in (0, Ny - 1):
                        if ix == mid:
                            RH[kuy] = V[iy]
                        elif ix == mid + 1:
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

                # ── ux block ──
                if ix < Nx:
                    if iy not in (0, Ny) and ix not in (0, Nx - 1):
                        yv = y[iy]
                        if ix == mid:
                            if 800 < yv <= 850:
                                RH[kux] = -dPdt * dx / G
                            if 1000 < yv <= 1050:
                                RH[kux] =  dPdt * dx / G
                        else:
                            if yv == 1050 and ix > mid + 1:
                                RH[kux] =  dPdt / dy * dx*dx / G * sina * cosa
                            if yv == 1000 and ix < mid + 1:
                                RH[kux] =  dPdt / dy * dx*dx / G * sina * cosa
                            if yv == 850 and ix > mid + 1:
                                RH[kux] = -dPdt / dy * dx*dx / G * sina * cosa
                            if yv == 800 and ix < mid + 1:
                                RH[kux] = -dPdt / dy * dx*dx / G * sina * cosa
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
        self.ksi    = build_ksi(self.p, self.fric, self.stress.sigman0)

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

            # ── elastic solve ──
            S   = self._solve(RH)
            vpx = S[0::2].reshape(Ny + 1, Nx + 1)
            vpy = S[1::2].reshape(Ny + 1, Nx + 1)
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
        ax.set_xlabel("Time [yr]");  ax.set_ylabel("Depth [m]")
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

        plt.figure(figsize=(6, 4))
        plt.plot(ratio, grid.y, lw=2)
        plt.gca().invert_yaxis()
        plt.xlabel(r"$\tau / \sigma_n$")
        plt.ylabel("Depth [m]")
        plt.title("Ratio shear / normal stress")
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

        # --- uy / vy ---
        im = axes[0, 0].pcolormesh(
            grid.Xuy, grid.Yuy, uy,
            shading='auto'
        )
        axes[0, 0].set_title("Uy displacement")
        fig.colorbar(im, ax=axes[0, 0])

        # --- tauqs ---
        im = axes[0, 1].pcolormesh(
            grid.Xtau, grid.Ytau, tauqs,
            shading='auto'
        )
        axes[0, 1].set_title("Shear stress τqs")
        fig.colorbar(im, ax=axes[0, 1])

        # --- ux ---
        im = axes[1, 0].pcolormesh(
            grid.Xux, grid.Yux, ux,
            shading='auto'
        )
        axes[1, 0].set_title("Ux displacement")
        fig.colorbar(im, ax=axes[1, 0])

        # --- sigmaqs ---
        im = axes[1, 1].pcolormesh(
            grid.Xsigma, grid.Ysigma, sigmaqs,
            shading='auto'
        )
        axes[1, 1].set_title("Normal stress σqs")
        fig.colorbar(im, ax=axes[1, 1])

        for ax in axes.flat:
            ax.set_aspect('equal')
            ax.invert_yaxis()

        plt.tight_layout()
        plt.show()

        return fig
# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Customise parameters here or leave all defaults
    params = ModelParameters(
        Nx=21, Ny=21,
        Nt=1000,
        output_interval=10,
        checkpoint_interval=100,
    )

    model = FaultSlipModel(params=params, output_dir="output")
    model.run()
    fig = model.grid.plot_mesh()
    fig.show()
    model.plot_results()