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

from dataclasses import dataclass, field

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
    Nt: int = 1000               # Number of time steps
    dt_init: float = 1.0         # Initial time step [s]
    dt_max: float = 1e6          # Maximum time step [s]
    yr = 365 * 24 * 3600.0       # Seconds in a year
    tload: float = 1000.0 * yr     # Time to apply pressure rate change [s]  #TODO: CHECK

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
        assert self.Nx % 2 == 1, "Nx must be odd (fault at centre column)."
        assert self.Ny % 2 == 1, "Ny must be odd."