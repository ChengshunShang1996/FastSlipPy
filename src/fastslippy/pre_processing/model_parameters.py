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

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
from fastslippy.pre_processing.layer_parameters import Layer, LayerParameters

class CaseType(str, Enum):
    GRONINGEN = "groningen"
    LAB = "lab"
    CALIFORNIA = "california"

class FrictionLaw(Enum):
    RATE_STATE = "rate_state"
    SLIP_WEAKENING = "slip_weakening"

class LinearSolver(str, Enum):
    DIRECT = "direct"
    ITERATIVE = "iterative"

class IterativeMethod(str, Enum):
    GMRES = "gmres"
    BICGSTAB = "bicgstab"

class BCType(str, Enum):
    FIXED = "fixed"
    FREE = "free"
    VELOCITY = "velocity"
    TRACTION = "traction"
    TRACTION_FREE = "traction_free"

@dataclass
class DirectionBC:

    type: BCType = BCType.FIXED
    value: float = 0.0

    def set_fixed(self):
        self.type = BCType.FIXED
        self.value = 0.0

    def set_free(self):
        self.type = BCType.FREE
        self.value = 0.0

    def set_velocity(self, value: float):
        self.type = BCType.VELOCITY
        self.value = value

    def set_traction(self, value: float):
        self.type = BCType.TRACTION
        self.value = value

    def set_traction_free(self):
        self.type = BCType.TRACTION_FREE
        self.value = 0.0

@dataclass
class BoundaryFace:

    ux: DirectionBC = field(default_factory=DirectionBC)
    uy: DirectionBC = field(default_factory=DirectionBC)

    def set_fixed(self):
        self.ux.set_fixed()
        self.uy.set_fixed()

    def set_free(self):
        self.ux.set_free()
        self.uy.set_free()

    def set_traction_free(self):
        self.ux.set_traction_free()
        self.uy.set_traction_free()

    def set_velocity_x(self, value: float):
        self.ux.set_velocity(value)

    def set_velocity_y(self, value: float):
        self.uy.set_velocity(value)

    def set_traction_x(self, value: float):
        self.ux.set_traction(value)

    def set_traction_y(self, value: float):
        self.uy.set_traction(value)

@dataclass
class BoundaryConditions:
    # Grid y increases with depth: top is y=0 (free surface) and bottom is
    # y=ysize (deep boundary).

    left: BoundaryFace = field(default_factory=BoundaryFace)
    right: BoundaryFace = field(default_factory=BoundaryFace)
    top: BoundaryFace = field(default_factory=BoundaryFace)
    bottom: BoundaryFace = field(default_factory=BoundaryFace)

    def set_all_fixed(self):
        self.left.set_fixed()
        self.right.set_fixed()
        self.top.set_fixed()
        self.bottom.set_fixed()

    def set_all_free(self):
        self.left.set_free()
        self.right.set_free()
        self.top.set_free()
        self.bottom.set_free()

@dataclass
class Layer:
    name: str
    top: float
    bottom: float
    a: float
    b: float

@dataclass
class LoadingConditions:
    """Loading conditions for the model. Edit the defaults here 
    or pass keyword arguments to the constructor."""

    yr = 365 * 24 * 3600.0      # Seconds in a year
    tload: float = 0.0 * yr     # Time to apply pressure rate change [s]

    # --- Pressure rate ---
    dPdt_pre: float = 0.0       # Pressure rate before depletion [Pa/s]
    dPdt_post: float = -0.0127  # Pressure rate after depletion starts [Pa/s]

    V_p: float = 0.0            # Plate velocity [m/s] for the california case
    V_L: float = 0.0            # Imposed fault slip velocity [m/s] for the california case

@dataclass
class ModelParameters:
    """
    All physical and numerical parameters for the fault-slip model.
    Edit the defaults here or pass keyword arguments to the constructor.
    """
    case_type: CaseType = CaseType.LAB
    run_mode: str = "release"  # "debug" or "release"
    
    # --- Fault geometry ---
    alpha: float = 90.0          # Fault dip angle [degrees]
    motion_sign: int = -1        # SEAS convention: +1 thrust, -1 normal
    auto_motion_sign: bool = True # Map BP3 loading to internal sign convention

    # --- Grid ---
    xsize: float = 1.0           # Horizontal model size [m]
    ysize: float = 1.0           # Vertical model size [m]
    Nx: int = 11                 # Horizontal grid points  (must be odd)
    Ny: int = 11                 # Vertical grid points    (must be odd)
    x_stretch_enabled: bool = False
    y_stretch_enabled: bool = False
    x_stretch_inner_size: float = 0.0
    y_stretch_inner_size: float = 0.0
    x_stretch_inner_points: int = 0
    y_stretch_inner_points: int = 0
    x_stretch_power: int = 2
    y_stretch_power: int = 2
    x_stretch_max_cell_size: Optional[float] = None
    y_stretch_max_cell_size: Optional[float] = None
    allow_nonuniform_solver: bool = False

    # --- Material ---
    rho: float = 2650            # Rock density [kg/m³]
    rhof: float = 1150.0         # Fluid density [kg/m³]
    rhog: float = 200.0          # Gas density [kg/m³]
    cs: float = 1645.0           # Shear-wave speed [m/s]
    nu: float = 0.25             # Poisson's ratio
    g: float = 9.81              # Gravitational acceleration [m/s²]
    K0: float = 0.75             # Ratio σ_min / σ_max
    E: float = 0.0               # Young's modulus [Pa] 
    sigma0: float = 50e6         # BP3 effective normal stress [Pa]

    # --- Rate-and-state defaults (used when heterogeneous profile is off) ---
    friction_law: FrictionLaw = FrictionLaw.RATE_STATE
    mu0: float = 0.72             # Reference friction coefficient
    V0: float = 1e-6             # Reference slip rate [m/s]
    a0: float = 0.0012             # Direct effect (homogeneous fallback)
    a_max: float = 0.025           # Maximum direct effect (for California case)
    b0: float = 0.00135            # Evolution effect (homogeneous fallback)
    L: float = 2.25e-6               # Characteristic slip distance [m]
    Vw: float = 1e90             # Dynamic weakening velocity [m/s]
    Vi: float = 1e-30            # Initial/background slip rate [m/s]
    flash_heating_option: bool = False  # Whether to include flash heating in the friction law
    extrapolate_surface_fault_rate: bool = False  # Whether to extrapolate the slip rate at the free-surface/fault intersection
    # At the BP3 free-surface/fault intersection, zero surface shear traction
    # and rate-state friction with finite effective normal stress cannot both
    # be imposed on the same point.  Treat y=0 as a boundary trace and copy the
    # first interior slip rate, matching the endpoint treatment used by the
    # original FastSlipPy friction solvers.
    H: float = 0.0               # California case parameter [m]
    h: float = 0.0               # California case parameter [m]
    W_f: float = 0.0               # California case parameter [m]

    # --- Time stepping ---
    Nt: int = 1000               # Number of time steps
    dt_init: float = 1e-5         # Initial time step [s]
    dt_max: float = 0.002          # Maximum time step [s]
    dt_growth: float = 1.2        # Maximum multiplicative timestep growth
    tfinal: float = np.inf        # Optional final physical time [s]
    friction_tolerance: float = 5.0  # Friction residual tolerance [Pa]

    # --- Output intervals ---
    output_interval: int = 10
    checkpoint_interval: int = 1000
    output_vtk_option: bool = True

    # --- SEAS BP3 output metadata ---
    code_name: str = "FastSlipPy"
    code_version: str = "0.1.2"
    modeler: str = "Chengshun Shang"

    # --- Linear solver ---
    linear_solver: LinearSolver = LinearSolver.DIRECT
    iterative_method: IterativeMethod = IterativeMethod.GMRES
    iterative_rtol: float = 1e-8
    iterative_atol: float = 0.0
    iterative_maxiter: int = 400
    ilu_drop_tol: float = 1e-3
    ilu_fill_factor: float = 10.0
    ilu_permc_spec: str = "COLAMD"
    fallback_to_iterative_on_oom: bool = False

    # --- Derived (computed in __post_init__) ---
    G: float = field(init=False)
    lam: float = field(init=False)   # First Lamé parameter (λ)
    eta: float = field(init=False)   # Radiation damping coefficient

    bc: BoundaryConditions = field(default_factory=BoundaryConditions)
    loading: LoadingConditions = field(default_factory=LoadingConditions)
    layers: LayerParameters = field(default_factory=LayerParameters)

    def __post_init__(self):
        case_value = (
            self.case_type.value
            if isinstance(self.case_type, CaseType)
            else str(self.case_type).lower()
        )
        try:
            self.case_type = CaseType(case_value)
        except ValueError as exc:
            supported = ", ".join(case.value for case in CaseType)
            raise ValueError(
                f"case_type must be one of: {supported}."
            ) from exc

        #self.G = self.rho * self.cs ** 2
        if self.E > 0:
            self.G = self.E / (2 * (1 + self.nu))
            self.cs = np.sqrt(self.G / self.rho)
        else:
            self.G = self.rho * self.cs ** 2
        self.lam = 2 * self.G * (1 + self.nu) / 3 / (1 - 2 * self.nu) - 2 / 3 * self.G
        self.eta = self.G / 2 / self.cs
        solver_mode = self.linear_solver.value if isinstance(self.linear_solver, LinearSolver) else str(self.linear_solver).lower()
        if solver_mode not in (LinearSolver.DIRECT.value, LinearSolver.ITERATIVE.value):
            raise ValueError(
                "linear_solver must be 'direct' or 'iterative'."
            )
        self.linear_solver = LinearSolver(solver_mode)

        method = self.iterative_method.value if isinstance(self.iterative_method, IterativeMethod) else str(self.iterative_method).lower()
        if method not in (IterativeMethod.GMRES.value, IterativeMethod.BICGSTAB.value):
            raise ValueError("iterative_method must be 'gmres' or 'bicgstab'.")
        self.iterative_method = IterativeMethod(method)

        if self.iterative_rtol <= 0.0:
            raise ValueError("iterative_rtol must be > 0.")
        if self.iterative_atol < 0.0:
            raise ValueError("iterative_atol must be >= 0.")
        if self.iterative_maxiter < 1:
            raise ValueError("iterative_maxiter must be >= 1.")
        if self.ilu_drop_tol < 0.0:
            raise ValueError("ilu_drop_tol must be >= 0.")
        if self.ilu_fill_factor <= 0.0:
            raise ValueError("ilu_fill_factor must be > 0.")
        if not self.ilu_permc_spec:
            raise ValueError("ilu_permc_spec must be a non-empty string.")
        assert self.Nx % 2 == 1, "Nx must be odd (fault at centre column)."
        if self.Ny < 4:
            raise ValueError("Ny must provide at least three stress-cell centres.")
        if self.motion_sign not in (-1, 1):
            raise ValueError("motion_sign must be +1 (thrust) or -1 (normal).")
        if self.dt_growth <= 0.0:
            raise ValueError("dt_growth must be positive.")
        if self.friction_tolerance < 0.0:
            raise ValueError("friction_tolerance must be non-negative.")
        if self.x_stretch_enabled:
            if not (0.0 < self.x_stretch_inner_size < self.xsize):
                raise ValueError("x_stretch_inner_size must be in (0, xsize).")
            if self.x_stretch_inner_points < 3 or self.x_stretch_inner_points >= self.Nx:
                raise ValueError("x_stretch_inner_points must be in [3, Nx-1].")
            if self.x_stretch_inner_points % 2 == 0:
                raise ValueError("x_stretch_inner_points must be odd for symmetric x-stretch.")
            if self.x_stretch_power < 1:
                raise ValueError("x_stretch_power must be >= 1.")
            dx_inner = self.x_stretch_inner_size / (self.x_stretch_inner_points - 1)
            dx_mean = self.xsize / (self.Nx - 1)
            if dx_inner > dx_mean:
                raise ValueError(
                    "x_stretch_inner_size/x_stretch_inner_points produces a coarser inner zone "
                    "than the domain-average spacing."
                )
            if self.x_stretch_max_cell_size is not None and self.x_stretch_max_cell_size <= 0.0:
                raise ValueError("x_stretch_max_cell_size must be > 0 when provided.")
        if self.y_stretch_enabled:
            if not (0.0 < self.y_stretch_inner_size < self.ysize):
                raise ValueError("y_stretch_inner_size must be in (0, ysize).")
            if self.y_stretch_inner_points < 2 or self.y_stretch_inner_points >= self.Ny:
                raise ValueError("y_stretch_inner_points must be in [2, Ny-1].")
            if self.y_stretch_power < 1:
                raise ValueError("y_stretch_power must be >= 1.")
            dy_inner = self.y_stretch_inner_size / (self.y_stretch_inner_points - 1)
            dy_mean = self.ysize / (self.Ny - 1)
            if dy_inner > dy_mean:
                raise ValueError(
                    "y_stretch_inner_size/y_stretch_inner_points produces a coarser inner zone "
                    "than the domain-average spacing."
                )
            if self.y_stretch_max_cell_size is not None and self.y_stretch_max_cell_size <= 0.0:
                raise ValueError("y_stretch_max_cell_size must be > 0 when provided.")

    def apply_bp3_motion_sign(self):
        """Apply the SEAS motion convention to BP3 internal velocities.

        SEAS uses ``+1`` for thrust and ``-1`` for normal motion, whereas the
        internal fault jump is ``uy(+) - uy(-)``. Consequently the internal
        velocity sign is ``-motion_sign``, matching the MATLAB reference.
        Magnitudes supplied by the caller are preserved.
        """
        if self.case_type != CaseType.CALIFORNIA or not self.auto_motion_sign:
            return

        internal_sign = -float(self.motion_sign)
        initial_magnitude = abs(float(self.Vi))
        plate_magnitude = abs(float(self.loading.V_p))
        creep_magnitude = abs(float(self.loading.V_L))
        if plate_magnitude == 0.0:
            plate_magnitude = initial_magnitude
        if creep_magnitude == 0.0:
            creep_magnitude = initial_magnitude

        self.Vi = internal_sign * initial_magnitude
        self.loading.V_p = internal_sign * plate_magnitude
        self.loading.V_L = internal_sign * creep_magnitude

        # These are face velocities. The California RHS doubles side values
        # because its LHS rows average ghost and interior unknowns.
        if self.bc.left.uy.type == BCType.VELOCITY:
            self.bc.left.uy.value = -0.5 * self.loading.V_p
        if self.bc.right.uy.type == BCType.VELOCITY:
            self.bc.right.uy.value = 0.5 * self.loading.V_p
        if self.bc.bottom.uy.type == BCType.VELOCITY:
            self.bc.bottom.uy.value = 0.5 * self.loading.V_L
