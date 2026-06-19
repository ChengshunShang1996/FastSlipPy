#/////////////////////////////////////////////////
__author__      = "Chengshun Shang (Utrecht University)"
__copyright__   = "Copyright (C) 2026-present by Chengshun Shang"
__version__     = "0.0.1"
__maintainer__  = "Chengshun Shang"
__email__       = "c.shang@uu.nl"
__status__      = "development"
__date__        = "May 22, 2026"
__license__     = "MIT License"
#/////////////////////////////////////////////////

import os
import json
import sys
import time
import numpy as np

from scipy.sparse.linalg import factorized
from typing import Optional
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fastslippy.pre_processing.model_parameters import ModelParameters
from fastslippy.pre_processing.grid import Grid
from fastslippy.pre_processing.frictional_zones import FrictionalZones
from fastslippy.solver.stress_state import StressState
from fastslippy.solver.fault_state import FaultState
from fastslippy.solver.matrix_builder import MatrixBuilder
from fastslippy.solver.matrix_builder_shear import MatrixBuilderShear
from fastslippy.utilities.stress_cal_util import StressCalUtil
from fastslippy.post_processing.output_manager import OutputManager
from fastslippy.post_processing.figure_creator import FigureCreator

class FastSlipPy:
    """
    Top-level driver.  Instantiate with a ModelParameters object (or use
    defaults), then call .run().
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
        self.ksi    = self._build_ksi(self.p, self.fric, self.stress.sigman0, self.grid.dy)

        # Displacement / velocity fields
        p  = self.p
        Nx, Ny = p.Nx, p.Ny
        self.ux = np.zeros((Ny + 1, Nx))
        self.uy = np.zeros((Ny, Nx + 1))
        self.vx = np.zeros((Ny + 1, Nx))
        self.vy = np.zeros((Ny, Nx + 1))
        self.tauqs   = np.zeros((Ny, Nx))
        self.sigmaqs = np.zeros((Ny - 1, Nx - 1))

        self.figure_creator = FigureCreator(self.output, self.grid)

    def _build_and_factor_LH(self, dPdt: float):
        builder = MatrixBuilderShear(self.p, self.grid)
        LH = builder.build_LH()
        self.RH_builder = builder
        self.dPdt = dPdt
        self._solve = factorized(LH.tocsc())   # sparse LU decomposition

    def _build_ksi(self, p: ModelParameters, fric: FrictionalZones,
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

    def before_run(self):
        pass
    
    def run(self):
        t0_wall = time.perf_counter()
        p = self.p
        Nx, Ny = p.Nx, p.Ny

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

            if it <= 10000:
                v_load = 1e-4
            elif it <= 20000:
                v_load = 1e-4
            else:
                v_load = 1e-5

            # ── update RH with current slip velocities ──
            RH = self.RH_builder.build_RH(dPdt, self.fault.V, v_load)
            # Inject velocity BC at fault column
            #fault_rows = (np.arange(1, Ny - 1) + (Nx // 2) * (Ny + 1)) * 2 + 1
            #RH[fault_rows] = self.fault.V[1: Ny - 1]

            # ── elastic solve ──
            S   = self._solve(RH)
            vpx = np.reshape(S[0::2], (p.Nx+1, p.Ny+1), order='C').T
            vpy = np.reshape(S[1::2], (p.Ny+1, p.Nx+1), order='C').T
            self.vy = vpy[:Ny, :]
            self.vx = vpx[:, :Nx]

            # ── integrate displacements ──
            self.uy += self.vy * dt
            self.ux += self.vx * dt

            # ── compute stress ──
            StressCalculator = StressCalUtil()
            self.tauqs, self.sigmaqs = StressCalculator.compute_stress_fields(
                self.uy, self.ux, self.grid.dx, self.grid.dy,
                p.lam, p.G, self.grid.cosa, self.grid.sina, Ny, Nx)

            # Update effective normal stress from sigmaqs
            mid_l = (Nx - 1) // 2 - 1
            mid_r = (Nx - 1) // 2
            sigmal = np.concatenate([[self.sigmaqs[0, mid_l]],
                                     StressCalculator._movmean_discard(self.sigmaqs[:, mid_l], 0),
                                     [self.sigmaqs[-1, mid_l]]])
            sigmar = np.concatenate([[self.sigmaqs[0, mid_r]],
                                     StressCalculator._movmean_discard(self.sigmaqs[:, mid_r], 0),
                                     [self.sigmaqs[-1, mid_r]]])
            self.fault.sigma = self.stress.sigman0 - np.minimum(sigmal, sigmar)

            # ── pressure update ──
            #self.stress.update_pressure(dt, dPdt)

            # ── logging ──
            self.output.log(it, t2 if phase == 2 else t, dt,
                            self.fault.V, self.fault.U, self.checkpointer)

            if it % p.output_interval == 0:
                self.output.write_memory(
                    it, self.fault.U, self.fault.V, self.fault.tau,
                    self.fault.sigma, self.stress.P, self.fault.theta,
                    dt, t, self.tauqs, self.sigmaqs,
                    self.uy, self.vy, self.ux, self.vx, self.stress.tau0)
                
                # self.output.write_vtk(
                #     it, self.grid,
                #     self.ux, self.uy, self.vx, self.vy,
                #     self.tauqs, self.sigmaqs,
                #     self.fault, t)

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

        mid = Nx//2
        print(
            "tauqs min/max",
            np.min(self.tauqs[:, mid]),
            np.max(self.tauqs[:, mid])
        )

        print(
            "sigma min/max",
            np.min(self.sigmaqs[:, mid]),
            np.max(self.sigmaqs[:, mid])
        )

        # ── wrap up ──
        self.output.save_all()
        self.output.close()
        self.figure_creator.plot_results_shear(Nx)
        print(f"Done.  Total running time: {time.perf_counter()-t0_wall:.1f}s")
    
    def after_run(self):
        pass


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

    model = FaultSlipPy(params=params, output_dir="output")
    model.run()
    fig = model.grid.plot_mesh()
    fig.show()
    #fig = model.grid.plot_grid()

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