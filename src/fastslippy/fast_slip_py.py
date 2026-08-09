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

from scipy.sparse.linalg import LinearOperator, bicgstab, factorized, gmres, spilu
from typing import Optional
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fastslippy.pre_processing.model_parameters import ModelParameters, CaseType
from fastslippy.pre_processing.grid import Grid
from fastslippy.pre_processing.frictional_zones import FrictionalZones
from fastslippy.solver.stress_state import StressState
from fastslippy.solver.fault_state import FaultState
from fastslippy.solver.matrix_builder import MatrixBuilder
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
        self.ksi    = self._build_ksi(self.p, self.fric, self.stress.sigman0, self.grid.dy_fault)

        # Displacement / velocity fields
        p  = self.p
        Nx, Ny = p.Nx, p.Ny
        self.ux = np.zeros((Ny + 1, Nx))
        self.uy = np.zeros((Ny, Nx + 1))
        self.vx = np.zeros((Ny + 1, Nx))
        self.vy = np.zeros((Ny, Nx + 1))
        self.tauqs   = np.zeros((Ny, Nx))
        self.sigmaqs = np.zeros((Ny - 1, Nx - 1))
        self.stress_calculator = StressCalUtil(prefer_numba=True)

        self.figure_creator = FigureCreator(self.output, self.grid)

    def _build_and_factor_LH(self, dPdt: float):
        builder = MatrixBuilder(self.p, self.grid)
        LH = builder.build_LH()
        self.RH_builder = builder
        self.dPdt = dPdt
        LH_csc = LH.tocsc()
        solver_mode = self.p.linear_solver.value
        if solver_mode == "direct":
            try:
                self._solve = factorized(LH_csc)   # sparse LU decomposition
            except MemoryError:
                if not self.p.fallback_to_iterative_on_oom:
                    raise
                print("Direct sparse LU ran out of memory; falling back to iterative solver.")
                self._setup_iterative_solver(LH_csc)
            else:
                return
        else:
            self._setup_iterative_solver(LH_csc)

    def _setup_iterative_solver(self, lhs_matrix):
        p = self.p
        ilu = spilu(
            lhs_matrix,
            drop_tol=p.ilu_drop_tol,
            fill_factor=p.ilu_fill_factor,
            permc_spec=p.ilu_permc_spec,
        )
        preconditioner = LinearOperator(lhs_matrix.shape, matvec=ilu.solve)

        def solve(rhs: np.ndarray) -> np.ndarray:
            if p.iterative_method.value == "gmres":
                solution, info = gmres(
                    lhs_matrix,
                    rhs,
                    M=preconditioner,
                    rtol=p.iterative_rtol,
                    atol=p.iterative_atol,
                    maxiter=p.iterative_maxiter,
                )
            else:
                solution, info = bicgstab(
                    lhs_matrix,
                    rhs,
                    M=preconditioner,
                    rtol=p.iterative_rtol,
                    atol=p.iterative_atol,
                    maxiter=p.iterative_maxiter,
                )
            if info != 0:
                raise RuntimeError(
                    f"Iterative solver did not converge (method={p.iterative_method.value}, info={info})."
                )
            return solution

        self._solve = solve

    def _build_ksi(self, p: ModelParameters, fric: FrictionalZones,
              sigman0: np.ndarray, dy) -> np.ndarray:
        """
        Stability factor ksi used for adaptive time stepping:
        """
        dy_arr = np.asarray(dy, dtype=float)
        if dy_arr.ndim == 0:
            dy_arr = np.full_like(sigman0, float(dy_arr), dtype=float)
        if dy_arr.shape != sigman0.shape:
            raise ValueError(f"dy shape {dy_arr.shape} does not match sigma shape {sigman0.shape}.")
        a = fric.a
        b = fric.b
        k1 = (np.pi / 4.0) * p.G / dy_arr * p.L / a / sigman0
        k2 = (b - a) / a
        k3 = (k1 - k2)**2 / 4.0 - k1
        k4 = np.minimum(1.0 / (k1 - k2), 0.2)
        k5 = np.minimum(1.0 - k2 / k1, 0.2)
        ksi = np.where(k3 > 0, k4, k5)
        return ksi

    def before_run(self):
        pass
    
    def run(self):
        t0_all = time.perf_counter()
        p = self.p
        Nx, Ny = p.Nx, p.Ny

        # ── initialise / load checkpoint ──
        if not self.checkpointer:
            dPdt = p.loading.dPdt_pre
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

        print(f"Setup complete in {time.perf_counter()-t0_all:.1f}s.  "
              f"Starting {p.Nt} time steps …")

        # ── time loop ────────────────────────────────────────────────
        for it in range(1, p.Nt + 1):

            # Phase transition: pre → post depletion
            if phase == 1:
                dPdt  = p.loading.dPdt_post
                self._build_and_factor_LH(dPdt)
                dt    = p.dt_init
                dt_max = p.dt_max
                t2    = 0.0
                phase = 2

            # ── velocity solve (rate-and-state) ──
            mid = Nx // 2
            self.fault.solve_slip_rate_newton(self.tauqs[:, mid], self.stress, self.fric)

            # ── adaptive time step ──
            if p.case_type == "california":
                mid_idx = int(p.W_f // (p.ysize / (p.Ny - 1)))
                V_inner = self.fault.V[1: mid_idx]
                ksi_inner = self.ksi[1: mid_idx]
            else:
                V_inner = self.fault.V[1: Ny - 1]
                ksi_inner = self.ksi[1: Ny - 1]
            dt_cand = np.min(ksi_inner * p.L / V_inner)
            dt_cand = max(dt_cand, 1e-150)
            dt      = min(min(1.2 * dt, dt_cand), dt_max)

            # Clamp dt so we hit tload exactly
            if phase == 0 and t + dt >= p.loading.tload:
                dt    = p.loading.tload - t
                phase = 1

            # ── aging law + fault advance ──
            self.fault.advance(dt, self.tauqs[:, mid], self.stress)

            if p.case_type == "lab":
                if it <= 30000:
                    p.bc.right.uy.set_velocity(1e-4)
                    p.bc.top.uy.set_velocity(1e-4)
                    p.bc.bottom.uy.set_velocity(1e-4)
                elif it <= 40000:
                    p.bc.right.uy.set_velocity(1e-4)
                    p.bc.top.uy.set_velocity(1e-4)
                    p.bc.bottom.uy.set_velocity(1e-4)
                else:
                    p.bc.right.uy.set_velocity(1e-5)
                    p.bc.top.uy.set_velocity(1e-5)
                    p.bc.bottom.uy.set_velocity(1e-5)

            # ── update RH with current slip velocities ──
            RH = self.RH_builder.build_RH(dPdt, self.fault.V)
            # Inject velocity BC at fault column
            #fault_rows = (np.arange(1, Ny - 1) + (Nx // 2) * (Ny + 1)) * 2 + 1
            #RH[fault_rows] = self.fault.V[1: Ny - 1]

            # ── elastic solve ──
            S   = self._solve(RH)
            # vpx = np.reshape(S[0::2], (p.Nx+1, p.Ny+1), order='C').T
            # vpy = np.reshape(S[1::2], (p.Ny+1, p.Nx+1), order='C').T
            # self.vy = vpy[:Ny, :]
            # self.vx = vpx[:, :Nx]

            vpx = np.reshape(S[0::2], (p.Nx+1, p.Ny+1), order='C').T
            vpy = np.reshape(S[1::2], (p.Nx+1, p.Ny+1), order='C').T
            self.vy = vpy[:Ny, :]
            self.vx = vpx[:, :Nx]

            # ── integrate displacements ──
            self.uy += self.vy * dt
            self.ux += self.vx * dt

            # ── compute stress ──
            if self.grid.is_nonuniform:
                self.tauqs, self.sigmaqs = self.stress_calculator.compute_stress_fields(
                    self.uy, self.ux, self.grid.dx, self.grid.dy,
                    p.lam, p.G, self.grid.cosa, self.grid.sina, Ny, Nx,
                    x=self.grid.x, y=self.grid.y, xp=self.grid.xp, yp=self.grid.yp)
            else:
                self.tauqs, self.sigmaqs = self.stress_calculator.compute_stress_fields(
                    self.uy, self.ux, self.grid.dx, self.grid.dy,
                    p.lam, p.G, self.grid.cosa, self.grid.sina, Ny, Nx)

            # Update effective normal stress from sigmaqs
            mid_l = (Nx - 1) // 2 - 1
            mid_r = (Nx - 1) // 2
            sigmal = np.concatenate([[self.sigmaqs[0, mid_l]],
                                     self.stress_calculator._movmean_discard(self.sigmaqs[:, mid_l], 0),
                                     [self.sigmaqs[-1, mid_l]]])
            sigmar = np.concatenate([[self.sigmaqs[0, mid_r]],
                                     self.stress_calculator._movmean_discard(self.sigmaqs[:, mid_r], 0),
                                     [self.sigmaqs[-1, mid_r]]])
            self.fault.sigma = self.stress.sigman0 - np.minimum(sigmal, sigmar)

            # ── pressure update ──
            if p.case_type == "groningen":
                self.stress.update_pressure(dt, dPdt)

            # ── logging ──
            self.output.log(it, t2 if phase == 2 else t, dt,
                            self.fault.V, self.fault.U, self.checkpointer)

            if it % p.output_interval == 0:
                self.output.write_memory(
                    it, self.fault.U, self.fault.V, self.fault.tau,
                    self.fault.sigma, self.stress.P, self.fault.theta,
                    dt, t, self.tauqs, self.sigmaqs,
                    self.uy, self.vy, self.ux, self.vx, self.stress.tau0)

            if it % p.checkpoint_interval == 0:
                self.output.save_checkpoint(
                    it, self.checkpointer, self.fault,
                    self.tauqs, self.sigmaqs,
                    self.uy, self.vy, self.ux, self.vx, dt, t)
                self.output.save_all()
                print(f"  Checkpoint it={it}, elapsed {time.perf_counter()-t0_all:.1f}s")
                
                if p.output_vtk_option:
                    self.output.write_vtk(
                        it, self.grid,
                        self.ux, self.uy, self.vx, self.vy,
                        self.tauqs, self.sigmaqs,
                        self.fault, t)

            t += dt
            if phase == 2:
                t2 += dt

        if p.run_mode == "debug":
            mid = Nx//2
            print("tauqs min/max",  np.min(self.tauqs[:, mid]), np.max(self.tauqs[:, mid]))
            print("sigma min/max", np.min(self.sigmaqs[:, mid]), np.max(self.sigmaqs[:, mid]))

        # ── wrap up ──
        self.output.save_checkpoint(
                    it, self.checkpointer, self.fault,
                    self.tauqs, self.sigmaqs,
                    self.uy, self.vy, self.ux, self.vx, dt, t)
        self.output.save_all()
        self.output.close()
        if p.case_type == "groningen":
            self.figure_creator.plot_results(Nx, shift_y=2000)
        elif p.case_type == "lab":
            self.figure_creator.plot_results_shear(Nx)
        elif p.case_type == "california":
            self.figure_creator.plot_results(Nx, shift_y=0)
        print(f"Done.  Total running time: {time.perf_counter()-t0_all:.1f}s")
    
    def after_run(self):
        pass

if __name__ == "__main__":
    
    # Customise parameters here or leave all defaults
    params = ModelParameters()

    model = FastSlipPy(params=params, output_dir="output")
    model.run()