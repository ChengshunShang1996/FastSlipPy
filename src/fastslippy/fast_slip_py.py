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

from fastslippy.pre_processing.model_parameters import (
    CaseType,
    ModelParameters,
    SlipRateSolver,
    TimeIntegrator,
)
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
        self.p.apply_bp3_motion_sign()
        self.checkpointer = checkpointer
        self.output       = OutputManager(
            self.p, Path(output_dir), append_log=bool(checkpointer)
        )

        # Build grid
        self.grid  = Grid(self.p)
        # Friction profile
        self.fric  = FrictionalZones(self.p, self.grid.y)
        # Initial stress
        self.stress = StressState(self.p, self.grid.y)
        # Fault state
        self.fault  = FaultState(self.p, self.stress, self.fric, fault_y=self.grid.y)
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
        # ``ksi_scale`` is an explicit convergence-control parameter.  It
        # tightens or relaxes only the fault-evolution stability limit; the
        # independent interseismic cap ``dt_max`` remains unchanged.
        ksi = p.ksi_scale * np.where(k3 > 0, k4, k5)
        return ksi

    def _select_adaptive_fault_window(self):
        Ny = self.p.Ny
        if self.p.case_type == "california":
            stop = self.fault.california_loading_start_idx()
            if stop > 0:
                return self.fault.V[:stop], self.ksi[:stop]
            return self.fault.V, self.ksi

        interior_start = 1
        interior_stop = Ny - 1
        return (
            self.fault.V[interior_start:interior_stop],
            self.ksi[interior_start:interior_stop],
        )

    def set_lab_case_velocity_bc(self, p: ModelParameters, t: float):
        if t <= 4:
            p.bc.right.uy.set_velocity(1e-5)
            p.bc.top.uy.set_velocity(1e-5)
            p.bc.bottom.uy.set_velocity(1e-5)
        elif t <= 6:
            p.bc.right.uy.set_velocity(1e-4)
            p.bc.top.uy.set_velocity(1e-4)
            p.bc.bottom.uy.set_velocity(1e-4)
        else:
            p.bc.right.uy.set_velocity(1e-5)
            p.bc.top.uy.set_velocity(1e-5)
            p.bc.bottom.uy.set_velocity(1e-5)

    def before_run(self):
        pass

    def _solve_fault_slip_rate(self, tauqs_col: Optional[np.ndarray] = None):
        """Solve the algebraic rate-and-state equation at one time level."""
        if tauqs_col is None:
            tauqs_col = self.tauqs[:, self.p.Nx // 2]
        if self.p.slip_rate_solver is SlipRateSolver.NEWTON_V2:
            self.fault.solve_slip_rate_newton_v2(
                tauqs_col, self.stress, self.fric
            )
        else:
            self.fault.solve_slip_rate_bisection(
                tauqs_col, self.stress, self.fric
            )

    def _solve_elastic_velocity(
        self, dPdt: float, fault_velocity: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(vx, vy)`` for a supplied fault-rate stage."""
        p = self.p
        RH = self.RH_builder.build_RH(dPdt, fault_velocity)
        solution = self._solve(RH)
        vpx = np.reshape(
            solution[0::2], (p.Nx + 1, p.Ny + 1), order="C"
        ).T
        vpy = np.reshape(
            solution[1::2], (p.Nx + 1, p.Ny + 1), order="C"
        ).T
        return vpx[:, :p.Nx], vpy[:p.Ny, :]

    def _stress_from_displacement(
        self, uy: np.ndarray, ux: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Recover quasistatic stresses and fault effective normal stress."""
        p = self.p
        Nx, Ny = p.Nx, p.Ny
        tauqs, sigmaqs = self.stress_calculator.compute_stress_fields(
            uy, ux, self.grid.dx, self.grid.dy,
            p.lam, p.G, self.grid.cosa, self.grid.sina, Ny, Nx,
            x=self.grid.x, y=self.grid.y,
            xp=self.grid.xp, yp=self.grid.yp,
        )
        mid_l = (Nx - 1) // 2 - 1
        mid_r = (Nx - 1) // 2
        sigmal, sigmar = self.stress_calculator.recover_fault_normal_stress(
            sigmaqs,
            self.grid.x,
            self.grid.y,
            self.grid.xp,
            self.grid.yp,
            mid_l,
            mid_r,
        )
        sigma_fault = self.stress.sigman0 - 0.5 * (sigmal + sigmar)
        return tauqs, sigmaqs, sigma_fault

    def _advance_euler_coupling(self, dt: float, dPdt: float):
        """Advance one step with the original first-order coupling."""
        mid = self.p.Nx // 2
        self.fault.advance(dt, self.tauqs[:, mid], self.stress)
        self.vx, self.vy = self._solve_elastic_velocity(dPdt, self.fault.V)
        self.uy += self.vy * dt
        self.ux += self.vx * dt
        self.tauqs, self.sigmaqs, self.fault.sigma = (
            self._stress_from_displacement(self.uy, self.ux)
        )

    def _advance_rk2_midpoint_coupling(self, dt: float, dPdt: float):
        """Advance all coupled BP3 states using the explicit midpoint rule.

        The first elastic solve predicts displacement and aging state at
        ``t + dt/2``.  The friction equation is then solved again on that
        midpoint state, and its velocity drives the full update of displacement,
        slip, and state.  Stress and effective normal stress are recovered from
        the accepted end-of-step displacement.
        """
        theta0 = self.fault.theta.copy()
        slip0 = self.fault.U.copy()
        ux0 = self.ux.copy()
        uy0 = self.uy.copy()
        velocity0 = self.fault.V.copy()

        vx0, vy0 = self._solve_elastic_velocity(dPdt, velocity0)
        half_dt = 0.5 * dt
        ux_mid = ux0 + half_dt * vx0
        uy_mid = uy0 + half_dt * vy0
        theta_mid = self.fault.theta_after_constant_velocity(
            theta0, velocity0, half_dt
        )
        tau_mid, _, sigma_mid = self._stress_from_displacement(
            uy_mid, ux_mid
        )

        self.fault.theta = theta_mid
        self.fault.sigma = sigma_mid
        self._solve_fault_slip_rate(tau_mid[:, self.p.Nx // 2])
        velocity_mid = self.fault.V.copy()
        vx_mid, vy_mid = self._solve_elastic_velocity(dPdt, velocity_mid)

        self.ux = ux0 + dt * vx_mid
        self.uy = uy0 + dt * vy_mid
        self.fault.U = slip0 + dt * velocity_mid
        self.fault.theta = self.fault.theta_after_constant_velocity(
            theta0, velocity_mid, dt
        )
        self.fault.V = velocity_mid
        self.vx, self.vy = vx_mid, vy_mid
        self.tauqs, self.sigmaqs, self.fault.sigma = (
            self._stress_from_displacement(self.uy, self.ux)
        )
        self.fault.tau = (
            tau_mid[:, self.p.Nx // 2]
            + self.stress.tau0
            - self.p.eta * velocity_mid
        )

    def _synchronized_output_state(
        self, dPdt: float, *, include_velocity_fields: bool
    ):
        """Evaluate an end-of-step algebraic snapshot without changing stages.

        ``V`` and ``tau`` are algebraic variables.  In the legacy Euler method
        the stored stage rate belongs to the beginning of the accepted step,
        while ``u``, ``U``, ``theta`` and ``sigma`` belong to its end.  This
        helper re-solves friction on the end state for output/checkpoints, then
        restores the integrator's stage value so the Euler trajectory remains
        backward compatible.
        """
        stage_velocity = self.fault.V.copy()
        try:
            self._solve_fault_slip_rate()
            velocity = self.fault.V.copy()
        finally:
            self.fault.V = stage_velocity
        traction = (
            self.tauqs[:, self.p.Nx // 2]
            + self.stress.tau0
            - self.p.eta * velocity
        )
        if include_velocity_fields:
            vx, vy = self._solve_elastic_velocity(dPdt, velocity)
        else:
            vx, vy = None, None
        return velocity, traction, vx, vy
    
    def run(self):
        t0_all = time.perf_counter()
        p = self.p
        Nx, Ny = p.Nx, p.Ny

        # ── initialise / load checkpoint ──
        if not self.checkpointer:
            dPdt = p.loading.dPdt_pre
            dt = p.dt_init
            t = 0.0
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
            dt = float(ckpt["dt"])
            t = float(ckpt["t"])
            if p.case_type == "groningen":
                if "Pl" in ckpt and "Pr" in ckpt:
                    self.stress.Pl = ckpt["Pl"]
                    self.stress.Pr = ckpt["Pr"]
                    self.stress.P = np.where(
                        self.grid.y < 1000, self.stress.Pl, self.stress.Pr
                    )
                else:
                    self.stress.update_pressure(min(t, p.loading.tload), p.loading.dPdt_pre)
                    self.stress.update_pressure(max(0.0, t - p.loading.tload), p.loading.dPdt_post)
            elif "P" in ckpt:
                self.stress.P = ckpt["P"]
            if t >= p.tfinal or p.Nt <= 0:
                self.output.close()
                print("No additional time steps requested; existing output retained.")
                return
            self.output.restore_history(t, self.checkpointer, self.stress.tau0)
            if p.case_type == "groningen" and t == p.loading.tload:
                dt = p.dt_init
            dPdt = (
                p.loading.dPdt_post
                if p.case_type == "groningen" and t >= p.loading.tload
                else p.loading.dPdt_pre
            )
            self._build_and_factor_LH(dPdt)

        dt_max = p.dt_max
        t2 = max(0.0, t - p.loading.tload) if p.case_type == "groningen" else t
        phase = (
            0
            if p.case_type == "groningen" and t < p.loading.tload
            else 2
        )

        print(f"Setup complete in {time.perf_counter()-t0_all:.1f}s.  "
              f"Starting {p.Nt} time steps …")

        # ── time loop ────────────────────────────────────────────────
        for it in range(1, p.Nt + 1):
            global_it = self.checkpointer + it

            # Phase transition: pre → post depletion
            if phase == 1:
                dPdt  = p.loading.dPdt_post
                self._build_and_factor_LH(dPdt)
                dt    = p.dt_init
                dt_max = p.dt_max
                t2    = 0.0
                phase = 2

            # Algebraic slip rate at the accepted beginning-of-step state.
            self._solve_fault_slip_rate()

            # ── adaptive time step ──
            V_inner, ksi_inner = self._select_adaptive_fault_window()
            speed = np.maximum(np.abs(V_inner), np.finfo(float).tiny)
            dt_cand = np.min(ksi_inner * p.L / speed)
            dt_cand = max(dt_cand, 1e-150)
            dt = min(p.dt_growth * dt, dt_cand, dt_max, p.tfinal - t)
            if dt <= 0.0:
                break

            # Clamp dt so we hit tload exactly
            if (
                p.case_type == "groningen"
                and phase == 0
                and t + dt >= p.loading.tload
            ):
                dt    = p.loading.tload - t
                phase = 1

            if p.case_type == "lab":
                self.set_lab_case_velocity_bc(p, t)

            if p.time_integrator is TimeIntegrator.EULER:
                self._advance_euler_coupling(dt, dPdt)
            else:
                self._advance_rk2_midpoint_coupling(dt, dPdt)

            # ── pressure update ──
            if p.case_type == "groningen":
                self.stress.update_pressure(dt, dPdt)

            t += dt
            if phase == 2:
                t2 += dt

            reached_final_time = t >= p.tfinal
            needs_velocity_fields = (
                global_it % p.output_interval == 0
                or global_it % p.checkpoint_interval == 0
                or reached_final_time
                or it == p.Nt
            )
            output_V, output_tau, output_vx, output_vy = (
                self._synchronized_output_state(
                    dPdt, include_velocity_fields=needs_velocity_fields
                )
            )

            # ── logging ──
            self.output.log(it, t2 if phase == 2 else t, dt,
                            output_V, self.fault.U, self.checkpointer)

            if global_it % p.output_interval == 0:
                self.output.write_memory(
                    it, self.fault.U, output_V, output_tau,
                    self.fault.sigma, self.stress.P, self.fault.theta,
                    dt, t, self.tauqs, self.sigmaqs,
                    self.uy, output_vy, self.ux, output_vx, self.stress.tau0)
                if p.case_type == "california":
                    self.output.record_bp3_surface(
                        it, t, self.grid, self.ux, self.uy,
                        output_vx, output_vy
                    )

            if global_it % p.checkpoint_interval == 0:
                self.output.save_checkpoint(
                    it, self.checkpointer, self.fault,
                    self.tauqs, self.sigmaqs,
                    self.uy, output_vy, self.ux, output_vx, dt, t,
                    fault_velocity=output_V,
                    fault_traction=output_tau,
                    pressure=self.stress.P,
                    pressure_left=self.stress.Pl, pressure_right=self.stress.Pr)
                self.output.save_all()
                print(f"  Checkpoint it={global_it}, elapsed {time.perf_counter()-t0_all:.1f}s")
                
                if p.output_vtk_option:
                    stage_velocity = self.fault.V
                    stage_traction = self.fault.tau
                    try:
                        self.fault.V = output_V
                        self.fault.tau = output_tau
                        self.output.write_vtk(
                            global_it, self.grid,
                            self.ux, self.uy, output_vx, output_vy,
                            self.tauqs, self.sigmaqs,
                            self.fault, t)
                    finally:
                        self.fault.V = stage_velocity
                        self.fault.tau = stage_traction

            if reached_final_time:
                break

        if p.run_mode == "debug":
            mid = Nx//2
            print("tauqs min/max",  np.min(self.tauqs[:, mid]), np.max(self.tauqs[:, mid]))
            print("sigma min/max", np.min(self.sigmaqs[:, mid]), np.max(self.sigmaqs[:, mid]))

        # ── wrap up ──
        self.output.save_checkpoint(
                    it, self.checkpointer, self.fault,
                    self.tauqs, self.sigmaqs,
                    self.uy, output_vy, self.ux, output_vx, dt, t,
                    fault_velocity=output_V,
                    fault_traction=output_tau,
                    pressure=self.stress.P,
                    pressure_left=self.stress.Pl, pressure_right=self.stress.Pr)
        self.output.save_all()
        if p.case_type == "california":
            self.output.write_bp3_outputs(self.grid)
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
