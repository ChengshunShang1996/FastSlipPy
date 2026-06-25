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

from fastslippy.pre_processing.model_parameters import ModelParameters, CaseType
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
        if self.p.case_type == CaseType.LAB:
            builder = MatrixBuilderShear(self.p, self.grid)
        else: # "groningen"
            builder = MatrixBuilder(self.p, self.grid)
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

        print(f"Setup complete in {time.perf_counter()-t0_wall:.1f}s.  "
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
            self.fault.solve_slip_rate(self.tauqs[:, mid],
                                       self.stress, self.fric)

            # ── adaptive time step ──
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

            if it <= 1000:
                v_load = 1e-4
            elif it <= 2000:
                v_load = 1e-4
            else:
                v_load = 1e-5

            # ── update RH with current slip velocities ──
            RH = self.RH_builder.build_RH(dPdt, self.fault.V)
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

    model = FastSlipPy(params=params, output_dir="output")
    model.run()
    fig = model.grid.plot_mesh()
    fig.show()
    #fig = model.grid.plot_grid()