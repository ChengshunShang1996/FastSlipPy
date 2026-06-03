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
from pathlib import Path

from fastslippy.pre_processing.model_parameters import ModelParameters
from fastslippy.solver.fault_state import FaultState

class OutputManager:
    """
    Handles:
      - in-memory snapshot arrays
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
        self.tau0   = np.zeros((Ny, n))

        self._logfile = open(self.out / "output.txt", "w")

    def log(self, it: int, t: float, dt: float, V: np.ndarray, U: np.ndarray,
            checkpointer: int = 0):
        yr = 365 * 24 * 3600
        line = (f"it={checkpointer+it}, t={t:.6f} yr, dt={dt:.3e}, "
                f"maxV={V.max():.3e}, minV={V.min():.3e}, maxU={U.max():.6f}\n")
        self._logfile.write(line)
        self._logfile.flush()

    def write_memory(self, it: int,
                     U, V, tau, sigma, P, theta, dt, t,
                     tauqs, sigmaqs, uy, vy, ux, vx, tau0):
        idx = it // self.p.output_interval - 1
        self.Um[:, idx]     = U
        self.Vm[:, idx]     = V
        self.taum[:, idx]   = tau
        self.tau0[:, idx]   = tau0
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

    def save_all(self):
        fname = self.out / "dataall.npz"
        np.savez(fname,
                 Um=self.Um, Vm=self.Vm, taum=self.taum,
                 sigmam=self.sigmam, Pm=self.Pm, thetam=self.thetam,
                 dtm=self.dtm, tm=self.tm)

    def close(self):
        self._logfile.close()

    def load_checkpoint(self, checkpointer: int) -> dict:
        fname = self.out / f"data_{checkpointer}.npz"
        return dict(np.load(fname))

    def write_vtk(self, it: int, grid, ux, uy, vx, vy, tauqs, sigmaqs, fault, t: float):
        """
        Write 2D field and fault-line VTK files for ParaView.

        Files written to self.out/:
        fields_it{it:05d}.vts   – structured grid with all 2D fields
        fault_it{it:05d}.vtp    – fault polyline with slip/velocity/stress
        """
        import meshio

        # ── 1. 2D structured grid (tau / stress nodes) ──────────────────────
        # Use the Xtau / Ytau coordinate arrays (shape Ny × Nx)
        # ParaView structured grids need shape (Nz, Ny, Nx) — add a dummy Z dim
        Ny, Nx = grid.p.Ny, grid.p.Nx

        # Build 3D point coordinates (Ny × Nx × 1) for a flat 2D grid
        X = grid.Xtau[:, :, np.newaxis]   # (Ny, Nx, 1)
        Y = grid.Ytau[:, :, np.newaxis]
        Z = np.zeros_like(X)

        # meshio expects points as (N_total, 3) — use structured grid writer
        # Flatten to (Ny*Nx, 3)
        pts = np.column_stack([
            X.ravel(order='C'),
            Y.ravel(order='C'),
            Z.ravel(order='C'),
        ])

        # tauqs and sigmaqs have different shapes — interpolate sigmaqs to Ny×Nx
        # sigmaqs is (Ny-1, Nx-1); pad to (Ny, Nx) with nearest-neighbour fill
        sigmaqs_full = np.zeros((Ny, Nx))
        sigmaqs_full[:-1, :-1] = sigmaqs          # top-left fill
        sigmaqs_full[-1,  :-1] = sigmaqs[-1, :]   # bottom row
        sigmaqs_full[:-1,  -1] = sigmaqs[:, -1]   # right column
        sigmaqs_full[-1,   -1] = sigmaqs[-1, -1]  # corner

        # ux is (Ny+1, Nx); uy is (Ny, Nx+1) — trim/average to (Ny, Nx)
        ux_c = 0.5 * (ux[:Ny, :] + ux[1:Ny+1, :])          # average over y
        uy_c = 0.5 * (uy[:, :Nx] + uy[:, 1:Nx+1])           # average over x
        vx_c = 0.5 * (vx[:Ny, :] + vx[1:Ny+1, :])
        vy_c = 0.5 * (vy[:, :Nx] + vy[:, 1:Nx+1])

        point_data = {
            "tauqs_Pa":    tauqs.ravel(order='C'),
            "sigmaqs_Pa":  sigmaqs_full.ravel(order='C'),
            "ux_m":        ux_c.ravel(order='C'),
            "uy_m":        uy_c.ravel(order='C'),
            "velocity_x":  vx_c.ravel(order='C'),
            "velocity_y":  vy_c.ravel(order='C'),
            "time_s":      np.full(Ny * Nx, t),
        }

        # Use a structured-topology mesh (quads connecting adjacent points)
        # Build connectivity for (Ny-1)*(Nx-1) quads
        i, j = np.meshgrid(np.arange(Ny - 1), np.arange(Nx - 1), indexing='ij')
        i, j = i.ravel(), j.ravel()
        idx  = lambda r, c: r * Nx + c
        cells = np.column_stack([
            idx(i,   j),
            idx(i,   j+1),
            idx(i+1, j+1),
            idx(i+1, j),
        ])

        mesh2d = meshio.Mesh(
            points=pts,
            cells=[("quad", cells)],
            point_data=point_data,
        )
        fname2d = str(self.out / f"fields_it{it:05d}.vtu")
        meshio.write(fname2d, mesh2d)

        # ── 2. Fault polyline ────────────────────────────────────────────────
        # The fault runs along x=0 (mid column) in rotated coords.
        # Grid.y gives the along-fault coordinates.
        y_fault = grid.y                        # (Ny,)
        x_fault = np.zeros_like(y_fault)

        X_f = y_fault * grid.cosa + x_fault
        Y_f = y_fault * grid.sina

        fault_pts = np.column_stack([X_f, Y_f, np.zeros_like(X_f)])

        # Line segments connecting consecutive fault points
        seg_i = np.arange(Ny - 1)
        lines = np.column_stack([seg_i, seg_i + 1])

        fault_point_data = {
            "slip_U_m":       fault.U,
            "slip_rate_V_ms": fault.V,
            "shear_stress_Pa": fault.tau,
            "normal_stress_Pa": fault.sigma,
            "state_theta_s":   np.real(fault.theta).astype(float),
            "time_s":          np.full(Ny, t),
        }

        mesh_fault = meshio.Mesh(
            points=fault_pts,
            cells=[("line", lines)],
            point_data=fault_point_data,
        )
        fname_fault = str(self.out / f"fault_it{it:05d}.vtu")
        meshio.write(fname_fault, mesh_fault)