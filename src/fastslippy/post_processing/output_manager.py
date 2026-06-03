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