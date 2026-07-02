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

import matplotlib.pyplot as plt
import numpy as np

from fastslippy.pre_processing.model_parameters import ModelParameters
from fastslippy.utilities.math_util import MathUtil

class Grid:
    """
    Builds and stores all spatial coordinate arrays needed by the model.
    """

    def __init__(self, p: ModelParameters):
        
        self.p = p
        self.sina = MathUtil.sind(p.alpha)
        self.cosa = MathUtil.cosd(p.alpha)

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

        if self.p.run_mode == "debug":
            self.plot_grid()

    def plot_mesh(self):
        X0, Y0 = np.meshgrid(self.x, self.y)
        X = Y0 * self.cosa + X0
        Y = Y0 * self.sina
        fig, ax = plt.subplots()
        ax.plot(X, Y, 'k', linewidth=0.4)
        ax.plot(X.T, Y.T, 'k', linewidth=0.4)

        y_line = self.y
        x_line = np.zeros_like(self.y)   # fault 在 x = 0
        X_fault = y_line * self.cosa + x_line
        Y_fault = y_line * self.sina
        ax.plot(X_fault, Y_fault, 'm--', linewidth=2, label='fault')

        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title('2-D Mesh Grid')
        plt.tight_layout()
        fig.savefig(f"mesh.png", dpi=150)
        return fig
    
    def plot_grid(self):

        fig, ax = plt.subplots(figsize=(6, 6))

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
        fig.savefig(f"grid.png", dpi=150)