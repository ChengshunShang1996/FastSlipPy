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

import matplotlib
matplotlib.use("Agg")
import numpy as np

import matplotlib.pyplot as plt

class FigureCreator:
    """
    Utility class for creating and saving figures.
    """

    def __init__(self, OutputData, GridData):
        self.output = OutputData
        self.grid = GridData

    def plot_results(self, Nx, shift_y, it: int = -1):
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
            # Snapshot arrays are preallocated to Nt/output_interval.  A run
            # stopped before Nt therefore has trailing all-zero columns; the
            # last allocated column is not necessarily the last result.
            valid = np.flatnonzero(om.tm > 0.0)
            if valid.size == 0:
                raise ValueError("No stored output snapshot is available.")
            it = int(valid[-1])

        actual_it = (it + 1) * om.p.output_interval

        # ─────────────────────────────────────────────
        # 1. Ratio τ/σ vs depth
        # ─────────────────────────────────────────────
        tau = om.taum[:, it]
        sigma = om.sigmam[:, it]
        ratio = tau / (sigma + 1e-12)

        fig = plt.figure(figsize=(8, 5))
        plt.plot(grid.y+shift_y, ratio, 'o-', lw=1)
        plt.ylabel(r"$\tau / \sigma_n$")
        plt.xlabel("Depth [m]")
        plt.title("Ratio shear / normal stress")
        plt.grid(True)
        plt.tight_layout()
        fig.savefig(self.output.out / f"tau_sigma_ratio_it{it+1}.png", dpi=150)

        fig = plt.figure(figsize=(8, 5))
        plt.plot(grid.y+shift_y, om.taum[:, it] / 1e6, 'o-', lw=1)
        plt.ylabel(r"Shear stress $\tau$ [MPa]")
        plt.xlabel("Depth [m]")
        plt.grid(True)
        plt.tight_layout()
        fig.savefig(self.output.out / f"tau_it{it+1}.png", dpi=150)

        fig = plt.figure(figsize=(8, 5))
        plt.plot(grid.y+shift_y, om.sigmam[:, it] / 1e6, 'o-', lw=1)
        plt.ylabel(r"Normal stress $\sigma_n$ [MPa]")
        plt.xlabel("Depth [m]")
        plt.grid(True)
        plt.tight_layout()
        fig.savefig(self.output.out / f"sigma_it{it+1}.png", dpi=150)

        fig = plt.figure(figsize=(8, 5))
        plt.plot(grid.y+shift_y, om.tau0[:, it] / 1e6, 'o-', lw=1)
        plt.ylabel(r"$\tau_0$ [MPa]")
        plt.xlabel("Depth [m]")
        plt.grid(True)
        plt.tight_layout()
        fig.savefig(self.output.out / f"tau0_it{it+1}.png", dpi=150)

        fig = plt.figure(figsize=(8, 5))
        plt.plot(grid.y+shift_y, om.Vm[:, it], 'o-', lw=1)
        plt.ylabel(r"Slip velocity $V$ [m/s]")
        plt.xlabel("Depth [m]")
        plt.grid(True)
        plt.tight_layout()
        fig.savefig(self.output.out / f"slip_velocity_it{it+1}.png", dpi=150)

        data_fname = om.out / f"data_{actual_it}.npz"
        if not data_fname.exists():
            print(f"Warning: Data file {data_fname} does not exist. Skipping 2D plots.")
            return
        
        with np.load(data_fname) as data:
            tauqs = data["tauqs"]
            sigmaqs = data["sigmaqs"]
            uy = data["uy"]
            vy = data["vy"]
            ux = data["ux"]
            vx = data["vx"]

        mid = Nx // 2
        fig_verify1 = plt.figure(figsize=(8,6))
        plt.plot(grid.y, om.taum[:, it], 'k', lw=3, label='stored tau')
        for k in [mid-2, mid-1, mid, mid+1, mid+2]:
            plt.plot(grid.y, tauqs[:, k], '--', label=f'col {k}')
        plt.legend()
        plt.grid(True)
        fig_verify1.savefig(self.output.out / f"verify_tau_it{it + 1}.png", dpi=150)
        plt.close(fig_verify1)

        fig_verify2 = plt.figure(figsize=(8,6))
        plt.plot(grid.y, om.taum[:, it] - om.tau0[:, it], 'k', lw=3, label='recovered tauqs')
        for k in [mid-2, mid-1, mid, mid+1, mid+2]:
            plt.plot(grid.y, tauqs[:, k], '--', label=f'col {k}')
        plt.legend()
        plt.grid(True)
        fig_verify2.savefig(self.output.out / f"verify_tauqs_it{it + 1}.png", dpi=150)
        plt.close(fig_verify2)

        # ─────────────────────────────────────────────
        # 2. Extract 2D fields
        # ─────────────────────────────────────────────
        # ux = om.uxmall[:, :, it]
        # uy = om.uymall[:, :, it]
        # vx = om.vxmall[:, :, it]
        # vy = om.vymall[:, :, it]
        # tauqs = om.taumall[:, :, it]
        # sigmaqs = om.sigmamall[:, :, it]

        # ─────────────────────────────────────────────
        # 3. 2D plots (match your layout)
        # ─────────────────────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # --- ux / vx ---
        im = axes[0, 0].pcolormesh(
            grid.Xux, grid.Yux, vx,
            shading='auto'
        )
        axes[0, 0].set_title("Vx")
        fig.colorbar(im, ax=axes[0, 0])

        # --- tauqs ---
        im = axes[0, 1].pcolormesh(
            grid.Xtau, grid.Ytau, tauqs,
            shading='auto'
        )
        axes[0, 1].set_title(r"Shear stress $\tau_{qs}$")
        fig.colorbar(im, ax=axes[0, 1])

        # --- uy ---
        im = axes[1, 0].pcolormesh(
            grid.Xuy, grid.Yuy, vy,
            shading='auto'
        )
        axes[1, 0].set_title("Vy")
        fig.colorbar(im, ax=axes[1, 0])

        # --- sigmaqs ---
        im = axes[1, 1].pcolormesh(
            grid.Xsigma, grid.Ysigma, sigmaqs,
            shading='auto'
        )
        axes[1, 1].set_title(r"Normal stress $\sigma_{qs}$")
        fig.colorbar(im, ax=axes[1, 1])

        for ax in axes.flat:
            ax.set_aspect('equal')
            ax.invert_yaxis()

        plt.tight_layout()
        fig.savefig(self.output.out / f"fields_it{it+1}.png", dpi=150)

        #plt.show()

        #return fig

    def plot_results_shear(self, Nx, it: int = -1):
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

        actual_it = (it + 1) * om.p.output_interval

        # ─────────────────────────────────────────────
        # 1. Ratio τ/σ vs depth
        # ─────────────────────────────────────────────
        tau = om.taum[:, it]
        sigma = om.sigmam[:, it]
        ratio = tau / (sigma + 1e-12)

        fig = plt.figure(figsize=(8, 5))
        plt.plot(grid.y, ratio, 'o-', lw=1)
        plt.ylabel(r"$\tau / \sigma_n$")
        plt.xlabel("Depth [m]")
        plt.title("Ratio shear / normal stress")
        plt.grid(True)
        plt.tight_layout()
        fig.savefig(self.output.out / f"tau_sigma_ratio_it{it+1}.png", dpi=150)

        fig = plt.figure(figsize=(8, 5))
        plt.plot(grid.y, om.taum[:, it] / 1e6, 'o-', lw=1)
        plt.ylabel(r"Shear stress $\tau$ [MPa]")
        plt.xlabel("Depth [m]")
        plt.grid(True)
        plt.tight_layout()
        fig.savefig(self.output.out / f"tau_it{it+1}.png", dpi=150)

        fig = plt.figure(figsize=(8, 5))
        plt.plot(grid.y, om.sigmam[:, it] / 1e6, 'o-', lw=1)
        plt.ylabel(r"Normal stress $\sigma_n$ [MPa]")
        plt.xlabel("Depth [m]")
        plt.grid(True)
        plt.tight_layout()
        fig.savefig(self.output.out / f"sigma_it{it+1}.png", dpi=150)

        fig = plt.figure(figsize=(8, 5))
        plt.plot(grid.y, om.tau0[:, it] / 1e6, 'o-', lw=1)
        plt.ylabel(r"$\tau_0$ [MPa]")
        plt.xlabel("Depth [m]")
        plt.grid(True)
        plt.tight_layout()
        fig.savefig(self.output.out / f"tau0_it{it+1}.png", dpi=150)

        fig = plt.figure(figsize=(8, 5))
        plt.plot(grid.y, om.Vm[:, it], 'o-', lw=1)
        plt.ylabel(r"Slip velocity $V$ [m/s]")
        plt.xlabel("Depth [m]")
        plt.grid(True)
        plt.tight_layout()
        fig.savefig(self.output.out / f"slip_velocity_it{it+1}.png", dpi=150)

        field_fname = om.out / f"data_{actual_it}.npz"
        
        if not field_fname.exists():
            print(f"Warning: Data file {field_fname} does not exist. Skipping 2D plots.")
            return
        
        with np.load(field_fname) as data:
            tauqs = data["tauqs"]
            sigmaqs = data["sigmaqs"]
            uy = data["uy"]
            vy = data["vy"]
            ux = data["ux"]
            vx = data["vx"]

        mid = Nx // 2
        fig_verify1 = plt.figure(figsize=(8,6))
        plt.plot(grid.y, om.taum[:, it], 'k', lw=3, label='stored tau')
        for k in [mid-2, mid-1, mid, mid+1, mid+2]:
            plt.plot(grid.y, tauqs[:, k], '--', label=f'col {k}')
        plt.legend()
        plt.grid(True)
        fig_verify1.savefig(self.output.out / f"verify_tau_it{it + 1}.png", dpi=150)
        plt.close(fig_verify1)

        fig_verify2 = plt.figure(figsize=(8,6))
        plt.plot(grid.y, om.taum[:, it] - om.tau0[:, it], 'k', lw=3, label='recovered tauqs')
        for k in [mid-2, mid-1, mid, mid+1, mid+2]:
            plt.plot(grid.y, tauqs[:, k], '--', label=f'col {k}')
        plt.legend()
        plt.grid(True)
        fig_verify2.savefig(self.output.out / f"verify_tauqs_it{it + 1}.png", dpi=150)
        plt.close(fig_verify2)

        # ─────────────────────────────────────────────
        # 2. Extract 2D fields
        # ─────────────────────────────────────────────
        # ux = om.uxmall[:, :, it]
        # uy = om.uymall[:, :, it]
        # vx = om.vxmall[:, :, it]
        # vy = om.vymall[:, :, it]
        # tauqs = om.taumall[:, :, it]
        # sigmaqs = om.sigmamall[:, :, it]

        # ─────────────────────────────────────────────
        # 3. 2D plots (match your layout)
        # ─────────────────────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # --- ux / vx ---
        im = axes[0, 0].pcolormesh(
            grid.Xux, grid.Yux, vx,
            shading='auto'
        )
        axes[0, 0].set_title("Vx")
        fig.colorbar(im, ax=axes[0, 0])

        # --- tauqs ---
        im = axes[0, 1].pcolormesh(
            grid.Xtau, grid.Ytau, tauqs,
            shading='auto'
        )
        axes[0, 1].set_title(r"Shear stress $\tau_{qs}$")
        fig.colorbar(im, ax=axes[0, 1])

        # --- uy ---
        im = axes[1, 0].pcolormesh(
            grid.Xuy, grid.Yuy, vy,
            shading='auto'
        )
        axes[1, 0].set_title("Vy")
        fig.colorbar(im, ax=axes[1, 0])

        # --- sigmaqs ---
        im = axes[1, 1].pcolormesh(
            grid.Xsigma, grid.Ysigma, sigmaqs,
            shading='auto'
        )
        axes[1, 1].set_title(r"Normal stress $\sigma_{qs}$")
        fig.colorbar(im, ax=axes[1, 1])

        for ax in axes.flat:
            ax.set_aspect('equal')

        plt.tight_layout()
        fig.savefig(self.output.out / f"fields_it{it + 1}.png", dpi=150)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        im = axes[0].pcolormesh(
            grid.Xux, grid.Yux, ux,
            shading='auto'
        )
        axes[0].set_title("Ux")
        fig.colorbar(im, ax=axes[0])

        im = axes[1].pcolormesh(
            grid.Xuy, grid.Yuy, uy,
            shading='auto'
        )
        axes[1].set_title("Uy")
        fig.colorbar(im, ax=axes[1])

        for ax in axes.flat:
            ax.set_aspect('equal')

        plt.tight_layout()
        fig.savefig(self.output.out / f"displacement_fields_it{it + 1}.png", dpi=150)

        #plt.show()

        #return fig
