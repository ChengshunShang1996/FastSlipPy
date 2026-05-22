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

class FigureCreator:
    """
    Utility class for creating and saving figures.
    """

    def __init__(self, OutputData, GridData):
        self.output = OutputData
        self.grid = GridData

    def plot_results(self, Nx, it: int = -1):
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

        # ─────────────────────────────────────────────
        # 1. Ratio τ/σ vs depth
        # ─────────────────────────────────────────────
        tau = om.taum[:, it]
        sigma = om.sigmam[:, it]
        ratio = tau / (sigma + 1e-12)

        fig = plt.figure(figsize=(8, 5))
        plt.plot(grid.y+2000, ratio, 'o-', lw=1)
        plt.ylabel(r"$\tau / \sigma_n$")
        plt.xlabel("Depth [m]")
        plt.title("Ratio shear / normal stress")
        plt.grid(True)
        plt.tight_layout()
        fig.savefig(self.output.out / f"tau_sigma_ratio_it{it}.png", dpi=150)

        fig = plt.figure(figsize=(8, 5))
        plt.plot(grid.y+2000, om.taum[:, it] / 1e6, 'o-', lw=1)
        plt.ylabel(r"Shear stress $\tau$ [MPa]")
        plt.xlabel("Depth [m]")
        plt.grid(True)
        plt.tight_layout()
        fig.savefig(self.output.out / f"tau_it{it}.png", dpi=150)

        fig = plt.figure(figsize=(8, 5))
        plt.plot(grid.y+2000, om.sigmam[:, it] / 1e6, 'o-', lw=1)
        plt.ylabel(r"Normal stress $\sigma_n$ [MPa]")
        plt.xlabel("Depth [m]")
        plt.grid(True)
        plt.tight_layout()
        fig.savefig(self.output.out / f"sigma_it{it}.png", dpi=150)

        fig = plt.figure(figsize=(8, 5))
        plt.plot(grid.y+2000, om.tau0[:, it] / 1e6, 'o-', lw=1)
        plt.ylabel(r"$\tau_0$ [MPa]")
        plt.xlabel("Depth [m]")
        plt.grid(True)
        plt.tight_layout()
        fig.savefig(self.output.out / f"tau0_it{it}.png", dpi=150)

        fig = plt.figure(figsize=(8, 5))
        plt.plot(grid.y+2000, om.Vm[:, it], 'o-', lw=1)
        plt.ylabel(r"Slip velocity $V$ [m/s]")
        plt.xlabel("Depth [m]")
        plt.grid(True)
        plt.tight_layout()
        fig.savefig(self.output.out / f"slip_velocity_it{it}.png", dpi=150)


        mid = Nx // 2
        plt.figure(figsize=(8,6))
        plt.plot(
            grid.y+2000,
            om.taum[:,it],
            'k',
            lw=3,
            label='stored tau'
        )

        for k in [mid-2, mid-1, mid, mid+1, mid+2]:
            plt.plot(grid.y+2000, om.taumall[:,k,it], '--', label=f'col {k}')

        plt.legend()
        plt.grid(True)

        plt.figure(figsize=(8,6))
        plt.plot(
            grid.y+2000,
            om.taum[:,it] - om.tau0[:, it],
            'k',
            lw=3,
            label='recovered tauqs'
        )

        for k in [mid-2, mid-1, mid, mid+1, mid+2]:
            plt.plot(
                grid.y+2000,
                om.taumall[:,k,it],
                '--',
                label=f'col {k}'
            )

        plt.legend()
        plt.grid(True)

        # ─────────────────────────────────────────────
        # 2. Extract 2D fields
        # ─────────────────────────────────────────────
        ux = om.uxmall[:, :, it]
        uy = om.uymall[:, :, it]
        vx = om.vxmall[:, :, it]
        vy = om.vymall[:, :, it]
        tauqs = om.taumall[:, :, it]
        sigmaqs = om.sigmamall[:, :, it]

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
        fig.savefig(self.output.out / f"fields_it{it}.png", dpi=150)

        plt.show()

        #return fig