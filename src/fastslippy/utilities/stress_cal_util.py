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

import numpy as np

class StressCalUtil:
    """
    Utility functions for stress calculations.
    """

    def __init__(self):
        pass

    def _movmean_discard(self, arr: np.ndarray, axis: int) -> np.ndarray:
        """Running mean of adjacent pairs along *axis*, discarding endpoints."""
        sl_a = [slice(None)] * arr.ndim
        sl_b = [slice(None)] * arr.ndim
        sl_a[axis] = slice(None, -1)
        sl_b[axis] = slice(1, None)
        return (arr[tuple(sl_a)] + arr[tuple(sl_b)]) / 2

    def compute_stress_fields(self, uy, ux, dx, dy, lam, G, cosa, sina, Ny, Nx):
        """
        Compute tauqs (Ny × Nx) and sigmaqs (Ny-1 × Nx-1) from displacement fields.

        Grid shapes (matching MATLAB staggered layout):
            uy  : (Ny,   Nx+1)   – y-displacement on uy-nodes
            ux  : (Ny+1, Nx)     – x-displacement on ux-nodes

            tauqs   = G/sina*(diff(uy,1,2)/dx
                            + (1-2*cosa²)*diff(ux,1,1)/dy
                            + cosa*(movmean(duxdx,2,1,'discard')
                                    - movmean(duydy,2,2,'discard')))
            sigmaqs = (λ+2G)*diff(ux[2:Ny,:],1,2)/dx
                    + λ*diff(uy[:,2:Nx],1,1)/dy
                    - 2G*cosa*movmean(movmean(diff(ux,1,1)/dy,2,2,'discard'),2,1,'discard')
        """
        # ── Term 1: diff(uy, axis=1) / dx  →  shape (Ny, Nx) ──────────────
        term1 = np.diff(uy, axis=1) / dx                    # (Ny, Nx)

        # ── Term 2: (1-2cos²α) * diff(ux, axis=0) / dy  →  shape (Ny, Nx) ─
        term2 = (1 - 2 * cosa**2) * np.diff(ux, axis=0) / dy   # (Ny, Nx)

        # ── Term 3a: movmean(duxdx, 2, axis=0, 'discard')  →  (Ny, Nx) ────
        duxdx = np.gradient(ux, dx, axis=1)                 # (Ny+1, Nx)
        mm_duxdx = self._movmean_discard(duxdx, axis=0)          # (Ny,   Nx)  ✓

        # ── Term 3b: movmean(duydy, 2, axis=1, 'discard')  →  (Ny, Nx) ────
        duydy = np.gradient(uy, dy, axis=0)                 # (Ny, Nx+1)
        mm_duydy = self._movmean_discard(duydy, axis=1)          # (Ny, Nx) 

        # ── Assemble tauqs ──────────────────────────────────────────────────
        tauqs = G / sina * (term1 + term2 + cosa * (mm_duxdx - mm_duydy))  # (Ny, Nx)

        # Interpolate across the fault column (fault sits at mid)
        mid = Nx // 2    # 0-based centre column index
        tauqs[:, mid] = (tauqs[:, mid - 1] + tauqs[:, mid + 1]) / 2

        # ══ sigmaqs  (Ny-1, Nx-1) ═══════════════════════════════════════════
        s_term1 = np.diff(ux[1:Ny, :], axis=1) / dx         # (Ny-1, Nx-1)
        s_term2 = np.diff(uy[:, 1:Nx], axis=0) / dy         # (Ny-1, Nx-1)

        dux_dy     = np.diff(ux, axis=0) / dy                # (Ny,   Nx)
        mm_inner   = self._movmean_discard(dux_dy, axis=1)        # (Ny,   Nx-1)
        mm_outer   = self._movmean_discard(mm_inner, axis=0)      # (Ny-1, Nx-1)

        sigmaqs = ((lam + 2*G) * s_term1
                + lam       * s_term2
                - 2*G*cosa  * mm_outer)                   # (Ny-1, Nx-1)

        return tauqs, sigmaqs