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
from typing import Optional

from fastslippy.utilities.grid_operators import (
    RecoveryOperators,
    build_recovery_operators,
)

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    njit = None
    _HAS_NUMBA = False


if _HAS_NUMBA:
    @njit(cache=True)
    def _compute_stress_fields_uniform_numba(uy, ux, dx, dy, lam, G, cosa, sina):
        ny = uy.shape[0]
        nx = uy.shape[1] - 1

        term1 = np.empty((ny, nx), dtype=np.float64)
        term2 = np.empty((ny, nx), dtype=np.float64)
        duxdx = np.empty((ny + 1, nx), dtype=np.float64)
        mm_duxdx = np.empty((ny, nx), dtype=np.float64)
        duydy = np.empty((ny, nx + 1), dtype=np.float64)
        mm_duydy = np.empty((ny, nx), dtype=np.float64)
        tauqs = np.empty((ny, nx), dtype=np.float64)

        s_term1 = np.empty((ny - 1, nx - 1), dtype=np.float64)
        s_term2 = np.empty((ny - 1, nx - 1), dtype=np.float64)
        dux_dy = np.empty((ny, nx), dtype=np.float64)
        mm_inner = np.empty((ny, nx - 1), dtype=np.float64)
        mm_outer = np.empty((ny - 1, nx - 1), dtype=np.float64)
        sigmaqs = np.empty((ny - 1, nx - 1), dtype=np.float64)

        inv_dx = 1.0 / dx
        inv_2dx = 0.5 / dx
        inv_dy = 1.0 / dy
        inv_2dy = 0.5 / dy

        for i in range(ny):
            for j in range(nx):
                term1[i, j] = (uy[i, j + 1] - uy[i, j]) * inv_dx

        c2 = (1.0 - 2.0 * cosa * cosa)
        for i in range(ny):
            for j in range(nx):
                term2[i, j] = c2 * (ux[i + 1, j] - ux[i, j]) * inv_dy

        for i in range(ny + 1):
            duxdx[i, 0] = (ux[i, 1] - ux[i, 0]) * inv_dx
            for j in range(1, nx - 1):
                duxdx[i, j] = (ux[i, j + 1] - ux[i, j - 1]) * inv_2dx
            duxdx[i, nx - 1] = (ux[i, nx - 1] - ux[i, nx - 2]) * inv_dx

        for i in range(ny):
            for j in range(nx):
                mm_duxdx[i, j] = 0.5 * (duxdx[i, j] + duxdx[i + 1, j])

        for j in range(nx + 1):
            duydy[0, j] = (uy[1, j] - uy[0, j]) * inv_dy
            for i in range(1, ny - 1):
                duydy[i, j] = (uy[i + 1, j] - uy[i - 1, j]) * inv_2dy
            duydy[ny - 1, j] = (uy[ny - 1, j] - uy[ny - 2, j]) * inv_dy

        for i in range(ny):
            for j in range(nx):
                mm_duydy[i, j] = 0.5 * (duydy[i, j] + duydy[i, j + 1])

        coef = G / sina
        for i in range(ny):
            for j in range(nx):
                tauqs[i, j] = coef * (term1[i, j] + term2[i, j] + cosa * (mm_duxdx[i, j] - mm_duydy[i, j]))

        mid = nx // 2
        for i in range(ny):
            tauqs[i, mid] = 0.5 * (tauqs[i, mid - 1] + tauqs[i, mid + 1])

        for i in range(ny - 1):
            for j in range(nx - 1):
                s_term1[i, j] = (ux[i + 1, j + 1] - ux[i + 1, j]) * inv_dx
                s_term2[i, j] = (uy[i + 1, j + 1] - uy[i, j + 1]) * inv_dy

        for i in range(ny):
            for j in range(nx):
                dux_dy[i, j] = (ux[i + 1, j] - ux[i, j]) * inv_dy

        for i in range(ny):
            for j in range(nx - 1):
                mm_inner[i, j] = 0.5 * (dux_dy[i, j] + dux_dy[i, j + 1])

        for i in range(ny - 1):
            for j in range(nx - 1):
                mm_outer[i, j] = 0.5 * (mm_inner[i, j] + mm_inner[i + 1, j])

        for i in range(ny - 1):
            for j in range(nx - 1):
                sigmaqs[i, j] = ((lam + 2.0 * G) * s_term1[i, j]
                                 + lam * s_term2[i, j]
                                 - 2.0 * G * cosa * mm_outer[i, j])
        return tauqs, sigmaqs


class StressCalUtil:
    """
    Utility functions for stress calculations.
    """

    def __init__(self, prefer_numba: bool = True):
        self._use_numba = bool(prefer_numba and _HAS_NUMBA)
        self._workspace = {}
        self._recovery_cache = {}

    def _recovery_operators(self, x, y, xp, yp) -> RecoveryOperators:
        arrays = tuple(np.asarray(value, dtype=float) for value in (x, y, xp, yp))
        key = tuple((id(value), value.size) for value in arrays)
        operators = self._recovery_cache.get(key)
        if operators is None:
            operators = build_recovery_operators(*arrays)
            self._recovery_cache[key] = operators
        return operators

    def _movmean_discard(self, arr: np.ndarray, axis: int) -> np.ndarray:
        """Running mean of adjacent pairs along *axis*, discarding endpoints."""
        sl_a = [slice(None)] * arr.ndim
        sl_b = [slice(None)] * arr.ndim
        sl_a[axis] = slice(None, -1)
        sl_b[axis] = slice(1, None)
        return (arr[tuple(sl_a)] + arr[tuple(sl_b)]) / 2

    @staticmethod
    def _edge_divisor(spacing, n):
        if np.isscalar(spacing):
            return float(spacing) * np.ones(n)
        arr = np.asarray(spacing, dtype=float)
        if arr.ndim != 1 or arr.size != n:
            raise ValueError(f"Expected spacing array of shape ({n},), got {arr.shape}.")
        return arr

    @staticmethod
    def _coord_or_uniform(coord: Optional[np.ndarray], n: int, step) -> np.ndarray:
        if coord is not None:
            arr = np.asarray(coord, dtype=float)
            if arr.ndim != 1 or arr.size != n:
                raise ValueError(f"Expected coordinate array of shape ({n},), got {arr.shape}.")
            return arr
        if np.isscalar(step):
            return np.arange(n, dtype=float) * float(step)
        raise ValueError("Nonuniform spacing requires explicit coordinate arrays.")

    def _get_workspace(self, Ny: int, Nx: int):
        key = (Ny, Nx)
        if key not in self._workspace:
            self._workspace[key] = {
                "term1": np.empty((Ny, Nx), dtype=float),
                "term2": np.empty((Ny, Nx), dtype=float),
                "duxdx": np.empty((Ny + 1, Nx), dtype=float),
                "mm_duxdx": np.empty((Ny, Nx), dtype=float),
                "duydy": np.empty((Ny, Nx + 1), dtype=float),
                "mm_duydy": np.empty((Ny, Nx), dtype=float),
                "tauqs": np.empty((Ny, Nx), dtype=float),
                "s_term1": np.empty((Ny - 1, Nx - 1), dtype=float),
                "s_term2": np.empty((Ny - 1, Nx - 1), dtype=float),
                "dux_dy": np.empty((Ny, Nx), dtype=float),
                "mm_inner": np.empty((Ny, Nx - 1), dtype=float),
                "mm_outer": np.empty((Ny - 1, Nx - 1), dtype=float),
                "sigmaqs": np.empty((Ny - 1, Nx - 1), dtype=float),
            }
        return self._workspace[key]

    def compute_stress_fields(self, uy, ux, dx, dy, lam, G, cosa, sina, Ny, Nx,
                              x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
                              xp: Optional[np.ndarray] = None, yp: Optional[np.ndarray] = None):
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
        use_uniform_fast_path = (
            x is None and y is None and xp is None and yp is None
            and np.isscalar(dx) and np.isscalar(dy)
        )

        if use_uniform_fast_path:
            if self._use_numba:
                return _compute_stress_fields_uniform_numba(uy, ux, dx, dy, lam, G, cosa, sina)

            ws = self._get_workspace(Ny, Nx)
            term1 = ws["term1"]
            term2 = ws["term2"]
            duxdx = ws["duxdx"]
            mm_duxdx = ws["mm_duxdx"]
            duydy = ws["duydy"]
            mm_duydy = ws["mm_duydy"]
            tauqs = ws["tauqs"]
            s_term1 = ws["s_term1"]
            s_term2 = ws["s_term2"]
            dux_dy = ws["dux_dy"]
            mm_inner = ws["mm_inner"]
            mm_outer = ws["mm_outer"]
            sigmaqs = ws["sigmaqs"]

            np.subtract(uy[:, 1:], uy[:, :-1], out=term1)
            term1 /= dx

            np.subtract(ux[1:, :], ux[:-1, :], out=term2)
            term2 *= (1 - 2 * cosa**2) / dy

            duxdx[:, 0] = (ux[:, 1] - ux[:, 0]) / dx
            duxdx[:, -1] = (ux[:, -1] - ux[:, -2]) / dx
            duxdx[:, 1:-1] = (ux[:, 2:] - ux[:, :-2]) / (2 * dx)
            np.add(duxdx[:-1, :], duxdx[1:, :], out=mm_duxdx)
            mm_duxdx *= 0.5

            duydy[0, :] = (uy[1, :] - uy[0, :]) / dy
            duydy[-1, :] = (uy[-1, :] - uy[-2, :]) / dy
            duydy[1:-1, :] = (uy[2:, :] - uy[:-2, :]) / (2 * dy)
            np.add(duydy[:, :-1], duydy[:, 1:], out=mm_duydy)
            mm_duydy *= 0.5

            tauqs[:, :] = G / sina * (term1 + term2 + cosa * (mm_duxdx - mm_duydy))
            mid = Nx // 2
            tauqs[:, mid] = 0.5 * (tauqs[:, mid - 1] + tauqs[:, mid + 1])

            np.subtract(ux[1:Ny, 1:], ux[1:Ny, :-1], out=s_term1)
            s_term1 /= dx
            np.subtract(uy[1:, 1:Nx], uy[:-1, 1:Nx], out=s_term2)
            s_term2 /= dy

            np.subtract(ux[1:, :], ux[:-1, :], out=dux_dy)
            dux_dy /= dy
            np.add(dux_dy[:, :-1], dux_dy[:, 1:], out=mm_inner)
            mm_inner *= 0.5
            np.add(mm_inner[:-1, :], mm_inner[1:, :], out=mm_outer)
            mm_outer *= 0.5

            sigmaqs[:, :] = ((lam + 2 * G) * s_term1
                             + lam * s_term2
                             - 2 * G * cosa * mm_outer)

            return tauqs, sigmaqs
        else:
            if any(value is None for value in (x, y, xp, yp)):
                raise ValueError(
                    "MATLAB-equivalent stress recovery requires x, y, xp, and yp."
                )
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            xp = np.asarray(xp, dtype=float)
            yp = np.asarray(yp, dtype=float)
            recovery = self._recovery_operators(x, y, xp, yp)

            dx_uy = self._edge_divisor(np.diff(xp), Nx)
            dy_ux = self._edge_divisor(np.diff(yp), Ny)
            dx_ux = self._edge_divisor(np.diff(x), Nx - 1)
            dy_uy = self._edge_divisor(np.diff(y), Ny - 1)

            term1 = np.diff(uy, axis=1) / dx_uy[None, :]
            term2 = (
                (1 - 2 * cosa**2)
                * np.diff(ux, axis=0)
                / dy_ux[:, None]
            )
            duxdx_nodes = (recovery.derivative_x @ ux.T).T
            mm_duxdx = recovery.midpoint_y_to_node @ duxdx_nodes
            duydy_nodes = recovery.derivative_y @ uy
            mm_duydy = (
                recovery.midpoint_x_to_node @ duydy_nodes.T
            ).T

        # ── Assemble tauqs ──────────────────────────────────────────────────
        tauqs = G / sina * (term1 + term2 + cosa * (mm_duxdx - mm_duydy))  # (Ny, Nx)

        # Interpolate across the fault column (fault sits at mid)
        mid = Nx // 2    # 0-based centre column index
        tauqs[:, mid] = (tauqs[:, mid - 1] + tauqs[:, mid + 1]) / 2

        # ══ sigmaqs  (Ny-1, Nx-1) ═══════════════════════════════════════════
        s_term1 = np.diff(ux[1:Ny, :], axis=1) / dx_ux[None, :]         # (Ny-1, Nx-1)
        s_term2 = np.diff(uy[:, 1:Nx], axis=0) / dy_uy[:, None]         # (Ny-1, Nx-1)
        dux_dy = np.diff(ux, axis=0) / dy_ux[:, None]                   # (Ny, Nx)
        mm_inner   = self._movmean_discard(dux_dy, axis=1)        # (Ny,   Nx-1)
        mm_outer   = self._movmean_discard(mm_inner, axis=0)      # (Ny-1, Nx-1)

        sigmaqs = ((lam + 2*G) * s_term1
                + lam       * s_term2
                - 2*G*cosa  * mm_outer)                   # (Ny-1, Nx-1)

        return tauqs, sigmaqs

    def recover_fault_normal_stress(
        self,
        sigmaqs: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        xp: np.ndarray,
        yp: np.ndarray,
        left_column: int,
        right_column: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Recover normal stress on both fault faces at every fault node."""

        recovery = self._recovery_operators(x, y, xp, yp)
        left = recovery.sigma_centres_to_nodes @ sigmaqs[:, left_column]
        right = recovery.sigma_centres_to_nodes @ sigmaqs[:, right_column]
        return np.asarray(left), np.asarray(right)
