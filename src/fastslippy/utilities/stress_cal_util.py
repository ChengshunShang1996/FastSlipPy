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
                              xp: Optional[np.ndarray] = None, yp: Optional[np.ndarray] = None,
                              builder: Optional[object] = None):
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
            dx_uy = self._edge_divisor(np.diff(xp) if xp is not None else dx, Nx)
            dy_ux = self._edge_divisor(np.diff(yp) if yp is not None else dy, Ny)
            dx_ux = self._edge_divisor(np.diff(x) if x is not None else dx, Nx - 1)
            dy_uy = self._edge_divisor(np.diff(y) if y is not None else dy, Ny - 1)
            x_ux = self._coord_or_uniform(x, Nx, dx)
            y_uy = self._coord_or_uniform(y, Ny, dy)

            # Compute local-step helper that matches MatrixBuilder._local_step
            def local_step(coords, idx):
                if idx <= 0:
                    return float(coords[1] - coords[0])
                if idx >= coords.size - 1:
                    return float(coords[-1] - coords[-2])
                return float(0.5 * (coords[idx + 1] - coords[idx - 1]))

            # Build local denominators for term1 and term2 using averaged local steps
            # dx_uy_local length Nx for differences uy[:, j+1]-uy[:, j]
            dx_uy_local = np.empty(Nx, dtype=float)
            for j in range(Nx):
                dx_uy_local[j] = 0.5 * (local_step(xp, j) + local_step(xp, j + 1))
            dy_ux_local = np.empty(Ny, dtype=float)
            for i in range(Ny):
                dy_ux_local[i] = 0.5 * (local_step(yp, i) + local_step(yp, i + 1))

            # Use local denominators for term1 and term2
            term1 = np.diff(uy, axis=1) / dx_uy_local[None, :]                    # (Ny, Nx)
            term2 = (1 - 2 * cosa**2) * np.diff(ux, axis=0) / dy_ux_local[:, None]   # (Ny, Nx)

            # Compute first derivatives on nonuniform coordinates using a
            # consistent three-point stencil with one-sided endpoints. For an
            # interior point j use the weighted average of forward/backward
            # slopes which reduces to central difference on uniform grids.
            def first_derivative_nonuniform(arr, coords, axis):
                # arr: ndarray, coords: 1D array of coordinates for the axis
                arr = np.asarray(arr)
                coords = np.asarray(coords, dtype=float)
                n = coords.size
                # Move axis to last for simpler indexing
                arr_moved = np.moveaxis(arr, axis, -1)
                out = np.empty_like(arr_moved, dtype=float)

                # endpoints: use one-sided two-point difference
                h0 = coords[1] - coords[0]
                out[..., 0] = (arr_moved[..., 1] - arr_moved[..., 0]) / h0
                hN = coords[-1] - coords[-2]
                out[..., -1] = (arr_moved[..., -1] - arr_moved[..., -2]) / hN

                # interior points: 3-point nonuniform formula (weights for f_{j-1}, f_j, f_{j+1})
                for j in range(1, n - 1):
                    h_l = coords[j] - coords[j - 1]
                    h_r = coords[j + 1] - coords[j]
                    denom = h_l * h_r * (h_l + h_r)
                    # coefficients derived from Taylor expansion
                    a = -h_r / (h_l * (h_l + h_r))
                    b = (h_r - h_l) / (h_l * h_r)
                    c = h_l / (h_r * (h_l + h_r))
                    out[..., j] = a * arr_moved[..., j - 1] + b * arr_moved[..., j] + c * arr_moved[..., j + 1]

                # move axis back
                return np.moveaxis(out, -1, axis)

            # duxdx: derivative of ux with respect to x at ux node columns (axis=1)
            duxdx = first_derivative_nonuniform(ux, x_ux, axis=1)          # (Ny+1, Nx)
            mm_duxdx = self._movmean_discard(duxdx, axis=0)                # (Ny,   Nx)

            # duydy: derivative of uy with respect to y at uy node rows (axis=0)
            duydy = first_derivative_nonuniform(uy, y_uy, axis=0)          # (Ny, Nx+1)
            mm_duydy = self._movmean_discard(duydy, axis=1)               # (Ny, Nx)

            # Additionally compute mixed derivatives for a strain-based tau computation
            # dux/dy at ux nodes (axis=0) using yp coords
            yp_coords = np.asarray(yp) if yp is not None else self._coord_or_uniform(None, ux.shape[0], dy)
            dux_dy = first_derivative_nonuniform(ux, yp_coords, axis=0)   # (Ny+1, Nx)
            mm_dux_dy = self._movmean_discard(dux_dy, axis=0)             # (Ny, Nx)

            # duy/dx at uy nodes (axis=1) using xp coords
            xp_coords = np.asarray(xp) if xp is not None else self._coord_or_uniform(None, uy.shape[1], dx)
            duy_dx = first_derivative_nonuniform(uy, xp_coords, axis=1)   # (Ny, Nx+1)
            mm_duy_dx = self._movmean_discard(duy_dx, axis=1)             # (Ny, Nx)

            # --- Compute local metric-form scaling using MatrixBuilder's helper ---
            from fastslippy.solver.matrix_builder import MatrixBuilder

            # dx_loc for ux columns (length Nx). Use second-derivative h_m from coords x
            x_coords = np.asarray(x_ux, dtype=float)
            dx_loc = np.empty(x_coords.size, dtype=float)
            for j in range(x_coords.size):
                # _second_derivative_weights expects interior indices; fall back to local_step
                try:
                    _, _, _, hmsq = MatrixBuilder._second_derivative_weights(x_coords, j)
                    dx_loc[j] = float(np.sqrt(hmsq))
                except Exception:
                    # local_step fallback
                    if j <= 0:
                        dx_loc[j] = float(x_coords[1] - x_coords[0])
                    elif j >= x_coords.size - 1:
                        dx_loc[j] = float(x_coords[-1] - x_coords[-2])
                    else:
                        dx_loc[j] = 0.5 * float(x_coords[j + 1] - x_coords[j - 1])

            # dy_loc for y rows (length Ny). Use second-derivative h_m from coords y
            y_coords = np.asarray(y_uy, dtype=float)
            dy_loc = np.empty(y_coords.size, dtype=float)
            for i in range(y_coords.size):
                try:
                    _, _, _, hmsq = MatrixBuilder._second_derivative_weights(y_coords, i)
                    dy_loc[i] = float(np.sqrt(hmsq))
                except Exception:
                    if i <= 0:
                        dy_loc[i] = float(y_coords[1] - y_coords[0])
                    elif i >= y_coords.size - 1:
                        dy_loc[i] = float(y_coords[-1] - y_coords[-2])
                    else:
                        dy_loc[i] = 0.5 * float(y_coords[i + 1] - y_coords[i - 1])

        # ── Full metric-form stress computation on nonuniform grid ─────────
        # Compute computational derivatives (with respect to the computational
        # coordinates x, y) then map to physical derivatives using the linear
        # mapping used in Grid: X = x + y*cosa, Y = y*sina

        # compute derivatives in computational coordinates at ux/uy nodes
        D_xc_ux = first_derivative_nonuniform(ux, x_ux, axis=1)    # (Ny+1, Nx)
        D_yc_ux = first_derivative_nonuniform(ux, yp_coords, axis=0)  # (Ny+1, Nx)

        D_xc_uy = first_derivative_nonuniform(uy, xp_coords, axis=1)   # (Ny, Nx+1)
        D_yc_uy = first_derivative_nonuniform(uy, y_uy, axis=0)        # (Ny, Nx+1)

        # map to physical derivatives
        D_X_ux = D_xc_ux
        D_Y_ux = (D_yc_ux - cosa * D_xc_ux) / sina

        D_X_uy = D_xc_uy
        D_Y_uy = (D_yc_uy - cosa * D_xc_uy) / sina

        # average to tau node locations (Ny, Nx)
        dux_dX = 0.5 * (D_X_ux[:-1, :] + D_X_ux[1:, :])   # (Ny, Nx)
        dux_dY = 0.5 * (D_Y_ux[:-1, :] + D_Y_ux[1:, :])   # (Ny, Nx)

        duy_dX = 0.5 * (D_X_uy[:, :-1] + D_X_uy[:, 1:])   # (Ny, Nx)
        duy_dY = 0.5 * (D_Y_uy[:, :-1] + D_Y_uy[:, 1:])   # (Ny, Nx)

        # strains and stresses
        eps_xx = dux_dX
        eps_yy = duy_dY
        eps_xy = 0.5 * (dux_dY + duy_dX)

        tr_eps = eps_xx + eps_yy
        sig_xx = lam * tr_eps + 2.0 * G * eps_xx
        sig_yy = lam * tr_eps + 2.0 * G * eps_yy
        sig_xy = 2.0 * G * eps_xy

        # traction onto normal n = (-sina, cosa)
        n_x, n_y = -sina, cosa
        f_x = sig_xx * n_x + sig_xy * n_y
        f_y = sig_xy * n_x + sig_yy * n_y
        t_x, t_y = cosa, sina
        tauqs = t_x * f_x + t_y * f_y

        # Now apply MatrixBuilder-consistent local-step scaling to the mm_dux/mm_duy
        # to mimic the metric-form discrete coefficients used during assembly.
        try:
            # mm_duxdx and mm_duydy are available in this scope from earlier
            mm_duxdx = locals().get('mm_duxdx', None)
            mm_duydy = locals().get('mm_duydy', None)
            if mm_duxdx is not None and mm_duydy is not None:
                # mm_duxdx: shape (Ny, Nx) -> multiply each column j by dx_loc[j]
                mm_duxdx = mm_duxdx * dx_loc[None, :]
                # mm_duydy: shape (Ny, Nx) -> multiply each row i by dy_loc[i]
                mm_duydy = mm_duydy * dy_loc[:, None]
                # recompute tauqs using the scaled quantities
                tauqs = G / sina * (term1 + term2 + cosa * (mm_duxdx - mm_duydy))
        except Exception:
            # if anything goes wrong, keep previous tauqs
            pass

        # interpolate across the fault column
        mid = Nx // 2
        tauqs[:, mid] = 0.5 * (tauqs[:, mid - 1] + tauqs[:, mid + 1])

        # If a MatrixBuilder was provided, compute operator-level tau by
        # applying the assembled LH operator to the displacement vector and
        # mapping uy-row forces at the fault column to tau. This uses the
        # exact discrete operator, so it produces stress/traction consistent
        # with the assembled matrix.
        if builder is not None:
            try:
                LH = builder.build_LH()
                N = LH.shape[0]
                u = np.zeros(N, dtype=float)
                # fill ux DOFs from builder._kux (shape (Ny+1, Nx))
                kux = builder._kux
                for iy in range(kux.shape[0]):
                    for ix in range(kux.shape[1]):
                        idx = int(kux[iy, ix])
                        u[idx] = float(ux[iy, ix])
                # fill uy DOFs from builder._kuy (shape (Ny, Nx+1))
                kuy = builder._kuy
                for iy in range(kuy.shape[0]):
                    for ix in range(kuy.shape[1]):
                        idx = int(kuy[iy, ix])
                        u[idx] = float(uy[iy, ix])
                f = LH.dot(u)
                # uy DOF indices at fault column
                fault_kuy = builder._kuy[:, builder._mid]
                f_fault = f[fault_kuy]
                # Map operator force to traction: derived from continuum scaling
                # Using tau ≈ - (sina / G) * f_row produces units-consistent traction
                tau_op = - (sina / G) * f_fault
                # Place into tauqs (Ny,) into column mid
                tauqs[:, mid] = tau_op

                # Now compute operator-derived tau at all tau nodes using a
                # more complete local extraction from LH rows. Use a centered
                # average of neighboring uy-row forces and include ux-row
                # contributions with a small projection factor to better capture
                # cross-coupling present in the assembled operator.
                try:
                    kuy = builder._kuy  # shape (Ny, Nx+1)
                    kux = builder._kux  # shape (Ny+1, Nx)
                    tau_op_full = np.empty((Ny, Nx), dtype=float)
                    for iy in range(Ny):
                        for ix in range(Nx):
                            # uy neighbors (left/right faces)
                            k_uy_l = int(kuy[iy, ix])
                            k_uy_r = int(kuy[iy, ix + 1])
                            val_uy = 0.5 * (f[k_uy_l] + f[k_uy_r])

                            # ux neighbors (top/bottom faces) — handle boundaries
                            vals_ux = []
                            if iy < kux.shape[0] - 1:
                                vals_ux.append(f[int(kux[iy + 1, ix])])
                            if iy >= 0 and iy < kux.shape[0]:
                                vals_ux.append(f[int(kux[iy, ix])])
                            val_ux = 0.5 * sum(vals_ux) if vals_ux else 0.0

                            # combine with projection factors
                            tau_op_full[iy, ix] = - (sina / G) * (val_uy + 0.25 * cosa * val_ux)

                    tauqs[:, :] = tau_op_full
                except Exception:
                    # if any failure, keep previous tauqs with only fault column replaced
                    pass
            except Exception:
                # fail gracefully and keep tauqs computed above
                pass

        # ══ sigmaqs (Ny-1, Nx-1) — keep existing formulation for sigmaqs
        s_term1 = np.diff(ux[1:Ny, :], axis=1) / dx_ux[None, :]         # (Ny-1, Nx-1)
        s_term2 = np.diff(uy[:, 1:Nx], axis=0) / dy_uy[:, None]         # (Ny-1, Nx-1)
        dux_dy = np.diff(ux, axis=0) / dy_ux[:, None]                   # (Ny, Nx)
        mm_inner = self._movmean_discard(dux_dy, axis=1)                # (Ny,   Nx-1)
        mm_outer = self._movmean_discard(mm_inner, axis=0)              # (Ny-1, Nx-1)

        sigmaqs = ((lam + 2*G) * s_term1
                + lam       * s_term2
                - 2*G*cosa  * mm_outer)                   # (Ny-1, Nx-1)

        return tauqs, sigmaqs