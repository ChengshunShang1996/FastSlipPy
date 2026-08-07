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
from scipy import sparse
import warnings

from fastslippy.pre_processing.model_parameters import ModelParameters, BCType
from fastslippy.pre_processing.grid import Grid


class MatrixBuilder:
    """
    Assembles the sparse stiffness matrix LH and right-hand-side vector RH
    for the quasi-static elastic equilibrium problem on the staggered grid.

    The stencil follows the original MATLAB build_LH / build_RH logic exactly.
    """

    def __init__(self, p: ModelParameters, grid: Grid):
        self.p    = p
        self.grid = grid
        if grid.is_nonuniform:
            if not p.allow_nonuniform_solver:
                raise ValueError(
                    "Nonuniform (stretched) mesh is enabled, but the nonuniform elastic operator "
                    "is still experimental and not yet C++-equivalent. "
                    "Set allow_nonuniform_solver=True to run it explicitly."
                )
            warnings.warn(
                "allow_nonuniform_solver=True: stretched-mesh operator is experimental and may "
                "produce inaccurate fields compared with the C++ implementation.",
                RuntimeWarning,
                stacklevel=2,
            )
        self._is_uniform = not (grid.is_nonuniform_x or grid.is_nonuniform_y)
        self._precompute_local_steps()
        self._precompute_rhs_layout()

    # Helper: global DOF indices
    @staticmethod
    def _dofs(ix: int, iy: int, Ny: int):
        """Return (kux, kuy) 0-based DOF indices for node (ix, iy)."""
        kux = ((ix) * (Ny + 1) + iy) * 2        # ux DOF (0-based)
        kuy = kux + 1                            # uy DOF (0-based)
        return kux, kuy

    @staticmethod
    def _local_step(coords: np.ndarray, idx: int) -> float:
        if idx <= 0:
            return float(coords[1] - coords[0])
        if idx >= coords.size - 1:
            return float(coords[-1] - coords[-2])
        return float(0.5 * (coords[idx + 1] - coords[idx - 1]))

    @staticmethod
    def _second_derivative_weights(coords: np.ndarray, idx: int):
        """
        Return (left, center, right, scale) for d2/dx2 on a nonuniform 1D grid.
        The `scale` is the squared local mean spacing and is used to keep the row
        scaling consistent with the legacy uniform formulation.
        """
        h_l = float(coords[idx] - coords[idx - 1])
        h_r = float(coords[idx + 1] - coords[idx])
        coeff_l = 2.0 / (h_l * (h_l + h_r))
        coeff_c = -2.0 / (h_l * h_r)
        coeff_r = 2.0 / (h_r * (h_l + h_r))
        h_m = 0.5 * (h_l + h_r)
        return coeff_l, coeff_c, coeff_r, h_m * h_m

    def _precompute_local_steps(self):
        p, g = self.p, self.grid
        Nx, Ny = p.Nx, p.Ny

        if self._is_uniform:
            dx = float(g.dx)
            dy = float(g.dy)
            self._dx_xuy = np.full(Nx + 1, dx)
            self._dy_yuy = np.full(Ny, dy)
            self._dx_xux = np.full(Nx, dx)
            self._dy_yux = np.full(Ny + 1, dy)
            return

        self._dx_xuy = np.array([self._local_step(g.xp, ix) for ix in range(Nx + 1)], dtype=float)
        self._dy_yuy = np.array([self._local_step(g.y, iy) for iy in range(Ny)], dtype=float)
        self._dx_xux = np.array([self._local_step(g.x, ix) for ix in range(Nx)], dtype=float)
        self._dy_yux = np.array([self._local_step(g.yp, iy) for iy in range(Ny + 1)], dtype=float)

    def _precompute_rhs_layout(self):
        p = self.p
        Nx, Ny = p.Nx, p.Ny

        self._mid = Nx // 2
        self._iy_int = np.arange(1, Ny - 1)  # interior iy for uy rows and ux interior rows

        self._ix_uy_bottom_top = np.arange(1, Nx)  # exclude left/right boundaries by precedence
        self._ix_uy_left = np.arange(1, self._mid)  # effective else branch ix <= mid with ix==mid excluded
        self._ix_uy_right = np.arange(self._mid + 2, Nx)  # effective else branch ix >= mid+2

        self._ix_ux_all = np.arange(Nx)
        self._ix_ux_left = np.arange(1, self._mid)  # effective else branch ix < mid+1 with ix==mid excluded
        self._ix_ux_right = np.arange(self._mid + 2, Nx - 1)  # effective else branch ix > mid+1

        self._kuy = np.empty((Ny, Nx + 1), dtype=int)
        self._kux = np.empty((Ny + 1, Nx), dtype=int)
        for ix in range(Nx + 1):
            for iy in range(Ny + 1):
                kux, kuy = self._dofs(ix, iy, Ny)
                if iy < Ny:
                    self._kuy[iy, ix] = kux + 1
                if ix < Nx:
                    self._kux[iy, ix] = kux

        case_type = p.case_type.value if hasattr(p.case_type, "value") else str(p.case_type)
        self._case_type = case_type.lower()
        self._rh = np.zeros(self.grid.N, dtype=float)

    def build_LH(self) -> sparse.csr_matrix:
        p, g = self.p, self.grid
        Nx, Ny, N = p.Nx, p.Ny, g.N
        lam, G   = p.lam, p.G
        sina, cosa = g.sina, g.cosa
        dx_xuy = self._dx_xuy
        dy_yuy = self._dy_yuy
        dx_xux = self._dx_xux
        dy_yux = self._dy_yux

        rows, cols, vals = [], [], []

        def add(r, c, v):
            rows.append(r); cols.append(c); vals.append(v)

        for ix in range(Nx+1):           # 0 … Nx  (MATLAB 1 … Nx+1)
            for iy in range(Ny+1):       # 0 … Ny

                kux, kuy = self._dofs(ix, iy, Ny)
                mid = (Nx) // 2            # fault column index (0-based)

                # ── uy equation (iy < Ny) ──────────────────────────────
                if iy < Ny:
                    dx_loc = dx_xuy[ix]
                    dy_loc = dy_yuy[iy]
                    if ix == 0: #left boundary
                        if p.bc.left.uy.type == BCType.FREE:
                            add(kuy, kuy, 1);  add(kuy, kuy + (Ny+1)*2, -1)
                        elif p.bc.left.uy.type == BCType.FIXED or p.bc.left.uy.type == BCType.VELOCITY:
                            add(kuy, kuy, 1)
                        else:
                            raise ValueError(f"Unknown BC type: {p.bc.left.uy.type}")
                    elif ix == Nx: #right boundary
                        if p.bc.right.uy.type == BCType.FREE:
                            add(kuy, kuy, 1);  add(kuy, kuy - (Ny+1)*2, -1)
                        elif p.bc.right.uy.type == BCType.FIXED or p.bc.right.uy.type == BCType.VELOCITY:
                            add(kuy, kuy, 1)
                        else:
                            raise ValueError(f"Unknown BC type: {p.bc.right.uy.type}")
                    elif iy == 0: #bottom boundary
                        if p.bc.bottom.uy.type == BCType.FREE:
                            #add(kuy, kuy, 1);  add(kuy, kuy + (Ny+1)*2, -1)
                            add(kuy, kuy, 1);  add(kuy, kuy + 2, -1)
                        elif p.bc.bottom.uy.type == BCType.FIXED or p.bc.bottom.uy.type == BCType.VELOCITY:
                            add(kuy, kuy, 1)
                        else:
                            raise ValueError(f"BC type: {p.bc.bottom.uy.type} is not supported for bottom boundary yet.")
                    elif iy == Ny - 1: #top boundary
                        if p.bc.top.uy.type == BCType.FIXED or p.bc.top.uy.type == BCType.VELOCITY:
                            add(kuy, kuy, 1)
                        elif p.bc.top.uy.type == BCType.FREE:
                            #add(kuy, kuy, 1);  add(kuy, kuy - (Ny+1)*2, -1)
                            add(kuy, kuy, 1);  add(kuy, kuy - 2, -1)
                        else:
                            raise ValueError(f"BC type: {p.bc.top.uy.type} is not supported for top boundary yet.")
                    elif ix == mid:
                        # Fault left side
                        add(kuy, kuy, -1); add(kuy, kuy + (Ny+1)*2, 1)
                    elif ix == mid + 1:
                        # Fault right side
                        kux_n, kuy_n = self._dofs(ix, iy, Ny)
                        add(kuy, kuy - 2*(Ny+1)*2, 1)
                        add(kuy, kuy - (Ny+1)*2,  -1)
                        add(kuy, kuy,             -1)
                        add(kuy, kuy + (Ny+1)*2,   1)
                        # Cross-coupling terms with ux
                        add(kuy, kux + (Ny+1)*2,       cosa/4)
                        add(kuy, kux + (Ny+1)*2 + 2,   cosa/4)
                        add(kuy, kux - (Ny+1)*2,       -cosa/2)
                        add(kuy, kux - (Ny+1)*2 + 2,   -cosa/2)
                        add(kuy, kux - 3*(Ny+1)*2,     cosa/4)
                        add(kuy, kux - 3*(Ny+1)*2 + 2, cosa/4)
                        add(kuy, kuy + (Ny+1)*2 - 2,   cosa/4/dy_loc*dx_loc)
                        add(kuy, kuy + (Ny+1)*2 + 2,  -cosa/4/dy_loc*dx_loc)
                        add(kuy, kuy - 2,               cosa/4/dy_loc*dx_loc)
                        add(kuy, kuy + 2,              -cosa/4/dy_loc*dx_loc)
                        add(kuy, kuy - (Ny+1)*2 - 2,  -cosa/4/dy_loc*dx_loc)
                        add(kuy, kuy - (Ny+1)*2 + 2,   cosa/4/dy_loc*dx_loc)
                        add(kuy, kuy - 2*(Ny+1)*2 - 2,  -cosa/4/dy_loc*dx_loc)
                        add(kuy, kuy - 2*(Ny+1)*2 + 2,   cosa/4/dy_loc*dx_loc)
                    else:
                        # Interior bulk
                        if self._is_uniform:
                            r2 = dx_loc*dx_loc / dy_loc/dy_loc * (lam + 2*G) / G
                            add(kuy, kuy, -2 - 2*r2)
                            add(kuy, kuy - (Ny+1)*2, 1)
                            add(kuy, kuy + (Ny+1)*2, 1)
                            add(kuy, kuy - 2,  r2)
                            add(kuy, kuy + 2,  r2)
                        else:
                            cxl, cxc, cxr, sx = self._second_derivative_weights(g.xp, ix)
                            cyl, cyc, cyr, _ = self._second_derivative_weights(g.y, iy)
                            a2 = (lam + 2 * G) / G
                            add(kuy, kuy, sx * (cxc + a2 * cyc))
                            add(kuy, kuy - (Ny+1)*2, sx * cxl)
                            add(kuy, kuy + (Ny+1)*2, sx * cxr)
                            add(kuy, kuy - 2, sx * a2 * cyl)
                            add(kuy, kuy + 2, sx * a2 * cyr)
                            dx_loc = np.sqrt(sx)
                        c_val = cosa/dy_loc*dx_loc*(lam + 3*G)/G/4
                        add(kuy, kuy + (Ny+1)*2 - 2,   c_val)
                        add(kuy, kuy + (Ny+1)*2 + 2,  -c_val)
                        add(kuy, kuy - (Ny+1)*2 - 2,  -c_val)
                        add(kuy, kuy - (Ny+1)*2 + 2,   c_val)
                        fac = 1/dy_loc*dx_loc*(lam + G)/G
                        if ix == 1 or ix == Nx - 1:
                            add(kuy, kux - (Ny+1)*2,      fac)
                            add(kuy, kux - (Ny+1)*2 + 2, -fac)
                            add(kuy, kux,                 -fac)
                            add(kuy, kux + 2,              fac)
                        else:
                            cf = cosa*(lam + G)/G/4
                            add(kuy, kux - (Ny+1)*2,      fac + cf)
                            add(kuy, kux - (Ny+1)*2 + 2, -fac + cf)
                            add(kuy, kux,                 -fac + cf)
                            add(kuy, kux + 2,              fac + cf)
                            add(kuy, kux - 2*(Ny+1)*2,    -cf)
                            add(kuy, kux - 2*(Ny+1)*2+2,  -cf)
                            add(kuy, kux + (Ny+1)*2,      -cf)
                            add(kuy, kux + (Ny+1)*2 + 2,  -cf)
                else:
                    # Neumann BC for ghost uy nodes at iy=Ny
                    add(kuy, kuy, 1)

                # ── ux equation (ix < Nx) ──────────────────────────────
                if ix < Nx:
                    dx_loc = dx_xux[ix]
                    dy_loc = dy_yux[iy]
                    r2 = dx_loc*dx_loc / dy_loc/dy_loc
                    r_lam = (lam + 2*G) / G
                    if iy == 0: #bottom boundary
                        if p.bc.bottom.ux.type == BCType.FIXED or p.bc.bottom.ux.type == BCType.VELOCITY:
                            add(kux, kux, 1)
                        elif p.bc.bottom.ux.type == BCType.FREE:
                            add(kux, kux, 1); add(kux, kux + 2, -1)
                        else:
                            raise ValueError(f"Unknown BC type: {p.bc.bottom.ux.type}")
                    elif iy == Ny:
                        if p.bc.top.ux.type == BCType.FIXED or p.bc.top.ux.type == BCType.VELOCITY:
                            add(kux, kux, 1)
                        elif p.bc.top.ux.type == BCType.FREE:
                            add(kux, kux, 1); add(kux, kux - 2, -1)
                        else:
                            raise ValueError(f"Unknown BC type: {p.bc.top.ux.type}")
                    elif ix == 0:
                        if p.bc.left.ux.type == BCType.FIXED or p.bc.left.ux.type == BCType.VELOCITY:
                            add(kux, kux, 1)
                        elif p.bc.left.ux.type == BCType.FREE:
                            add(kux, kux, 1); add(kux, kux + (Ny+1)*2, -1)
                        else:
                            raise ValueError(f"BC type: {p.bc.left.ux.type} is not supported for left boundary yet.")
                    elif ix == Nx - 1:
                        if p.bc.right.ux.type == BCType.FIXED or p.bc.right.ux.type == BCType.VELOCITY:
                            add(kux, kux, 1)
                        elif p.bc.right.ux.type == BCType.FREE:
                            add(kux, kux, 1); add(kux, kux - (Ny+1)*2, -1)
                        else:
                            raise ValueError(f"BC type: {p.bc.right.ux.type} is not supported for right boundary yet.")
                    elif ix == mid:
                        # Fault column – normal stress jump condition
                        if self._is_uniform:
                            add(kux, kux,              -2*r_lam)
                            add(kux, kux + (Ny+1)*2,   r_lam)
                            add(kux, kux - (Ny+1)*2,   r_lam)
                        else:
                            cxl, cxc, cxr, sx = self._second_derivative_weights(g.x, ix)
                            add(kux, kux, sx * r_lam * cxc)
                            add(kux, kux + (Ny+1)*2, sx * r_lam * cxr)
                            add(kux, kux - (Ny+1)*2, sx * r_lam * cxl)
                            dx_loc = np.sqrt(sx)
                        fac = lam/G/dy_loc*dx_loc
                        add(kux, kuy,                    -fac)
                        add(kux, kuy + (Ny+1)*2,          fac)
                        add(kux, kuy - 2,                 fac)
                        add(kux, kuy + (Ny+1)*2 - 2,     -fac)
                    else:
                        # Interior bulk
                        if self._is_uniform:
                            add(kux, kux, -2*r_lam - 2*r2)
                            add(kux, kux - (Ny+1)*2, r_lam)
                            add(kux, kux + (Ny+1)*2, r_lam)
                            add(kux, kux - 2, r2)
                            add(kux, kux + 2, r2)
                        else:
                            cxl, cxc, cxr, sx = self._second_derivative_weights(g.x, ix)
                            cyl, cyc, cyr, _ = self._second_derivative_weights(g.yp, iy)
                            add(kux, kux, sx * (r_lam * cxc + cyc))
                            add(kux, kux - (Ny+1)*2, sx * r_lam * cxl)
                            add(kux, kux + (Ny+1)*2, sx * r_lam * cxr)
                            add(kux, kux - 2, sx * cyl)
                            add(kux, kux + 2, sx * cyr)
                            dx_loc = np.sqrt(sx)
                        c_val = cosa/dy_loc*dx_loc*(lam + 3*G)/G/4
                        add(kux, kux + (Ny+1)*2 - 2,   c_val)
                        add(kux, kux + (Ny+1)*2 + 2,  -c_val)
                        add(kux, kux - (Ny+1)*2 - 2,  -c_val)
                        add(kux, kux - (Ny+1)*2 + 2,   c_val)
                        fac = 1/dy_loc*dx_loc*(lam + G)/G
                        if iy == 1 or iy == Ny - 1:
                            add(kux, kuy + (Ny+1)*2,      fac)
                            add(kux, kuy + (Ny+1)*2 - 2, -fac)
                            add(kux, kuy,                 -fac)
                            add(kux, kuy - 2,              fac)
                        else:
                            cf = cosa/dy_loc/dy_loc*dx_loc*dx_loc*(lam + G)/G/4
                            add(kux, kuy + (Ny+1)*2,        fac + cf)
                            add(kux, kuy + (Ny+1)*2 - 2,   -fac + cf)
                            add(kux, kuy,                   -fac + cf)
                            add(kux, kuy - 2,                fac + cf)
                            add(kux, kuy + (Ny+1)*2 + 2,   -cf)
                            add(kux, kuy + (Ny+1)*2 - 4,   -cf)
                            add(kux, kuy + 2,               -cf)
                            add(kux, kuy - 4,               -cf)
                else:
                    # Neumann BC for ghost ux nodes at ix=Nx
                    add(kux, kux, 1)

        LH = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))

        return LH

    def build_RH(self, dPdt: float, V: np.ndarray) -> np.ndarray:
        p, g = self.p, self.grid
        Nx, Ny = p.Nx, p.Ny
        G = p.G
        sina, cosa = g.sina, g.cosa
        y = g.y
        mid = self._mid
        dx_xuy = self._dx_xuy
        dy_yuy = self._dy_yuy
        dx_xux = self._dx_xux
        dy_yux = self._dy_yux
        RH = self._rh
        RH.fill(0.0)

        is_lab = self._case_type == "lab"
        is_california = self._case_type == "california"
        is_groningen = self._case_type == "groningen"

        # --- uy block (exact branch priority) ---
        if p.bc.left.uy.type == BCType.VELOCITY:
            RH[self._kuy[:, 0]] = p.bc.left.uy.value
        if p.bc.right.uy.type == BCType.VELOCITY:
            RH[self._kuy[:, Nx]] = p.bc.right.uy.value

        if p.bc.bottom.uy.type == BCType.VELOCITY:
            if is_lab:
                RH[self._kuy[0, mid + 1:Nx]] = p.bc.bottom.uy.value
            else:
                RH[self._kuy[0, self._ix_uy_bottom_top]] = p.bc.bottom.uy.value

        if p.bc.top.uy.type == BCType.VELOCITY:
            if is_lab:
                RH[self._kuy[Ny - 1, mid + 1:Nx]] = p.bc.top.uy.value
            elif is_california:
                RH[self._kuy[Ny - 1, 1:mid]] = -p.bc.top.uy.value
                RH[self._kuy[Ny - 1, mid + 1:Nx]] = p.bc.top.uy.value
            else:
                RH[self._kuy[Ny - 1, self._ix_uy_bottom_top]] = p.bc.top.uy.value

        iy_int = self._iy_int
        if is_california:
            cal_mask = y[iy_int] >= p.W_f
            fault_idx = self._kuy[iy_int, mid]
            RH[fault_idx[~cal_mask]] = V[iy_int[~cal_mask]]
            RH[fault_idx[cal_mask]] = p.loading.V_L
        else:
            RH[self._kuy[iy_int, mid]] = V[iy_int]

        if is_groningen:
            y_int = y[iy_int]
            dy_uy_int = dy_yuy[iy_int]

            mask_850 = y_int == 850
            mask_1050 = y_int == 1050
            mask_800 = y_int == 800
            mask_1000 = y_int == 1000

            if np.any(mask_850):
                for iy, dy_loc in zip(iy_int[mask_850], dy_uy_int[mask_850]):
                    ix = self._ix_uy_right
                    RH[self._kuy[iy, ix]] = dPdt / dy_loc * dx_xuy[ix] * dx_xuy[ix] / G * sina
            if np.any(mask_1050):
                for iy, dy_loc in zip(iy_int[mask_1050], dy_uy_int[mask_1050]):
                    ix = self._ix_uy_right
                    RH[self._kuy[iy, ix]] = -dPdt / dy_loc * dx_xuy[ix] * dx_xuy[ix] / G * sina
            if np.any(mask_800):
                for iy, dy_loc in zip(iy_int[mask_800], dy_uy_int[mask_800]):
                    ix = self._ix_uy_left
                    RH[self._kuy[iy, ix]] = dPdt / dy_loc * dx_xuy[ix] * dx_xuy[ix] / G * sina
            if np.any(mask_1000):
                for iy, dy_loc in zip(iy_int[mask_1000], dy_uy_int[mask_1000]):
                    ix = self._ix_uy_left
                    RH[self._kuy[iy, ix]] = -dPdt / dy_loc * dx_xuy[ix] * dx_xuy[ix] / G * sina

        # --- ux block (exact branch priority) ---
        if p.bc.bottom.ux.type == BCType.VELOCITY:
            RH[self._kux[0, self._ix_ux_all]] = p.bc.bottom.ux.value
        if p.bc.top.ux.type == BCType.VELOCITY:
            if is_california:
                RH[self._kux[Ny, :mid]] = -p.bc.top.ux.value
                RH[self._kux[Ny, mid + 1:]] = p.bc.top.ux.value
            else:
                RH[self._kux[Ny, self._ix_ux_all]] = p.bc.top.ux.value

        if p.bc.left.ux.type == BCType.VELOCITY:
            RH[self._kux[1:Ny, 0]] = p.bc.left.ux.value
        if p.bc.right.ux.type == BCType.VELOCITY:
            RH[self._kux[1:Ny, Nx - 1]] = p.bc.right.ux.value

        if is_groningen:
            y_int = y[iy_int]
            dx_mid = dx_xux[mid]
            RH[self._kux[iy_int[(y_int > 800) & (y_int <= 850)], mid]] = -dPdt * dx_mid / G
            RH[self._kux[iy_int[(y_int > 1000) & (y_int <= 1050)], mid]] = dPdt * dx_mid / G

            dy_ux_int = dy_yux[iy_int]
            mask_1050 = y_int == 1050
            mask_1000 = y_int == 1000
            mask_850 = y_int == 850
            mask_800 = y_int == 800

            if np.any(mask_1050):
                for iy, dy_loc in zip(iy_int[mask_1050], dy_ux_int[mask_1050]):
                    ix = self._ix_ux_right
                    RH[self._kux[iy, ix]] = dPdt / dy_loc * dx_xux[ix] * dx_xux[ix] / G * sina * cosa
            if np.any(mask_1000):
                for iy, dy_loc in zip(iy_int[mask_1000], dy_ux_int[mask_1000]):
                    ix = self._ix_ux_left
                    RH[self._kux[iy, ix]] = dPdt / dy_loc * dx_xux[ix] * dx_xux[ix] / G * sina * cosa
            if np.any(mask_850):
                for iy, dy_loc in zip(iy_int[mask_850], dy_ux_int[mask_850]):
                    ix = self._ix_ux_right
                    RH[self._kux[iy, ix]] = -dPdt / dy_loc * dx_xux[ix] * dx_xux[ix] / G * sina * cosa
            if np.any(mask_800):
                for iy, dy_loc in zip(iy_int[mask_800], dy_ux_int[mask_800]):
                    ix = self._ix_ux_left
                    RH[self._kux[iy, ix]] = -dPdt / dy_loc * dx_xux[ix] * dx_xux[ix] / G * sina * cosa
        return RH