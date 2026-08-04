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

    # Helper: global DOF indices
    @staticmethod
    def _dofs(ix: int, iy: int, Ny: int):
        """Return (kux, kuy) 0-based DOF indices for node (ix, iy)."""
        kux = ((ix) * (Ny + 1) + iy) * 2        # ux DOF (0-based)
        kuy = kux + 1                            # uy DOF (0-based)
        return kux, kuy

    def build_LH(self) -> sparse.csr_matrix:
        p, g = self.p, self.grid
        Nx, Ny, N = p.Nx, p.Ny, g.N
        dx, dy   = g.dx, g.dy
        lam, G   = p.lam, p.G
        sina, cosa = g.sina, g.cosa

        rows, cols, vals = [], [], []

        def add(r, c, v):
            rows.append(r); cols.append(c); vals.append(v)

        for ix in range(Nx+1):           # 0 … Nx  (MATLAB 1 … Nx+1)
            for iy in range(Ny+1):       # 0 … Ny

                kux, kuy = self._dofs(ix, iy, Ny)
                mid = (Nx) // 2            # fault column index (0-based)

                # ── uy equation (iy < Ny) ──────────────────────────────
                if iy < Ny:
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
                        add(kuy, kuy + (Ny+1)*2 - 2,   cosa/4/dy*dx)
                        add(kuy, kuy + (Ny+1)*2 + 2,  -cosa/4/dy*dx)
                        add(kuy, kuy - 2,               cosa/4/dy*dx)
                        add(kuy, kuy + 2,              -cosa/4/dy*dx)
                        add(kuy, kuy - (Ny+1)*2 - 2,  -cosa/4/dy*dx)
                        add(kuy, kuy - (Ny+1)*2 + 2,   cosa/4/dy*dx)
                        add(kuy, kuy - 2*(Ny+1)*2 - 2,  -cosa/4/dy*dx)
                        add(kuy, kuy - 2*(Ny+1)*2 + 2,   cosa/4/dy*dx)
                    else:
                        # Interior bulk
                        r2 = dx*dx / dy/dy * (lam + 2*G) / G
                        add(kuy, kuy, -2 - 2*r2)
                        add(kuy, kuy - (Ny+1)*2, 1)
                        add(kuy, kuy + (Ny+1)*2, 1)
                        add(kuy, kuy - 2,  r2)
                        add(kuy, kuy + 2,  r2)
                        c_val = cosa/dy*dx*(lam + 3*G)/G/4
                        add(kuy, kuy + (Ny+1)*2 - 2,   c_val)
                        add(kuy, kuy + (Ny+1)*2 + 2,  -c_val)
                        add(kuy, kuy - (Ny+1)*2 - 2,  -c_val)
                        add(kuy, kuy - (Ny+1)*2 + 2,   c_val)
                        fac = 1/dy*dx*(lam + G)/G
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
                    r2 = dx*dx / dy/dy
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
                        add(kux, kux,              -2*r_lam)
                        add(kux, kux + (Ny+1)*2,   r_lam)
                        add(kux, kux - (Ny+1)*2,   r_lam)
                        fac = lam/G/dy*dx
                        add(kux, kuy,                    -fac)
                        add(kux, kuy + (Ny+1)*2,          fac)
                        add(kux, kuy - 2,                 fac)
                        add(kux, kuy + (Ny+1)*2 - 2,     -fac)
                    else:
                        # Interior bulk
                        add(kux, kux, -2*r_lam - 2*r2)
                        add(kux, kux - (Ny+1)*2, r_lam)
                        add(kux, kux + (Ny+1)*2, r_lam)
                        add(kux, kux - 2, r2)
                        add(kux, kux + 2, r2)
                        c_val = cosa/dy*dx*(lam + 3*G)/G/4
                        add(kux, kux + (Ny+1)*2 - 2,   c_val)
                        add(kux, kux + (Ny+1)*2 + 2,  -c_val)
                        add(kux, kux - (Ny+1)*2 - 2,  -c_val)
                        add(kux, kux - (Ny+1)*2 + 2,   c_val)
                        fac = 1/dy*dx*(lam + G)/G
                        if iy == 1 or iy == Ny - 1:
                            add(kux, kuy + (Ny+1)*2,      fac)
                            add(kux, kuy + (Ny+1)*2 - 2, -fac)
                            add(kux, kuy,                 -fac)
                            add(kux, kuy - 2,              fac)
                        else:
                            cf = cosa/dy/dy*dx*dx*(lam + G)/G/4
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
        Nx, Ny, N = p.Nx, p.Ny, g.N
        dx, dy   = g.dx, g.dy
        G        = p.G
        sina, cosa = g.sina, g.cosa
        y        = g.y
        mid      = Nx // 2    # fault column 0-based

        RH = np.zeros(N)

        for ix in range(Nx+1):
            for iy in range(Ny+1):
                kux, kuy = self._dofs(ix, iy, Ny)

                # ── uy block ──
                if iy < Ny:
                    if ix == 0:
                        if p.bc.left.uy.type == BCType.VELOCITY:
                            RH[kuy] = p.bc.left.uy.value
                        else:
                            pass
                    elif ix == Nx:
                        if p.bc.right.uy.type == BCType.VELOCITY:
                            RH[kuy] = p.bc.right.uy.value
                        else:
                            pass
                    elif iy == 0:
                        if p.bc.bottom.uy.type == BCType.VELOCITY:
                            if p.case_type == "lab":
                                if ix > mid:
                                    RH[kuy] = p.bc.bottom.uy.value
                            else:
                                RH[kuy] = p.bc.bottom.uy.value
                        else:
                            pass
                    elif iy == Ny - 1: #top boundary
                        if p.bc.top.uy.type == BCType.VELOCITY:
                            if p.case_type == "lab":
                                if ix > mid:
                                    RH[kuy] = p.bc.top.uy.value
                            elif p.case_type == "california":
                                if ix < mid:
                                    RH[kuy] = -1 * p.bc.top.uy.value
                                elif ix > mid:
                                    RH[kuy] = p.bc.top.uy.value
                            else:
                                RH[kuy] = p.bc.top.uy.value
                        else:
                            pass
                    elif ix == mid:
                        if p.case_type == "california" and y[iy] >= p.W_f:
                            RH[kuy] = p.loading.V_L
                        else:
                            RH[kuy] = V[iy]
                    elif ix == mid + 1:
                        pass
                    else:
                        if p.case_type == "groningen":
                            yv = y[iy]
                            #TODO: make this more general, not hard-coded
                            if yv == 850 and ix >= mid + 1:
                                RH[kuy] =  dPdt / dy * dx*dx / G * sina
                            if yv == 1050 and ix >= mid + 1:
                                RH[kuy] = -dPdt / dy * dx*dx / G * sina
                            if yv == 800 and ix <= mid:
                                RH[kuy] =  dPdt / dy * dx*dx / G * sina
                            if yv == 1000 and ix <= mid:
                                RH[kuy] = -dPdt / dy * dx*dx / G * sina

                # ── ux block ──
                if ix < Nx:
                    if iy == 0: #bottom boundary
                        if p.bc.bottom.ux.type == BCType.VELOCITY:
                            RH[kux] = p.bc.bottom.ux.value
                        else:
                            pass
                    elif iy == Ny: #top boundary
                        if p.bc.top.ux.type == BCType.VELOCITY:
                            if p.case_type == "california":
                                if ix < mid:
                                    RH[kux] = -1 * p.bc.top.ux.value
                                elif ix > mid:
                                    RH[kux] = p.bc.top.ux.value
                            else:
                                RH[kux] = p.bc.top.ux.value
                        else:
                            pass
                    elif ix == 0:
                        if p.bc.left.ux.type == BCType.VELOCITY:
                            RH[kux] = p.bc.left.ux.value
                        else:
                            pass
                    elif ix == Nx - 1:
                        if p.bc.right.ux.type == BCType.VELOCITY:
                            RH[kux] = p.bc.right.ux.value
                        else:
                            pass
                    elif ix == mid:
                        if p.case_type == "groningen":
                            yv = y[iy]
                            if 800 < yv <= 850:
                                RH[kux] = -dPdt * dx / G
                            if 1000 < yv <= 1050:
                                RH[kux] =  dPdt * dx / G
                    else:
                        if p.case_type == "groningen":
                            yv = y[iy]
                            if yv == 1050 and ix > mid + 1:
                                RH[kux] =  dPdt / dy * dx*dx / G * sina * cosa
                            if yv == 1000 and ix < mid + 1:
                                RH[kux] =  dPdt / dy * dx*dx / G * sina * cosa
                            if yv == 850 and ix > mid + 1:
                                RH[kux] = -dPdt / dy * dx*dx / G * sina * cosa
                            if yv == 800 and ix < mid + 1:
                                RH[kux] = -dPdt / dy * dx*dx / G * sina * cosa
        return RH