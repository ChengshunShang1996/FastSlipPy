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

from fastslippy.pre_processing.model_parameters import ModelParameters
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
                    if ix == 0: # Neumann BC
                        add(kuy, kuy, 1);  add(kuy, kuy + (Ny+1)*2, -1)
                    elif ix == Nx:
                        add(kuy, kuy, 1);  add(kuy, kuy - (Ny+1)*2, -1)
                    elif iy == 0:
                        add(kuy, kuy, 1)
                    elif iy == Ny - 1:
                        add(kuy, kuy, 1)
                    elif ix == mid:
                        # Fault left side
                        add(kuy, kuy, -1); add(kuy, kuy + (Ny+1)*2, 1)
                    elif ix == mid + 1:
                        # Fault right side
                        kux_n, kuy_n = self._dofs(ix, iy, Ny)
                        add(kuy, kuy - 2*(Ny+1)*2, 1)
                        add(kuy, kuy - (Ny+1)*2,  -1)
                        add(kuy, kuy,              -1)
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
                    add(kuy, kuy, 1)

                # ── ux equation (ix < Nx) ──────────────────────────────
                if ix < Nx:
                    r2 = dx*dx / dy/dy
                    r_lam = (lam + 2*G) / G
                    if iy == 0:
                        add(kux, kux, 1); add(kux, kux + 2, -1)
                    elif iy == Ny:
                        add(kux, kux, 1); add(kux, kux - 2, -1)
                    elif ix == 0:
                        add(kux, kux, 1)
                    elif ix == Nx - 1:
                        add(kux, kux, 1)
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
                        pass
                    elif ix == Nx:
                        pass
                    elif iy == 0:
                        pass
                    elif iy == Ny - 1:
                        pass
                    elif ix == mid:
                        RH[kuy] = V[iy]
                    elif ix == mid + 1:
                        pass
                    else:
                        yv = y[iy]
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
                    if iy == 0:
                        pass
                    elif iy == Ny:
                        pass
                    elif ix == 0:
                        pass
                    elif ix == Nx - 1:
                        pass
                    elif ix == mid:
                        yv = y[iy]
                        if 800 < yv <= 850:
                            RH[kux] = -dPdt * dx / G
                        if 1000 < yv <= 1050:
                            RH[kux] =  dPdt * dx / G
                    else:
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