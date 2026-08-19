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
                    "Nonuniform (stretched) mesh is enabled, "
                    "please set allow_nonuniform_solver=True to run it explicitly."
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

        self._dx_xuy = np.array(g.metric_xp, dtype=float)
        self._dy_yuy = np.array(g.metric_y, dtype=float)
        self._dx_xux = np.array(g.metric_x, dtype=float)
        self._dy_yux = np.array(g.metric_yp, dtype=float)

    def _precompute_rhs_layout(self):
        p = self.p
        Nx, Ny = p.Nx, p.Ny

        self._mid = Nx // 2
        self._iy_int = np.arange(1, Ny - 1)  # interior iy for uy rows and ux interior rows

        self._ix_uy_y_boundaries = np.arange(1, Nx)  # exclude left/right boundaries by precedence
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
        use_coordinate_nonuniform_operator = (
            not self._is_uniform and self._case_type == "california"
        )

        rows, cols, vals = [], [], []
        _point_weight_cache = {}

        def add(r, c, v):
            rows.append(r); cols.append(c); vals.append(v)

        def add_ux(ix, iy, value):
            """Add a coefficient for ux(ix, iy) to a fault-interface row."""
            kux_local, _ = self._dofs(ix, iy, Ny)
            add(fault_row, kux_local, fault_scale * value)

        def add_uy(ix, iy, value):
            """Add a coefficient for uy(ix, iy) to a fault-interface row."""
            _, kuy_local = self._dofs(ix, iy, Ny)
            add(fault_row, kuy_local, fault_scale * value)

        def first_derivative_weights(coords: np.ndarray, idx: int):
            """Weights used by numpy.gradient(..., edge_order=1) at one node."""
            if idx == 0:
                h = float(coords[1] - coords[0])
                return ((0, -1.0 / h), (1, 1.0 / h))
            if idx == coords.size - 1:
                h = float(coords[-1] - coords[-2])
                return ((idx - 1, -1.0 / h), (idx, 1.0 / h))
            h_l = float(coords[idx] - coords[idx - 1])
            h_r = float(coords[idx + 1] - coords[idx])
            return (
                (idx - 1, -h_r / (h_l * (h_l + h_r))),
                (idx, (h_r - h_l) / (h_l * h_r)),
                (idx + 1, h_l / (h_r * (h_l + h_r))),
            )

        def point_weights(coords: np.ndarray, target: float, indices, derivative: int):
            """Polynomial-exact weights for a derivative at an off-grid point.

            The velocity components live on staggered grids.  On a stretched
            mesh a mixed derivative therefore cannot be obtained by applying
            a uniform-grid ``dx/dy`` correction to the old four-point stencil:
            the derivative has to be evaluated at the actual target point.
            Three neighbouring points give a second-order, quadratic-exact
            approximation for derivative orders zero, one, and two.
            """
            key = (id(coords), float(target), tuple(indices), derivative)
            cached = _point_weight_cache.get(key)
            if cached is not None:
                return cached
            pts = np.asarray(coords)[list(indices)]
            n = len(indices)
            system = np.empty((n, n), dtype=float)
            shifted = pts - target
            for power in range(n):
                system[power, :] = shifted ** power
            rhs = np.zeros(n, dtype=float)
            rhs[derivative] = (1.0, 1.0, 2.0)[derivative]
            weights = np.linalg.solve(system, rhs)
            _point_weight_cache[key] = weights
            return weights

        def add_tensor_derivative(row, component, x_coords, x_indices, x_target,
                                  x_order, y_coords, y_indices, y_target,
                                  y_order, coefficient):
            """Add a staggered-grid tensor-product derivative to ``row``."""
            wx = point_weights(x_coords, x_target, x_indices, x_order)
            wy = point_weights(y_coords, y_target, y_indices, y_order)
            for j, wxj in zip(x_indices, wx):
                for i, wyi in zip(y_indices, wy):
                    kux_local, kuy_local = self._dofs(j, i, Ny)
                    add(row, kux_local if component == "ux" else kuy_local,
                        coefficient * wxj * wyi)

        def three_point_stencil(size: int, centre: int):
            """Return a valid three-point stencil, one-sided near an edge."""
            start = min(max(centre - 1, 0), size - 3)
            return (start, start + 1, start + 2)

        def three_point_stencil_in_range(centre: int, lower: int, upper: int):
            """Three-point stencil confined to one physical side of the fault."""
            start = min(max(centre - 1, lower), upper - 2)
            return (start, start + 1, start + 2)

        for ix in range(Nx+1):           # 0 … Nx  (MATLAB 1 … Nx+1)
            for iy in range(Ny+1):       # 0 … Ny

                kux, kuy = self._dofs(ix, iy, Ny)
                fault_row = None
                fault_scale = None
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
                    elif iy == 0: #top boundary (y=0 / free surface)
                        if p.bc.top.uy.type == BCType.FREE:
                            #add(kuy, kuy, 1);  add(kuy, kuy + (Ny+1)*2, -1)
                            add(kuy, kuy, 1);  add(kuy, kuy + 2, -1)
                        elif p.bc.top.uy.type == BCType.FIXED or p.bc.top.uy.type == BCType.VELOCITY:
                            add(kuy, kuy, 1)
                        elif p.bc.top.uy.type == BCType.TRACTION_FREE:
                            # Direct global sigma_ZZ=0 at the uy node:
                            # lambda*ux_x + (lambda+2G)*uy_y
                            #     - 2G*cos(alpha)*uy_x = 0.
                            dy_top = float(g.y[1] - g.y[0])
                            dx_ux = float(g.x[ix] - g.x[ix - 1])
                            normal_scale = 1.0 / (lam + 2.0 * G)
                            add(kuy, kuy + 2, 1.0 / dy_top)
                            add(kuy, kuy,     -1.0 / dy_top)
                            add(kuy, kux,      lam * normal_scale / dx_ux)
                            add(kuy, kux - (Ny + 1) * 2, -lam * normal_scale / dx_ux)
                            for ix_d, w in first_derivative_weights(g.xp, ix):
                                _, kuy_d = self._dofs(ix_d, iy, Ny)
                                add(kuy, kuy_d, -2.0 * G * normal_scale * cosa * w)
                        else:
                            raise ValueError(f"BC type: {p.bc.top.uy.type} is not supported for top boundary yet.")
                    elif iy == Ny - 1: #bottom boundary (y=ysize / deep boundary)
                        if p.bc.bottom.uy.type == BCType.FIXED or p.bc.bottom.uy.type == BCType.VELOCITY:
                            add(kuy, kuy, 1)
                        elif p.bc.bottom.uy.type == BCType.FREE:
                            #add(kuy, kuy, 1);  add(kuy, kuy - (Ny+1)*2, -1)
                            add(kuy, kuy, 1);  add(kuy, kuy - 2, -1)
                        else:
                            raise ValueError(f"BC type: {p.bc.bottom.uy.type} is not supported for bottom boundary yet.")
                    elif ix == mid:
                        # Fault left side
                        add(kuy, kuy, -1); add(kuy, kuy + (Ny+1)*2, 1)
                    elif ix == mid + 1:
                        # Enforce tau(left)-tau(right)=0 using precisely the
                        # same staggered, coordinate-aware operator as
                        # StressCalUtil.compute_stress_fields.
                        fault_row = kuy
                        jl, jr = mid - 1, mid + 1
                        dx_uy = np.diff(g.xp)
                        dy_ux = np.diff(g.yp)
                        # The recovered expression below is tau/G.  Scale the
                        # row by a local length, as the legacy dimensionless
                        # matrix did, so sparse factorization remains well
                        # conditioned without altering the constraint.
                        fault_scale = dx_uy[jr]
                        a2 = 1.0 - 2.0 * cosa * cosa

                        # uy_x
                        add_uy(jl + 1, iy,  1.0 / dx_uy[jl])
                        add_uy(jl,     iy, -1.0 / dx_uy[jl])
                        add_uy(jr + 1, iy, -1.0 / dx_uy[jr])
                        add_uy(jr,     iy,  1.0 / dx_uy[jr])

                        # (1-2*cos(a)^2) ux_y
                        cy = a2 / dy_ux[iy]
                        add_ux(jl, iy + 1,  cy); add_ux(jl, iy, -cy)
                        add_ux(jr, iy + 1, -cy); add_ux(jr, iy,  cy)

                        # cos(a) * averaged ux_x
                        for j, sign in ((jl, 1.0), (jr, -1.0)):
                            for ix_d, w in first_derivative_weights(g.x, j):
                                add_ux(ix_d, iy,     0.5 * cosa * sign * w)
                                add_ux(ix_d, iy + 1, 0.5 * cosa * sign * w)

                        # -cos(a) * averaged uy_y
                        for j, sign in ((jl, 1.0), (jr, -1.0)):
                            for ix_u in (j, j + 1):
                                for iy_d, w in first_derivative_weights(g.y, iy):
                                    add_uy(ix_u, iy_d, -0.5 * cosa * sign * w)
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
                            # Coordinate-aware mixed derivatives.  The
                            # uniform-grid stencil below is not valid after
                            # stretching because ux and uy are staggered in
                            # both directions.
                            scale = sx
                            coordinate_scale = scale if use_coordinate_nonuniform_operator else 0.0
                            if ix < mid:
                                x_ux = three_point_stencil_in_range(ix, 0, mid)
                                x_uy = three_point_stencil_in_range(ix, 0, mid)
                            else:
                                x_ux = three_point_stencil_in_range(ix, mid, Nx - 1)
                                x_uy = three_point_stencil_in_range(ix, mid + 1, Nx)
                            y_ux = three_point_stencil(Ny + 1, iy)
                            y_uy = three_point_stencil(Ny, iy)
                            b2 = (lam + G) / G
                            c3 = cosa * (lam + 3 * G) / G
                            # -c*b2 ux_xx.  The angle-independent ux_xy
                            # coupling is assembled with the same coordinate
                            # operator for BP3's nonuniform California mesh.
                            add_tensor_derivative(kuy, "ux", g.x, x_ux, g.xp[ix],
                                                  2, g.yp, y_ux, g.y[iy], 0,
                                                  -coordinate_scale * cosa * b2)
                            add_tensor_derivative(kuy, "ux", g.x, x_ux, g.xp[ix],
                                                  1, g.yp, y_ux, g.y[iy], 1,
                                                  coordinate_scale * b2)
                            # -c*(lambda+3G)/G uy_xy
                            add_tensor_derivative(kuy, "uy", g.xp, x_uy, g.xp[ix],
                                                  1, g.y, y_uy, g.y[iy], 1,
                                                  -coordinate_scale * c3)
                        fac = 1/dy_loc*dx_loc*(lam + G)/G
                        if self._is_uniform:
                            c_val = cosa/dy_loc*dx_loc*(lam + 3*G)/G/4
                            add(kuy, kuy + (Ny+1)*2 - 2,   c_val)
                            add(kuy, kuy + (Ny+1)*2 + 2,  -c_val)
                            add(kuy, kuy - (Ny+1)*2 - 2,  -c_val)
                            add(kuy, kuy - (Ny+1)*2 + 2,   c_val)
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
                        elif not use_coordinate_nonuniform_operator:
                            # Unchanged angle-independent staggered ux_xy.
                            add(kuy, kux - (Ny+1)*2,      fac)
                            add(kuy, kux - (Ny+1)*2 + 2, -fac)
                            add(kuy, kux,                 -fac)
                            add(kuy, kux + 2,              fac)
                else:
                    # Neumann BC for ghost uy nodes at iy=Ny
                    add(kuy, kuy, 1)

                # ── ux equation (ix < Nx) ──────────────────────────────
                if ix < Nx:
                    dx_loc = dx_xux[ix]
                    dy_loc = dy_yux[iy]
                    r2 = dx_loc*dx_loc / dy_loc/dy_loc
                    r_lam = (lam + 2*G) / G
                    if iy == 0: #top boundary (y=0 / free surface)
                        if p.bc.top.ux.type == BCType.FIXED or p.bc.top.ux.type == BCType.VELOCITY:
                            add(kux, kux, 1)
                        elif p.bc.top.ux.type == BCType.FREE:
                            add(kux, kux, 1); add(kux, kux + 2, -1)
                        elif p.bc.top.ux.type == BCType.TRACTION_FREE:
                            # Direct global sigma_XZ=0 at the ux node:
                            # ux_y - cos(alpha)*ux_x
                            #   + cos(alpha)*average_x(uy_y)
                            #   + (1-2*cos(alpha)^2)*uy_x = 0.
                            # It is deliberately not obtained by eliminating
                            # uy_y from the sigma_ZZ row: the two tractions are
                            # sampled at different staggered x locations.
                            dy_ux_top = float(g.yp[1] - g.yp[0])
                            dy_uy_top = float(g.y[1] - g.y[0])
                            dx_uy = float(g.xp[ix + 1] - g.xp[ix])
                            a2 = 1.0 - 2.0 * cosa * cosa
                            add(kux, kux + 2,  1.0 / dy_ux_top)
                            add(kux, kux,     -1.0 / dy_ux_top)
                            for ix_d, w in first_derivative_weights(g.x, ix):
                                kux_d, _ = self._dofs(ix_d, iy, Ny)
                                add(kux, kux_d, -cosa * w)
                            for ix_u in (ix, ix + 1):
                                _, kuy_0 = self._dofs(ix_u, iy, Ny)
                                _, kuy_1 = self._dofs(ix_u, iy + 1, Ny)
                                add(kux, kuy_1,  0.5 * cosa / dy_uy_top)
                                add(kux, kuy_0, -0.5 * cosa / dy_uy_top)
                            add(kux, kuy + (Ny + 1) * 2,  a2 / dx_uy)
                            add(kux, kuy,                 -a2 / dx_uy)
                        else:
                            raise ValueError(f"Unknown BC type: {p.bc.top.ux.type}")
                    elif iy == Ny:
                        if p.bc.bottom.ux.type == BCType.FIXED or p.bc.bottom.ux.type == BCType.VELOCITY:
                            add(kux, kux, 1)
                        elif p.bc.bottom.ux.type == BCType.FREE:
                            add(kux, kux, 1); add(kux, kux - 2, -1)
                        else:
                            raise ValueError(f"Unknown BC type: {p.bc.bottom.ux.type}")
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
                        # Enforce sigma(left)-sigma(right)=0 using the same
                        # cell-centred normal-stress recovery operator used
                        # after every elastic solve.  The ux row at iy maps to
                        # sigmaqs row iy-1 on the staggered grid.
                        fault_row = kux
                        isigma = iy - 1
                        jl, jr = mid - 1, mid
                        dx_ux = np.diff(g.x)
                        dy_uy = np.diff(g.y)
                        dy_ux = np.diff(g.yp)
                        # This expression is in stress units.  Use the same
                        # local dx/G normalization as the original row.
                        fault_scale = dx_ux[jr] / G

                        # (lambda+2G) ux_x
                        cx_l = (lam + 2.0 * G) / dx_ux[jl]
                        add_ux(jl + 1, iy,  cx_l)
                        add_ux(jl,     iy, -cx_l)
                        cx_r = (lam + 2.0 * G) / dx_ux[jr]
                        add_ux(jr + 1, iy, -cx_r)
                        add_ux(jr,     iy,  cx_r)

                        # lambda uy_y
                        cy = lam / dy_uy[isigma]
                        add_uy(jl + 1, isigma + 1,  cy)
                        add_uy(jl + 1, isigma,     -cy)
                        add_uy(jr + 1, isigma + 1, -cy)
                        add_uy(jr + 1, isigma,      cy)

                        # -2G*cos(a)*mm_outer(ux_y).  The common central
                        # ux column cancels between the two adjacent cells.
                        cxy = -0.5 * G * cosa
                        for iy_d in (isigma, isigma + 1):
                            wy = cxy / dy_ux[iy_d]
                            add_ux(jl, iy_d + 1,  wy)
                            add_ux(jl, iy_d,     -wy)
                            add_ux(jr + 1, iy_d + 1, -wy)
                            add_ux(jr + 1, iy_d,      wy)
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
                            # Coordinate-aware mixed derivatives at the ux
                            # node (x[ix], yp[iy]).  The old terms below are
                            # retained only for a uniform mesh.
                            scale = sx
                            coordinate_scale = scale if use_coordinate_nonuniform_operator else 0.0
                            if ix < mid:
                                x_ux = three_point_stencil_in_range(ix, 0, mid)
                                x_uy = three_point_stencil_in_range(ix, 0, mid)
                            else:
                                x_ux = three_point_stencil_in_range(ix, mid, Nx - 1)
                                x_uy = three_point_stencil_in_range(ix, mid + 1, Nx)
                            y_ux = three_point_stencil(Ny + 1, iy)
                            y_uy = three_point_stencil(Ny, iy)
                            b2 = (lam + G) / G
                            c3 = cosa * (lam + 3 * G) / G
                            # -c*(lambda+3G)/G ux_xy
                            add_tensor_derivative(kux, "ux", g.x, x_ux, g.x[ix],
                                                  1, g.yp, y_ux, g.yp[iy], 1,
                                                  -coordinate_scale * c3)
                            # -c*b2 uy_yy.  The angle-independent uy_xy
                            # coupling remains in its legacy staggered form.
                            add_tensor_derivative(kux, "uy", g.xp, x_uy, g.x[ix],
                                                  0, g.y, y_uy, g.yp[iy], 2,
                                                  -coordinate_scale * cosa * b2)
                            add_tensor_derivative(kux, "uy", g.xp, x_uy, g.x[ix],
                                                  1, g.y, y_uy, g.yp[iy], 1,
                                                  coordinate_scale * b2)
                        fac = 1/dy_loc*dx_loc*(lam + G)/G
                        if self._is_uniform:
                            c_val = cosa/dy_loc*dx_loc*(lam + 3*G)/G/4
                            add(kux, kux + (Ny+1)*2 - 2,   c_val)
                            add(kux, kux + (Ny+1)*2 + 2,  -c_val)
                            add(kux, kux - (Ny+1)*2 - 2,  -c_val)
                            add(kux, kux - (Ny+1)*2 + 2,   c_val)
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
                        elif not use_coordinate_nonuniform_operator:
                            # Unchanged angle-independent staggered uy_xy.
                            add(kux, kuy + (Ny+1)*2,      fac)
                            add(kux, kuy + (Ny+1)*2 - 2, -fac)
                            add(kux, kuy,                 -fac)
                            add(kux, kuy - 2,              fac)
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

        if p.bc.top.uy.type == BCType.VELOCITY:
            if is_lab:
                RH[self._kuy[0, mid + 1:Nx]] = p.bc.top.uy.value
            else:
                RH[self._kuy[0, self._ix_uy_y_boundaries]] = p.bc.top.uy.value

        if p.bc.bottom.uy.type == BCType.VELOCITY:
            if is_lab:
                RH[self._kuy[Ny - 1, mid + 1:Nx]] = p.bc.bottom.uy.value
            elif is_california:
                # The two duplicated fault-face uy nodes are mid (left) and
                # mid+1 (right).  Both must receive the far-field plate rate.
                RH[self._kuy[Ny - 1, 1:mid + 1]] = -p.bc.bottom.uy.value
                RH[self._kuy[Ny - 1, mid + 1:Nx]] = p.bc.bottom.uy.value
            else:
                RH[self._kuy[Ny - 1, self._ix_uy_y_boundaries]] = p.bc.bottom.uy.value

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
        if p.bc.top.ux.type == BCType.VELOCITY:
            RH[self._kux[0, self._ix_ux_all]] = p.bc.top.ux.value
        if p.bc.bottom.ux.type == BCType.VELOCITY:
            if is_california: #TODO: should be removed later, as it is not used
                RH[self._kux[Ny, :mid]] = -p.bc.bottom.ux.value
                RH[self._kux[Ny, mid + 1:]] = p.bc.bottom.ux.value
            else:
                RH[self._kux[Ny, self._ix_ux_all]] = p.bc.bottom.ux.value

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
