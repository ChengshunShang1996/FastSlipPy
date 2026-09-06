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
from fastslippy.utilities.grid_operators import (
    build_recovery_operators,
    finite_difference_weights,
)


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

    @staticmethod
    def _spacing_metrics(coords: np.ndarray):
        """Return MATLAB grid_metrics-style backward/forward/centred spacing."""

        coords = np.asarray(coords, dtype=float)
        differences = np.diff(coords)
        backward = np.concatenate(([differences[0]], differences))
        forward = np.concatenate((backward[1:], [backward[-1]]))
        centred = 0.5 * (backward + forward)
        return backward, forward, centred

    def _precompute_local_steps(self):
        p, g = self.p, self.grid
        Nx, Ny = p.Nx, p.Ny

        self._hxm_uy, self._hxp_uy, self._dx_xuy = self._spacing_metrics(g.xp)
        self._hym_uy, self._hyp_uy, self._dy_yuy = self._spacing_metrics(g.y)
        self._hxm_ux, self._hxp_ux, self._dx_xux = self._spacing_metrics(g.x)
        self._hym_ux, self._hyp_ux, self._dy_yux = self._spacing_metrics(g.yp)

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
        use_coordinate_nonuniform_operator = not self._is_uniform
        is_california = self._case_type == "california"
        is_vertical_california = is_california and cosa == 0.0
        recovery = build_recovery_operators(g.x, g.y, g.xp, g.yp)

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
            """Quadratic-exact three-point derivative at a node."""
            stencil = three_point_stencil(coords.size, idx)
            weights = finite_difference_weights(
                coords[idx], coords[list(stencil)], 1
            )
            return tuple(zip(stencil, weights))

        def sparse_row_entries(matrix: sparse.csr_matrix, row: int):
            """Return non-zero column/value pairs from one recovery row."""
            start, stop = matrix.indptr[row : row + 2]
            return zip(matrix.indices[start:stop], matrix.data[start:stop])

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
            weights = finite_difference_weights(target, pts, derivative)
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
                            if is_california:
                                add(kuy, kuy + (Ny+1)*2, 1)
                        else:
                            raise ValueError(f"Unknown BC type: {p.bc.left.uy.type}")
                    elif ix == Nx: #right boundary
                        if p.bc.right.uy.type == BCType.FREE:
                            add(kuy, kuy, 1);  add(kuy, kuy - (Ny+1)*2, -1)
                        elif p.bc.right.uy.type == BCType.FIXED or p.bc.right.uy.type == BCType.VELOCITY:
                            add(kuy, kuy, 1)
                            if is_california:
                                add(kuy, kuy - (Ny+1)*2, 1)
                        else:
                            raise ValueError(f"Unknown BC type: {p.bc.right.uy.type}")
                    elif iy == 0 and not (
                        is_california and ix in (mid, mid + 1)
                    ): #top boundary (y=0 / free surface)
                        if p.bc.top.uy.type == BCType.FREE:
                            #add(kuy, kuy, 1);  add(kuy, kuy + (Ny+1)*2, -1)
                            add(kuy, kuy, 1);  add(kuy, kuy + 2, -1)
                        elif p.bc.top.uy.type == BCType.FIXED or p.bc.top.uy.type == BCType.VELOCITY:
                            add(kuy, kuy, 1)
                        elif p.bc.top.uy.type == BCType.TRACTION_FREE:
                            if is_california:
                                wy = finite_difference_weights(
                                    g.y[0], g.y[:3], 1
                                )
                                normal_scale = dx_loc / G
                                for iy_d, weight in enumerate(wy):
                                    add(
                                        kuy,
                                        kuy + 2 * iy_d,
                                        normal_scale * (lam + 2 * G) * weight,
                                    )
                                add(kuy, kuy - (Ny+1)*2, normal_scale * G * cosa / dx_loc)
                                add(kuy, kuy + (Ny+1)*2, -normal_scale * G * cosa / dx_loc)
                                dx_ux = float(g.x[ix] - g.x[ix - 1])
                                coefficient = normal_scale * lam / (2 * dx_ux)
                                add(kuy, kux, coefficient)
                                add(kuy, kux + 2, coefficient)
                                add(kuy, kux - (Ny+1)*2, -coefficient)
                                add(kuy, kux - (Ny+1)*2 + 2, -coefficient)
                            else:
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
                    elif iy == Ny - 1 and is_california and ix == mid:
                        add(kuy, kuy, -1)
                        add(kuy, kuy + (Ny+1)*2, 1)
                    elif iy == Ny - 1: #bottom boundary (y=ysize / deep boundary)
                        if p.bc.bottom.uy.type == BCType.FIXED or p.bc.bottom.uy.type == BCType.VELOCITY:
                            add(kuy, kuy, 1)
                        elif p.bc.bottom.uy.type == BCType.FREE:
                            #add(kuy, kuy, 1);  add(kuy, kuy - (Ny+1)*2, -1)
                            add(kuy, kuy, 1);  add(kuy, kuy - 2, -1)
                        elif p.bc.bottom.uy.type == BCType.TRACTION_FREE:
                            if is_california:
                                wy = finite_difference_weights(
                                    g.y[-1], g.y[-3:], 1
                                )
                                normal_scale = dx_loc / G
                                for iy_d, weight in zip(range(Ny - 3, Ny), wy):
                                    _, kuy_d = self._dofs(ix, iy_d, Ny)
                                    add(
                                        kuy,
                                        kuy_d,
                                        normal_scale * (lam + 2 * G) * weight,
                                    )
                                add(kuy, kuy - (Ny+1)*2, normal_scale * G * cosa / dx_loc)
                                add(kuy, kuy + (Ny+1)*2, -normal_scale * G * cosa / dx_loc)
                                dx_ux = float(g.x[ix] - g.x[ix - 1])
                                coefficient = normal_scale * lam / (2 * dx_ux)
                                add(kuy, kux, coefficient)
                                add(kuy, kux + 2, coefficient)
                                add(kuy, kux - (Ny+1)*2, -coefficient)
                                add(kuy, kux - (Ny+1)*2 + 2, -coefficient)
                            else:
                                dy_bottom = float(g.y[-1] - g.y[-2])
                                dx_ux = float(g.x[ix] - g.x[ix - 1])
                                normal_scale = 1.0 / (lam + 2.0 * G)
                                add(kuy, kuy,      1.0 / dy_bottom)
                                add(kuy, kuy - 2, -1.0 / dy_bottom)
                                kux_bottom, _ = self._dofs(ix, Ny, Ny)
                                kux_left, _ = self._dofs(ix - 1, Ny, Ny)
                                add(kuy, kux_bottom, lam * normal_scale / dx_ux)
                                add(kuy, kux_left, -lam * normal_scale / dx_ux)
                                for ix_d, w in first_derivative_weights(g.xp, ix):
                                    _, kuy_d = self._dofs(ix_d, iy, Ny)
                                    add(kuy, kuy_d, -2.0 * G * normal_scale * cosa * w)
                        else:
                            raise ValueError(f"BC type: {p.bc.bottom.uy.type} is not supported for bottom boundary yet.")
                    elif ix == mid:
                        # Fault left side
                        add(kuy, kuy, -1); add(kuy, kuy + (Ny+1)*2, 1)
                    elif is_vertical_california and ix == mid + 1:
                        # Preserve the MATLAB BP3 row for the vertical case.
                        # With cos(alpha)=0 its cross-coupling terms vanish,
                        # and this row already matches recovered shear traction.
                        dx_fault = dx_xuy[ix]
                        dy_fault = dy_yuy[iy]
                        spacing_left = g.xp[ix - 1] - g.xp[ix - 2]
                        spacing_right = g.xp[ix + 1] - g.xp[ix]
                        scale_left = dx_fault / spacing_left
                        scale_right = dx_fault / spacing_right
                        add(kuy, kuy - 2*(Ny+1)*2, scale_left)
                        add(kuy, kuy - (Ny+1)*2, -scale_left)
                        add(kuy, kuy, -scale_right)
                        add(kuy, kuy + (Ny+1)*2, scale_right)

                        if iy == 0:
                            wy = finite_difference_weights(
                                g.y[0], g.y[:3], 1
                            )
                            for iy_d, weight in enumerate(wy):
                                add(
                                    kuy,
                                    kuy - (Ny+1)*2 + 2*iy_d,
                                    cosa * dx_fault * weight,
                                )
                                add(
                                    kuy,
                                    kuy + 2*iy_d,
                                    -cosa * dx_fault * weight,
                                )
                        else:
                            coefficient_left = cosa * dx_fault / (2 * dy_fault)
                            coefficient_right = -coefficient_left
                            add(kuy, kuy - (Ny+1)*2 - 2, -coefficient_left)
                            add(kuy, kuy - (Ny+1)*2 + 2, coefficient_left)
                            add(kuy, kuy - 2, -coefficient_right)
                            add(kuy, kuy + 2, coefficient_right)

                        add(kuy, kux, cosa / 2)
                        add(kuy, kux + 2, cosa / 2)
                        add(kuy, kux - (Ny+1)*2, -cosa)
                        add(kuy, kux - (Ny+1)*2 + 2, -cosa)
                        add(kuy, kux - 2*(Ny+1)*2, cosa / 2)
                        add(kuy, kux - 2*(Ny+1)*2 + 2, cosa / 2)
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

                        if is_california:
                            # Reuse the exact sparse derivative/interpolation
                            # rows used by StressCalUtil.  A hard-coded 1/2
                            # average is inconsistent on a stretched mesh.
                            for j, sign in ((jl, 1.0), (jr, -1.0)):
                                for iy_d, wy in sparse_row_entries(
                                    recovery.midpoint_y_to_node, iy
                                ):
                                    for ix_d, wx in sparse_row_entries(
                                        recovery.derivative_x, j
                                    ):
                                        add_ux(
                                            ix_d,
                                            iy_d,
                                            cosa * sign * wy * wx,
                                        )

                            for j, sign in ((jl, 1.0), (jr, -1.0)):
                                for ix_u, wx in sparse_row_entries(
                                    recovery.midpoint_x_to_node, j
                                ):
                                    for iy_d, wy in sparse_row_entries(
                                        recovery.derivative_y, iy
                                    ):
                                        add_uy(
                                            ix_u,
                                            iy_d,
                                            -cosa * sign * wx * wy,
                                        )
                        else:
                            # Preserve the existing non-BP3 discretisation.
                            for j, sign in ((jl, 1.0), (jr, -1.0)):
                                for ix_d, w in first_derivative_weights(g.x, j):
                                    add_ux(
                                        ix_d, iy, 0.5 * cosa * sign * w
                                    )
                                    add_ux(
                                        ix_d, iy + 1, 0.5 * cosa * sign * w
                                    )
                            for j, sign in ((jl, 1.0), (jr, -1.0)):
                                for ix_u in (j, j + 1):
                                    for iy_d, w in first_derivative_weights(
                                        g.y, iy
                                    ):
                                        add_uy(
                                            ix_u,
                                            iy_d,
                                            -0.5 * cosa * sign * w,
                                        )
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
                            x_uy = three_point_stencil(Nx + 1, ix)
                            y_uy = three_point_stencil(Ny, iy)
                            b2 = (lam + G) / G
                            c3 = cosa * (lam + 3 * G) / G
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
                        elif use_coordinate_nonuniform_operator:
                            # (lambda+G) ux_xy: the staggered four-point mixed
                            # derivative spans one physical interval each way.
                            add(kuy, kux - (Ny+1)*2,      fac)
                            add(kuy, kux - (Ny+1)*2 + 2, -fac)
                            add(kuy, kux,                 -fac)
                            add(kuy, kux + 2,              fac)

                            # MATLAB increment 2e: keep -c*ux_xx separate and
                            # evaluate it at xp[ix] from four ux columns.  The
                            # adjacent-boundary rows intentionally omit this
                            # wider contribution, matching build_LH.m.
                            if ix not in (1, Nx - 1):
                                x_indices = (ix - 2, ix - 1, ix, ix + 1)
                                weights = point_weights(
                                    g.x, g.xp[ix], x_indices, 2
                                )
                                coefficient = -cosa * b2 * dx_loc * dx_loc / 2.0
                                for ix_d, weight in zip(x_indices, weights):
                                    kux_0, _ = self._dofs(ix_d, iy, Ny)
                                    kux_1, _ = self._dofs(ix_d, iy + 1, Ny)
                                    add(kuy, kux_0, coefficient * weight)
                                    add(kuy, kux_1, coefficient * weight)
                        else:
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
                    if iy == 0 and is_california and ix == mid:
                        # Fault-line ghost above the free surface: zero
                        # curvature along the fault, not zero shear traction.
                        add(kux, kux, 1)
                        add(kux, kux + 2, -2)
                        add(kux, kux + 4, 1)
                    elif iy == 0: #top boundary (y=0 / free surface)
                        if p.bc.top.ux.type == BCType.FIXED or p.bc.top.ux.type == BCType.VELOCITY:
                            add(kux, kux, 1)
                        elif p.bc.top.ux.type == BCType.FREE:
                            add(kux, kux, 1); add(kux, kux + 2, -1)
                        elif p.bc.top.ux.type == BCType.TRACTION_FREE:
                            if is_california:
                                wy = finite_difference_weights(
                                    g.y[0], g.y[:3], 1
                                )
                                shear_scale = dx_loc / sina
                                hux_surface = g.yp[1] - g.yp[0]
                                add(kux, kux, -shear_scale / hux_surface)
                                add(kux, kux + 2, shear_scale / hux_surface)

                                a2 = 1.0 - 2.0 * cosa * cosa
                                for iy_d, weight in enumerate(wy):
                                    coefficient = shear_scale * cosa * weight / 2
                                    add(kux, kuy + 2*iy_d, coefficient)
                                    add(kux, kuy + (Ny+1)*2 + 2*iy_d, coefficient)
                                add(kux, kuy, -shear_scale * a2 / dx_loc)
                                add(kux, kuy + (Ny+1)*2, shear_scale * a2 / dx_loc)

                                if ix == 0:
                                    x_terms = ((ix, 0.5), (ix + 1, -0.5))
                                elif ix == Nx - 1:
                                    x_terms = ((ix, -0.5), (ix - 1, 0.5))
                                else:
                                    x_terms = ((ix + 1, -0.25), (ix - 1, 0.25))
                                for ix_d, factor in x_terms:
                                    kux_d, _ = self._dofs(ix_d, iy, Ny)
                                    add(kux, kux_d, shear_scale * cosa * factor / dx_loc)
                                    add(kux, kux_d + 2, shear_scale * cosa * factor / dx_loc)
                            else:
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
                            if is_california:
                                add(kux, kux - 2, 1)
                        elif p.bc.bottom.ux.type == BCType.FREE:
                            add(kux, kux, 1); add(kux, kux - 2, -1)
                        elif p.bc.bottom.ux.type == BCType.TRACTION_FREE:
                            if is_california:
                                shear_scale = dx_loc / sina
                                hux_bottom = g.yp[-1] - g.yp[-2]
                                add(kux, kux - 2, -shear_scale / hux_bottom)
                                add(kux, kux,      shear_scale / hux_bottom)

                                wy = finite_difference_weights(
                                    g.y[-1], g.y[-3:], 1
                                )
                                a2 = 1.0 - 2.0 * cosa * cosa
                                for iy_d, weight in zip(range(Ny - 3, Ny), wy):
                                    _, kuy_left = self._dofs(ix, iy_d, Ny)
                                    _, kuy_right = self._dofs(ix + 1, iy_d, Ny)
                                    coefficient = shear_scale * cosa * weight / 2
                                    add(kux, kuy_left, coefficient)
                                    add(kux, kuy_right, coefficient)

                                _, kuy_left = self._dofs(ix, Ny - 1, Ny)
                                _, kuy_right = self._dofs(ix + 1, Ny - 1, Ny)
                                add(kux, kuy_left, -shear_scale * a2 / dx_loc)
                                add(kux, kuy_right, shear_scale * a2 / dx_loc)

                                if ix == 0:
                                    x_terms = ((ix, 0.5), (ix + 1, -0.5))
                                elif ix == Nx - 1:
                                    x_terms = ((ix, -0.5), (ix - 1, 0.5))
                                else:
                                    x_terms = ((ix + 1, -0.25), (ix - 1, 0.25))
                                for ix_d, factor in x_terms:
                                    kux_inner, _ = self._dofs(ix_d, Ny - 1, Ny)
                                    kux_ghost, _ = self._dofs(ix_d, Ny, Ny)
                                    add(kux, kux_inner, shear_scale * cosa * factor / dx_loc)
                                    add(kux, kux_ghost, shear_scale * cosa * factor / dx_loc)
                            else:
                                dy_ux_bottom = float(g.yp[-1] - g.yp[-2])
                                dy_uy_bottom = float(g.y[-1] - g.y[-2])
                                dx_uy = float(g.xp[ix + 1] - g.xp[ix])
                                a2 = 1.0 - 2.0 * cosa * cosa
                                add(kux, kux,      1.0 / dy_ux_bottom)
                                add(kux, kux - 2, -1.0 / dy_ux_bottom)
                                for ix_d, w in first_derivative_weights(g.x, ix):
                                    kux_d, _ = self._dofs(ix_d, iy, Ny)
                                    add(kux, kux_d, -cosa * w)
                                for ix_u in (ix, ix + 1):
                                    _, kuy_0 = self._dofs(ix_u, Ny - 2, Ny)
                                    _, kuy_1 = self._dofs(ix_u, Ny - 1, Ny)
                                    add(kux, kuy_1,  0.5 * cosa / dy_uy_bottom)
                                    add(kux, kuy_0, -0.5 * cosa / dy_uy_bottom)
                                _, kuy_left = self._dofs(ix, Ny - 1, Ny)
                                _, kuy_right = self._dofs(ix + 1, Ny - 1, Ny)
                                add(kux, kuy_right,  a2 / dx_uy)
                                add(kux, kuy_left,  -a2 / dx_uy)
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
                    elif is_vertical_california and ix == mid:
                        # Preserve the MATLAB BP3 row for alpha=90, where it
                        # is already consistent and has an exact regression
                        # reference.  Inclined faults use the recovered-node
                        # traction row below.
                        dx_fault = dx_xux[ix]
                        dy_fault = dy_yux[iy]
                        h_minus = self._hxm_ux[ix]
                        h_plus = self._hxp_ux[ix]
                        ratio = (lam + 2 * G) / G
                        add(kux, kux, -dx_fault * ratio * (
                            1.0 / h_minus + 1.0 / h_plus
                        ))
                        add(kux, kux - (Ny+1)*2, dx_fault * ratio / h_minus)
                        add(kux, kux + (Ny+1)*2, dx_fault * ratio / h_plus)

                        coefficient = lam / G * dx_fault / dy_fault
                        add(kux, kuy, -coefficient)
                        add(kux, kuy + (Ny+1)*2, coefficient)
                        add(kux, kuy - 2, coefficient)
                        add(kux, kuy + (Ny+1)*2 - 2, -coefficient)
                    elif ix == mid:
                        # For inclined BP3, enforce equality of the fault-node
                        # normal tractions consumed by the friction law.
                        # sigmaqs is cell centred, so the BP3 path includes
                        # the same recovery used after the solve.  Other case
                        # types retain their existing cell-centred row below.
                        fault_row = kux
                        jl, jr = mid - 1, mid
                        dx_ux = np.diff(g.x)
                        dy_uy = np.diff(g.y)
                        dy_ux = np.diff(g.yp)
                        # This expression is in stress units.  Use the same
                        # local dx/G normalization as the original row.
                        fault_scale = dx_ux[jr] / G

                        if is_california:
                            for isigma, node_weight in sparse_row_entries(
                                recovery.sigma_centres_to_nodes, iy
                            ):
                                # (lambda+2G) ux_x
                                cx_l = (
                                    node_weight * (lam + 2.0 * G) / dx_ux[jl]
                                )
                                add_ux(jl + 1, isigma + 1,  cx_l)
                                add_ux(jl,     isigma + 1, -cx_l)
                                cx_r = (
                                    node_weight * (lam + 2.0 * G) / dx_ux[jr]
                                )
                                add_ux(jr + 1, isigma + 1, -cx_r)
                                add_ux(jr,     isigma + 1,  cx_r)

                                # lambda uy_y
                                cy = node_weight * lam / dy_uy[isigma]
                                add_uy(jl + 1, isigma + 1,  cy)
                                add_uy(jl + 1, isigma,     -cy)
                                add_uy(jr + 1, isigma + 1, -cy)
                                add_uy(jr + 1, isigma,      cy)

                                # -2G*cos(a)*mm_outer(ux_y).  The common
                                # central ux column cancels between adjacent
                                # cells.
                                cxy = -0.5 * G * cosa * node_weight
                                for iy_d in (isigma, isigma + 1):
                                    wy = cxy / dy_ux[iy_d]
                                    add_ux(jl, iy_d + 1,  wy)
                                    add_ux(jl, iy_d,     -wy)
                                    add_ux(jr + 1, iy_d + 1, -wy)
                                    add_ux(jr + 1, iy_d,      wy)
                        else:
                            # Preserve the existing non-BP3 cell-centred row.
                            isigma = iy - 1
                            cx_l = (lam + 2.0 * G) / dx_ux[jl]
                            add_ux(jl + 1, iy,  cx_l)
                            add_ux(jl,     iy, -cx_l)
                            cx_r = (lam + 2.0 * G) / dx_ux[jr]
                            add_ux(jr + 1, iy, -cx_r)
                            add_ux(jr,     iy,  cx_r)

                            cy = lam / dy_uy[isigma]
                            add_uy(jl + 1, isigma + 1,  cy)
                            add_uy(jl + 1, isigma,     -cy)
                            add_uy(jr + 1, isigma + 1, -cy)
                            add_uy(jr + 1, isigma,      cy)

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
                            x_ux = three_point_stencil(Nx, ix)
                            y_ux = three_point_stencil(Ny + 1, iy)
                            b2 = (lam + G) / G
                            c3 = cosa * (lam + 3 * G) / G
                            # -c*(lambda+3G)/G ux_xy
                            add_tensor_derivative(kux, "ux", g.x, x_ux, g.x[ix],
                                                  1, g.yp, y_ux, g.yp[iy], 1,
                                                  -coordinate_scale * c3)
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
                        elif use_coordinate_nonuniform_operator:
                            # (lambda+G) uy_xy on its native staggered cell.
                            add(kux, kuy + (Ny+1)*2,      fac)
                            add(kux, kuy + (Ny+1)*2 - 2, -fac)
                            add(kux, kuy,                 -fac)
                            add(kux, kuy - 2,              fac)

                            # MATLAB increment 2e: -c*uy_yy at yp[iy] from
                            # four uy rows, averaged across the two columns.
                            if iy not in (1, Ny - 1):
                                y_indices = (iy - 2, iy - 1, iy, iy + 1)
                                weights = point_weights(
                                    g.y, g.yp[iy], y_indices, 2
                                )
                                coefficient = -cosa * b2 * dx_loc * dx_loc / 2.0
                                for iy_d, weight in zip(y_indices, weights):
                                    _, kuy_0 = self._dofs(ix, iy_d, Ny)
                                    _, kuy_1 = self._dofs(ix + 1, iy_d, Ny)
                                    add(kux, kuy_0, coefficient * weight)
                                    add(kux, kuy_1, coefficient * weight)
                        else:
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
            factor = 2.0 if is_california else 1.0
            RH[self._kuy[:, 0]] = factor * p.bc.left.uy.value
        if p.bc.right.uy.type == BCType.VELOCITY:
            factor = 2.0 if is_california else 1.0
            RH[self._kuy[:, Nx]] = factor * p.bc.right.uy.value

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
            # MATLAB applies the fault jump at every physical fault node,
            # including the free-surface and bottom intersections.
            RH[self._kuy[:, mid]] = V
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
