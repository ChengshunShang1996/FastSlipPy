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

from fastslippy.pre_processing.model_parameters import (
    BCType,
    BoundaryProfile,
    FaultBottomTreatment,
    FaultSurfaceTreatment,
    ModelParameters,
)
from fastslippy.pre_processing.grid import Grid
from fastslippy.utilities.grid_operators import finite_difference_weights


class MatrixBuilder:
    """
    Assembles the sparse stiffness matrix LH and right-hand-side vector RH
    for the quasi-static elastic equilibrium problem on the staggered grid.

    Fault traction continuity follows the MATLAB BP3 stencil for every case;
    only the physical outer-boundary and endpoint treatments are configurable.
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
        use_surface_fault_intersection = (
            p.fault_surface_treatment
            == FaultSurfaceTreatment.FREE_SURFACE_FAULT_INTERSECTION
        )
        use_bottom_fault_intersection = (
            p.fault_bottom_treatment
            == FaultBottomTreatment.DEEP_BOUNDARY_FAULT_INTERSECTION
        )

        rows, cols, vals = [], [], []
        _point_weight_cache = {}

        def add(r, c, v):
            rows.append(r); cols.append(c); vals.append(v)

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
                            add(kuy, kuy + (Ny+1)*2, 1)
                        else:
                            raise ValueError(f"Unknown BC type: {p.bc.left.uy.type}")
                    elif ix == Nx: #right boundary
                        if p.bc.right.uy.type == BCType.FREE:
                            add(kuy, kuy, 1);  add(kuy, kuy - (Ny+1)*2, -1)
                        elif p.bc.right.uy.type == BCType.FIXED or p.bc.right.uy.type == BCType.VELOCITY:
                            add(kuy, kuy, 1)
                            add(kuy, kuy - (Ny+1)*2, 1)
                        else:
                            raise ValueError(f"Unknown BC type: {p.bc.right.uy.type}")
                    elif iy == 0 and not (
                        use_surface_fault_intersection and ix in (mid, mid + 1)
                    ): #top boundary (y=0 / free surface)
                        if p.bc.top.uy.type == BCType.FREE:
                            #add(kuy, kuy, 1);  add(kuy, kuy + (Ny+1)*2, -1)
                            add(kuy, kuy, 1);  add(kuy, kuy + 2, -1)
                        elif p.bc.top.uy.type == BCType.FIXED or p.bc.top.uy.type == BCType.VELOCITY:
                            add(kuy, kuy, 1)
                        elif p.bc.top.uy.type == BCType.TRACTION_FREE:
                            # Coordinate-aware physical free-surface row.  It
                            # is a discretisation property, not a BP3-only BC.
                            wy = finite_difference_weights(g.y[0], g.y[:3], 1)
                            normal_scale = dx_loc / G
                            for iy_d, weight in enumerate(wy):
                                add(
                                    kuy,
                                    kuy + 2 * iy_d,
                                    normal_scale * (lam + 2 * G) * weight,
                                )
                            add(kuy, kuy - (Ny+1)*2, normal_scale * G * cosa / dx_loc)
                            add(kuy, kuy + (Ny+1)*2, -normal_scale * G * cosa / dx_loc)
                            coefficient = normal_scale * lam / (2 * dx_loc)
                            add(kuy, kux, coefficient)
                            add(kuy, kux + 2, coefficient)
                            add(kuy, kux - (Ny+1)*2, -coefficient)
                            add(kuy, kux - (Ny+1)*2 + 2, -coefficient)
                        else:
                            raise ValueError(f"BC type: {p.bc.top.uy.type} is not supported for top boundary yet.")
                    elif iy == Ny - 1 and use_bottom_fault_intersection and ix == mid:
                        add(kuy, kuy, -1)
                        add(kuy, kuy + (Ny+1)*2, 1)
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
                        # Shared MATLAB-validated shear-traction continuity.
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
                    if iy == 0 and use_surface_fault_intersection and ix == mid:
                        # Fault-line ghost above the free surface: zero
                        # curvature along the fault, not zero shear traction.
                        add(kux, kux, 1)
                        add(kux, kux + 2, -2)
                        add(kux, kux + 4, 1)
                    elif iy == 0: #top boundary (y=0 / free surface)
                        if p.bc.top.ux.type == BCType.FIXED or p.bc.top.ux.type == BCType.VELOCITY:
                            add(kux, kux, 1)
                            add(kux, kux + 2, 1)
                        elif p.bc.top.ux.type == BCType.FREE:
                            add(kux, kux, 1); add(kux, kux + 2, -1)
                        elif p.bc.top.ux.type == BCType.TRACTION_FREE:
                            # Same complete, coordinate-aware free-surface
                            # equation for every case that requests it.
                            wy = finite_difference_weights(g.y[0], g.y[:3], 1)
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
                            raise ValueError(f"Unknown BC type: {p.bc.top.ux.type}")
                    elif iy == Ny:
                        if p.bc.bottom.ux.type == BCType.FIXED or p.bc.bottom.ux.type == BCType.VELOCITY:
                            add(kux, kux, 1)
                            add(kux, kux - 2, 1)
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
                        # Shared MATLAB-validated normal-traction continuity.
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

        is_groningen = self._case_type == "groningen"
        use_surface_fault_intersection = (
            p.fault_surface_treatment
            == FaultSurfaceTreatment.FREE_SURFACE_FAULT_INTERSECTION
        )
        use_bottom_fault_intersection = (
            p.fault_bottom_treatment
            == FaultBottomTreatment.DEEP_BOUNDARY_FAULT_INTERSECTION
        )

        def apply_uy_boundary_profile(iy, boundary, reserved=()):
            """Apply one top/bottom uy velocity profile with row precedence."""
            indices = self._ix_uy_y_boundaries
            if reserved:
                indices = indices[~np.isin(indices, reserved)]

            if boundary.profile == BoundaryProfile.FULL:
                RH[self._kuy[iy, indices]] = boundary.value
            elif boundary.profile == BoundaryProfile.POSITIVE_FAULT_BLOCK:
                positive = indices[indices > mid]
                RH[self._kuy[iy, positive]] = boundary.value
            elif boundary.profile == BoundaryProfile.NEGATIVE_FAULT_BLOCK:
                negative = indices[indices <= mid]
                RH[self._kuy[iy, negative]] = boundary.value
            elif boundary.profile == BoundaryProfile.ANTISYMMETRIC_ABOUT_FAULT:
                negative = indices[indices <= mid]
                positive = indices[indices > mid]
                RH[self._kuy[iy, negative]] = -boundary.value
                RH[self._kuy[iy, positive]] = boundary.value
            else:
                raise ValueError(f"Unsupported boundary profile: {boundary.profile}")

        # --- uy block (exact branch priority) ---
        if p.bc.left.uy.type == BCType.VELOCITY:
            RH[self._kuy[:, 0]] = 2.0 * p.bc.left.uy.value
        if p.bc.right.uy.type == BCType.VELOCITY:
            RH[self._kuy[:, Nx]] = 2.0 * p.bc.right.uy.value

        if p.bc.top.uy.type == BCType.VELOCITY:
            reserved = (mid, mid + 1) if use_surface_fault_intersection else ()
            apply_uy_boundary_profile(0, p.bc.top.uy, reserved)

        if p.bc.bottom.uy.type == BCType.VELOCITY:
            reserved = (mid,) if use_bottom_fault_intersection else ()
            apply_uy_boundary_profile(Ny - 1, p.bc.bottom.uy, reserved)

        iy_int = self._iy_int
        RH[self._kuy[iy_int, mid]] = V[iy_int]
        if use_surface_fault_intersection:
            RH[self._kuy[0, mid]] = V[0]
        if use_bottom_fault_intersection:
            RH[self._kuy[Ny - 1, mid]] = V[Ny - 1]

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
            RH[self._kux[0, self._ix_ux_all]] = 2.0 * p.bc.top.ux.value
        if p.bc.bottom.ux.type == BCType.VELOCITY:
            RH[self._kux[Ny, self._ix_ux_all]] = 2.0 * p.bc.bottom.ux.value

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
