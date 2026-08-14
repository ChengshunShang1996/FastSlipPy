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
import numpy as np

from fastslippy.pre_processing.model_parameters import ModelParameters
from fastslippy.utilities.math_util import MathUtil

class Grid:
    """
    Builds and stores all spatial coordinate arrays needed by the model.
    """

    @staticmethod
    def _piecewise_stretch_axis(length: float, n_points: int, inner_length: float,
                                inner_points: int, power: int) -> np.ndarray:
        """
        One-sided piecewise-stretched axis on [0, length].
        """
        n_total = n_points - 1
        n_inner = inner_points - 1

        s = np.linspace(0.0, 1.0, n_points)
        if n_inner >= n_total:
            return length * s

        sb = n_inner / n_total
        b = n_total * inner_length / n_inner

        out = b * s
        outer = s > sb
        if np.any(outer):
            chi = (s[outer] - sb) / (1.0 - sb)
            out[outer] = b * s[outer] + (length - b) * np.power(chi, power)
        return out

    @staticmethod
    def _piecewise_stretch_metric(length: float, n_points: int, inner_length: float,
                                  inner_points: int, power: int,
                                  index_positions: np.ndarray) -> np.ndarray:
        """
        Analytic local spacing dx/dj for the piecewise stretch map x(j), where j is
        the grid index coordinate (node spacing is one index unit).
        """
        n_total = n_points - 1
        n_inner = inner_points - 1
        if n_inner >= n_total:
            return np.full_like(index_positions, length / n_total, dtype=float)

        sb = n_inner / n_total
        b = n_total * inner_length / n_inner
        s = np.clip(np.asarray(index_positions, dtype=float) / n_total, 0.0, 1.0)

        metric = np.empty_like(s, dtype=float)
        inner = s <= sb
        metric[inner] = b / n_total
        if np.any(~inner):
            chi = (s[~inner] - sb) / (1.0 - sb)
            metric[~inner] = (
                b + (length - b) * power * np.power(chi, power - 1) / (1.0 - sb)
            ) / n_total
        return metric

    @classmethod
    def _symmetric_piecewise_stretch_axis(cls, length: float, n_points: int, inner_length: float,
                                          inner_points: int, power: int) -> np.ndarray:
        """
        Symmetric piecewise-stretched axis on [-length/2, length/2].
        """
        half_intervals = (n_points - 1) // 2
        half_points = half_intervals + 1
        inner_half_length = inner_length / 2.0
        inner_half_points = (inner_points + 1) // 2

        positive = cls._piecewise_stretch_axis(
            length=length / 2.0,
            n_points=half_points,
            inner_length=inner_half_length,
            inner_points=inner_half_points,
            power=power,
        )
        return np.concatenate((-positive[:0:-1], positive))

    @classmethod
    def _symmetric_piecewise_stretch_metric(cls, length: float, n_points: int, inner_length: float,
                                            inner_points: int, power: int,
                                            index_positions: np.ndarray) -> np.ndarray:
        """
        Analytic local spacing dx/dj for the symmetric x(j) map around the centre.
        """
        half_intervals = (n_points - 1) // 2
        half_points = half_intervals + 1
        inner_half_length = inner_length / 2.0
        inner_half_points = (inner_points + 1) // 2
        center = float(half_intervals)
        mirrored = np.abs(np.asarray(index_positions, dtype=float) - center)
        return cls._piecewise_stretch_metric(
            length=length / 2.0,
            n_points=half_points,
            inner_length=inner_half_length,
            inner_points=inner_half_points,
            power=power,
            index_positions=mirrored,
        )

    @staticmethod
    def _staggered_from_centers(nodes: np.ndarray) -> np.ndarray:
        """
        Build staggered coordinates from nodal coordinates (supports nonuniform spacing).
        """
        staggered = np.empty(nodes.size + 1, dtype=float)
        staggered[1:-1] = 0.5 * (nodes[:-1] + nodes[1:])
        staggered[0] = nodes[0] - 0.5 * (nodes[1] - nodes[0])
        staggered[-1] = nodes[-1] + 0.5 * (nodes[-1] - nodes[-2])
        return staggered

    @classmethod
    def _max_cell_for_axis(cls, length: float, n_points: int, inner_length: float,
                           inner_points: int, power: int, symmetric: bool) -> float:
        if symmetric:
            coords = cls._symmetric_piecewise_stretch_axis(
                length=length,
                n_points=n_points,
                inner_length=inner_length,
                inner_points=inner_points,
                power=power,
            )
        else:
            coords = cls._piecewise_stretch_axis(
                length=length,
                n_points=n_points,
                inner_length=inner_length,
                inner_points=inner_points,
                power=power,
            )
        return float(np.max(np.diff(coords)))

    @classmethod
    def _suggest_points_for_max_cell(cls, *, length: float, inner_length: float, inner_points: int,
                                     power: int, max_cell_size: float, symmetric: bool,
                                     current_points: int) -> int:
        min_points = max(current_points, inner_points + 2)
        if min_points % 2 == 0:
            min_points += 1
        for n_points in range(min_points, 10001, 2):
            max_cell = cls._max_cell_for_axis(
                length=length,
                n_points=n_points,
                inner_length=inner_length,
                inner_points=inner_points,
                power=power,
                symmetric=symmetric,
            )
            if max_cell <= max_cell_size:
                return n_points
        return -1

    def __init__(self, p: ModelParameters):
        
        self.p = p
        self.sina = MathUtil.sind(p.alpha)
        self.cosa = MathUtil.cosd(p.alpha)

        Nx, Ny = p.Nx, p.Ny

        if p.x_stretch_enabled:
            self.x = self._symmetric_piecewise_stretch_axis(
                length=p.xsize,
                n_points=Nx,
                inner_length=p.x_stretch_inner_size,
                inner_points=p.x_stretch_inner_points,
                power=p.x_stretch_power,
            )
        else:
            self.x = np.linspace(-p.xsize / 2, p.xsize / 2, Nx)

        if p.y_stretch_enabled:
            self.y = self._piecewise_stretch_axis(
                length=p.ysize,
                n_points=Ny,
                inner_length=p.y_stretch_inner_size,
                inner_points=p.y_stretch_inner_points,
                power=p.y_stretch_power,
            )
        else:
            self.y = np.linspace(0, p.ysize, Ny)

        self.dx_edges = np.diff(self.x)
        self.dy_edges = np.diff(self.y)
        if p.x_stretch_enabled and p.x_stretch_max_cell_size is not None:
            max_dx = float(np.max(self.dx_edges))
            if max_dx > p.x_stretch_max_cell_size:
                suggested_nx = self._suggest_points_for_max_cell(
                    length=p.xsize,
                    inner_length=p.x_stretch_inner_size,
                    inner_points=p.x_stretch_inner_points,
                    power=p.x_stretch_power,
                    max_cell_size=p.x_stretch_max_cell_size,
                    symmetric=True,
                    current_points=p.Nx,
                )
                suggestion = (
                    f" Suggested Nx={suggested_nx} (or larger odd value) while keeping "
                    f"x_stretch_inner_points={p.x_stretch_inner_points}."
                    if suggested_nx > 0
                    else " No feasible Nx suggestion found up to 10001 points."
                )
                raise ValueError(
                    f"x-stretch max cell size exceeded: max(dx)={max_dx:.6g} > "
                    f"x_stretch_max_cell_size={p.x_stretch_max_cell_size:.6g}.{suggestion}"
                )
        if p.y_stretch_enabled and p.y_stretch_max_cell_size is not None:
            max_dy = float(np.max(self.dy_edges))
            if max_dy > p.y_stretch_max_cell_size:
                suggested_ny = self._suggest_points_for_max_cell(
                    length=p.ysize,
                    inner_length=p.y_stretch_inner_size,
                    inner_points=p.y_stretch_inner_points,
                    power=p.y_stretch_power,
                    max_cell_size=p.y_stretch_max_cell_size,
                    symmetric=False,
                    current_points=p.Ny,
                )
                suggestion = (
                    f" Suggested Ny={suggested_ny} (or larger odd value) while keeping "
                    f"y_stretch_inner_points={p.y_stretch_inner_points}."
                    if suggested_ny > 0
                    else " No feasible Ny suggestion found up to 10001 points."
                )
                raise ValueError(
                    f"y-stretch max cell size exceeded: max(dy)={max_dy:.6g} > "
                    f"y_stretch_max_cell_size={p.y_stretch_max_cell_size:.6g}.{suggestion}"
                )
        self.dx = float(np.mean(self.dx_edges))
        self.dy = float(np.mean(self.dy_edges))
        self.dy_fault = np.gradient(self.y)
        self.is_nonuniform_x = not np.allclose(self.dx_edges, self.dx_edges[0])
        self.is_nonuniform_y = not np.allclose(self.dy_edges, self.dy_edges[0])
        self.is_nonuniform = self.is_nonuniform_x or self.is_nonuniform_y

        if p.x_stretch_enabled:
            self.metric_x = self._symmetric_piecewise_stretch_metric(
                length=p.xsize,
                n_points=Nx,
                inner_length=p.x_stretch_inner_size,
                inner_points=p.x_stretch_inner_points,
                power=p.x_stretch_power,
                index_positions=np.arange(Nx, dtype=float),
            )
            self.metric_xp = self._symmetric_piecewise_stretch_metric(
                length=p.xsize,
                n_points=Nx,
                inner_length=p.x_stretch_inner_size,
                inner_points=p.x_stretch_inner_points,
                power=p.x_stretch_power,
                index_positions=np.arange(Nx + 1, dtype=float) - 0.5,
            )
        else:
            self.metric_x = np.full(Nx, self.dx, dtype=float)
            self.metric_xp = np.full(Nx + 1, self.dx, dtype=float)

        if p.y_stretch_enabled:
            self.metric_y = self._piecewise_stretch_metric(
                length=p.ysize,
                n_points=Ny,
                inner_length=p.y_stretch_inner_size,
                inner_points=p.y_stretch_inner_points,
                power=p.y_stretch_power,
                index_positions=np.arange(Ny, dtype=float),
            )
            self.metric_yp = self._piecewise_stretch_metric(
                length=p.ysize,
                n_points=Ny,
                inner_length=p.y_stretch_inner_size,
                inner_points=p.y_stretch_inner_points,
                power=p.y_stretch_power,
                index_positions=np.arange(Ny + 1, dtype=float) - 0.5,
            )
        else:
            self.metric_y = np.full(Ny, self.dy, dtype=float)
            self.metric_yp = np.full(Ny + 1, self.dy, dtype=float)

        # Pressure / staggered nodes
        self.xp = self._staggered_from_centers(self.x)
        self.yp = self._staggered_from_centers(self.y)


        # Rotated coordinate helpers (kept for post-processing / plotting)
        self.Xuy  = self.y[:, None] * self.cosa + self.xp[None, :]
        self.Yuy  = self.y[:, None] * self.sina + 0 * self.xp[None, :]
        self.Xux  = self.yp[:, None] * self.cosa + self.x[None, :]
        self.Yux  = self.yp[:, None] * self.sina + 0 * self.x[None, :]
        self.Xtau = self.y[:, None] * self.cosa + self.x[None, :]
        self.Ytau = self.y[:, None] * self.sina + 0 * self.x[None, :]
        self.Xsigma = self.yp[1:Ny, None] * self.cosa + self.xp[None, 1:Nx]
        self.Ysigma = self.yp[1:Ny, None] * self.sina + 0 * self.xp[None, 1:Nx]

        # Total DOF count
        self.N = (Nx + 1) * (Ny + 1) * 2

        if self.p.run_mode == "debug":
            self.plot_grid()

    def plot_mesh(self):
        X0, Y0 = np.meshgrid(self.x, self.y)
        X = Y0 * self.cosa + X0
        Y = Y0 * self.sina
        fig, ax = plt.subplots()
        ax.plot(X, Y, 'k', linewidth=0.4)
        ax.plot(X.T, Y.T, 'k', linewidth=0.4)

        y_line = self.y
        x_line = np.zeros_like(self.y)   # fault 在 x = 0
        X_fault = y_line * self.cosa + x_line
        Y_fault = y_line * self.sina
        ax.plot(X_fault, Y_fault, 'm--', linewidth=2, label='fault')

        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title('2-D Mesh Grid')
        plt.tight_layout()
        fig.savefig(f"mesh.png", dpi=150)
        return fig
    
    def plot_grid(self):

        fig, ax = plt.subplots(figsize=(6, 6))

        X0, Y0 = np.meshgrid(self.x, self.y)
        X = Y0 * self.cosa + X0
        Y = Y0 * self.sina
        ax.plot(X, Y, 'k-', linewidth=0.5)
        ax.plot(X.T, Y.T, 'k-', linewidth=0.5)

        # ─────────────────────────────
        # 2. uy points（blue）
        # ─────────────────────────────
        ax.scatter(self.Xuy, self.Yuy, marker='^', c='blue', label='uy nodes')

        # ─────────────────────────────
        # 3. ux points（red）
        # ─────────────────────────────
        ax.scatter(self.Xux, self.Yux, marker='^', c='red', label='ux nodes')

        # ─────────────────────────────
        # 4. sigma points（green)
        # ─────────────────────────────

        ax.scatter(self.Xsigma, self.Ysigma, marker='s', c='green', label='sigma nodes')

        ax.scatter(self.Xtau, self.Ytau, s=5, c='black', label='tau nodes')

        XP0, YP0 = np.meshgrid(self.xp, self.yp)
        XP = YP0 * self.cosa + XP0
        YP = YP0 * self.sina
        ax.scatter(XP, YP, s=5, c='orange', label='pressure nodes')

        # ─────────────────────────────
        # 5. fault position（middle column）
        # ─────────────────────────────
        mid = len(self.x) // 2
        ax.axvline(self.x[mid], color='magenta', linestyle='--', label='fault')

        # ─────────────────────────────
        # 6. settings
        # ─────────────────────────────
        ax.set_aspect('equal')
        ax.set_title("Grid + Staggered Layout")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc='upper right', fontsize=8)

        plt.tight_layout()
        fig.savefig(f"grid.png", dpi=150)