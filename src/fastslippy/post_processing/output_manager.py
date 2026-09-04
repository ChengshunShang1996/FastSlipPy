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
from datetime import datetime
from pathlib import Path

from fastslippy.pre_processing.model_parameters import BCType, ModelParameters
from fastslippy.solver.fault_state import FaultState

class OutputManager:
    """
    Handles:
      - in-memory snapshot arrays
      - ASCII log file (output.txt)
      - NumPy checkpoints
    """

    def __init__(self, p: ModelParameters, output_dir: Path = Path("."),
                 append_log: bool = False):
        self.p   = p
        self.out = output_dir
        self.out.mkdir(parents=True, exist_ok=True)
        Ny, Nx = p.Ny, p.Nx
        n = p.Nt // p.output_interval

        self.Um      = np.zeros((Ny, n))
        self.Vm      = np.zeros((Ny, n))
        self.taum    = np.zeros((Ny, n))
        self.sigmam  = np.zeros((Ny, n))
        self.Pm      = np.zeros((Ny, n))
        self.thetam  = np.zeros((Ny, n))
        self.dtm     = np.zeros(n)
        self.tm      = np.zeros(n)
        # They require a lot of memory, so we don't store them in memory by default.
        # self.taumall    = np.zeros((Ny, Nx, n))
        # self.sigmamall  = np.zeros((Ny-1, Nx-1, n))
        # self.uymall     = np.zeros((Ny, Nx+1, n))
        # self.vymall     = np.zeros((Ny, Nx+1, n))
        # self.uxmall     = np.zeros((Ny+1, Nx, n))
        # self.vxmall     = np.zeros((Ny+1, Nx, n))
        self.tau0   = np.zeros((Ny, n))
        self._written_count = 0
        self._bp3_surface_x = None
        self._bp3_surface_disp1 = np.zeros((8, n))
        self._bp3_surface_disp2 = np.zeros((8, n))
        self._bp3_surface_vel1 = np.zeros((8, n))
        self._bp3_surface_vel2 = np.zeros((8, n))

        self._logfile = open(self.out / "output.txt", "a" if append_log else "w")

    def log(self, it: int, t: float, dt: float, V: np.ndarray, U: np.ndarray,
            checkpointer: int = 0):
        yr = 365 * 24 * 3600
        line = (f"it={checkpointer+it}, t={t/yr:.6f} yr, dt={dt:.3e}, "
                f"maxV={V.max():.3e}, minV={V.min():.3e}, maxU={U.max():.6f}\n")
        self._logfile.write(line)
        self._logfile.flush()

    def write_memory(self, it: int,
                     U, V, tau, sigma, P, theta, dt, t,
                     tauqs, sigmaqs, uy, vy, ux, vx, tau0):
        idx = it // self.p.output_interval - 1
        self.Um[:, idx]     = U
        self.Vm[:, idx]     = V
        self.taum[:, idx]   = tau
        self.tau0[:, idx]   = tau0
        self.sigmam[:, idx] = sigma
        self.Pm[:, idx]     = P
        self.thetam[:, idx] = theta
        self.dtm[idx]       = dt
        self.tm[idx]        = t
        self._written_count = max(self._written_count, idx + 1)
        # self.taumall[:, :, idx]   = tauqs
        # self.sigmamall[:, :, idx] = sigmaqs
        # self.uymall[:, :, idx]    = uy
        # self.vymall[:, :, idx]    = vy
        # self.uxmall[:, :, idx]    = ux
        # self.vxmall[:, :, idx]    = vx

        # field_fname = self.fields_dir / f"fields_it_{it}.npz"
        # np.savez_compressed(
        #     field_fname,
        #     tauqs=tauqs, sigmaqs=sigmaqs,
        #     uy=uy, vy=vy, ux=ux, vx=vx
        # )

    def record_bp3_surface(self, it: int, t: float, grid, ux, uy, vx, vy):
        """Record the eight SEAS BP3 free-surface stations."""
        if self.p.case_type != "california" or it % self.p.output_interval:
            return
        idx = it // self.p.output_interval - 1
        if idx < 0 or idx >= self._bp3_surface_disp1.shape[1]:
            return

        mid = self.p.Nx // 2
        core_dx = float(grid.x[mid + 1] - grid.x[mid])
        surface_x = np.array(
            [-32e3, -16e3, -8e3, core_dx / 2.0, -core_dx / 2.0,
             8e3, 16e3, 32e3],
            dtype=float,
        )
        self._bp3_surface_x = surface_x

        def interp(coords, values):
            return np.interp(surface_x, coords, values, left=np.nan, right=np.nan)

        normal_displacement = np.mean(ux[:2, :], axis=0)
        normal_velocity = np.mean(vx[:2, :], axis=0)
        un = interp(grid.x, normal_displacement)
        vn = interp(grid.x, normal_velocity)
        ut = interp(grid.xp, uy[0, :])
        vt = interp(grid.xp, vy[0, :])

        sides_loaded = (
            self.p.bc.left.uy.type == BCType.VELOCITY
            and self.p.bc.right.uy.type == BCType.VELOCITY
        )
        surface_side = np.array([-1, -1, -1, 1, -1, 1, 1, 1], dtype=float)
        rigid_rate = (
            np.zeros_like(surface_side)
            if sides_loaded
            else -surface_side * self.p.loading.V_p / 2.0
        )
        cosa = grid.cosa
        sina = grid.sina
        self._bp3_surface_disp1[:, idx] = (
            un + ut * cosa + rigid_rate * t * cosa
        )
        self._bp3_surface_disp2[:, idx] = ut * sina + rigid_rate * t * sina
        self._bp3_surface_vel1[:, idx] = vn + vt * cosa + rigid_rate * cosa
        self._bp3_surface_vel2[:, idx] = vt * sina + rigid_rate * sina

    @staticmethod
    def _matlab_round_positive(value: float) -> int:
        return int(np.floor(value + 0.5))

    def _bp3_element_size(self, grid) -> float:
        mid = self.p.Nx // 2
        return float(
            min(
                abs(grid.x[mid + 1] - grid.x[mid]),
                abs(grid.y[1] - grid.y[0]),
            )
        )

    def _write_bp3_common_header(
        self, handle, *, motion_name: str, element_size: float, nt: int
    ):
        p = self.p
        handle.write("# This is the file header:\n")
        handle.write("# problem=SEAS Benchmark BP3-QD\n")
        handle.write(f"# code={p.code_name}\n")
        if p.code_version:
            handle.write(f"# version={p.code_version}\n")
        handle.write(f"# modeler={p.modeler}\n")
        handle.write(f"# date={datetime.now().strftime('%Y/%m/%d')}\n")
        handle.write(f"# element size={element_size:g} m\n")
        handle.write(f"# motion={motion_name}\n")
        handle.write(f"# dip angle={p.alpha:g} degrees\n")
        handle.write(f"# num time steps={nt}\n")

    def _write_bp3_profile(
        self, filename: Path, *, field_name: str, description: str,
        grid, times, velocities, field, scale: float, indices,
        element_size: float,
    ):
        p = self.p
        with filename.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("# This is the file header:\n")
            handle.write("# problem=SEAS Benchmark BP3-QD\n")
            handle.write(f"# modeler={p.modeler}\n")
            handle.write(f"# date={datetime.now().strftime('%Y/%m/%d')}\n")
            handle.write(f"# code={p.code_name}\n")
            if p.code_version:
                handle.write(f"# code version={p.code_version}\n")
            handle.write(f"# element size={element_size:g} m\n")
            handle.write("# Row #1 = Distance down dip (m) with two zeros first\n")
            handle.write("# Column #1 = Time (s)\n")
            handle.write("# Column #2 = Max slip rate (log10 m/s)\n")
            handle.write(
                f"# Columns #3-{len(indices) + 2} = {description}\n"
            )
            handle.write(
                "# Computational domain size: down-dip "
                f"{p.ysize / 1e3:g} km, distance off fault "
                f"{p.xsize / 2e3:g} km, dip {p.alpha:g} degrees\n"
            )
            handle.write("# The line below lists the names of the data fields\n")
            handle.write("xd\n")
            handle.write(f"t max_slip_rate {field_name}\n")
            handle.write("# Here are the data\n")
            first_row = np.concatenate(([0.0, 0.0], grid.y[indices]))
            handle.write(" ".join(f"{value:14.6E}" for value in first_row) + "\n")
            tiny = np.finfo(float).tiny
            for column, time_value in enumerate(times):
                max_rate = np.log10(max(np.max(np.abs(velocities[:, column])), tiny))
                values = field[indices, column] * scale
                row = np.concatenate(([time_value, max_rate], values))
                handle.write(
                    f"{row[0]:21.13E} "
                    + " ".join(f"{value:14.6E}" for value in row[1:])
                    + "\n"
                )

    def write_bp3_outputs(self, grid):
        """Write MATLAB-compatible SEAS BP3 ASCII products in addition to NPZ."""
        if self.p.case_type != "california" or self._written_count == 0:
            return

        p = self.p
        nt = self._written_count
        times = self.tm[:nt]
        U = self.Um[:, :nt]
        V = self.Vm[:, :nt]
        tau = self.taum[:, :nt]
        sigma = self.sigmam[:, :nt]
        theta = self.thetam[:, :nt]
        output_dir = self.out / "output_BP3_QD"
        output_dir.mkdir(parents=True, exist_ok=True)
        motion_name = "thrust" if p.motion_sign > 0 else "normal"
        element_size = self._bp3_element_size(grid)
        tiny = np.finfo(float).tiny

        fault_stations = np.array(
            [0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 25, 30, 35]
        ) * 1e3
        fault_names = [
            "000", "025", "050", "075", "100", "125",
            "150", "175", "200", "250", "300", "350",
        ]
        for station, name in zip(fault_stations, fault_names):
            iy = int(np.argmin(np.abs(grid.y - station)))
            with (output_dir / f"fltst_dp{name}").open(
                "w", encoding="utf-8", newline="\n"
            ) as handle:
                self._write_bp3_common_header(
                    handle, motion_name=motion_name,
                    element_size=element_size, nt=nt,
                )
                handle.write(
                    f"# location=on fault, {station / 1e3:.1f} km down-dip distance\n"
                )
                handle.write("# Column #1 = Time (s)\n")
                handle.write("# Column #2 = Slip (m)\n")
                handle.write("# Column #3 = Slip rate (log10 m/s)\n")
                handle.write("# Column #4 = Shear stress (MPa)\n")
                handle.write("# Column #5 = Normal stress (MPa)\n")
                handle.write("# Column #6 = State (log10 s)\n")
                handle.write("# The line below lists the names of the data fields\n")
                handle.write("t slip slip_rate shear_stress normal_stress state\n")
                handle.write("# Here is the time-series data.\n")
                for column, time_value in enumerate(times):
                    values = (
                        time_value,
                        -U[iy, column],
                        np.log10(max(abs(V[iy, column]), tiny)),
                        -tau[iy, column] / 1e6,
                        sigma[iy, column] / 1e6,
                        np.log10(max(theta[iy, column], tiny)),
                    )
                    handle.write(
                        f"{values[0]:21.13E} "
                        + " ".join(f"{value:14.6E}" for value in values[1:])
                        + "\n"
                    )

        surface_names = [
            "srfst_fn-32", "srfst_fn-16", "srfst_fn-08", "srfst_fn+00",
            "srfst_fn-00", "srfst_fn+08", "srfst_fn+16", "srfst_fn+32",
        ]
        surface_nominal = np.array([-32, -16, -8, 0, 0, 8, 16, 32]) * 1e3
        if self._bp3_surface_x is not None:
            for station_idx, (name, nominal) in enumerate(
                zip(surface_names, surface_nominal)
            ):
                with (output_dir / name).open(
                    "w", encoding="utf-8", newline="\n"
                ) as handle:
                    self._write_bp3_common_header(
                        handle, motion_name=motion_name,
                        element_size=element_size, nt=nt,
                    )
                    handle.write(
                        f"# location=on surface, {nominal / 1e3:+g} km distance off-fault\n"
                    )
                    handle.write(
                        f"# sampled at x={self._bp3_surface_x[station_idx] / 1e3:+.4f} km\n"
                    )
                    handle.write("# Column #1 = Time (s)\n")
                    handle.write("# Column #2 = Displacement 1 (m)\n")
                    handle.write("# Column #3 = Displacement 2 (m)\n")
                    handle.write("# Column #4 = Velocity 1 (m/s)\n")
                    handle.write("# Column #5 = Velocity 2 (m/s)\n")
                    handle.write("# The line below lists the names of the data fields\n")
                    handle.write("t disp_1 disp_2 vel_1 vel_2\n")
                    handle.write("# Here is the time-series data.\n")
                    arrays = (
                        self._bp3_surface_disp1,
                        self._bp3_surface_disp2,
                        self._bp3_surface_vel1,
                        self._bp3_surface_vel2,
                    )
                    for column, time_value in enumerate(times):
                        values = [array[station_idx, column] for array in arrays]
                        handle.write(
                            f"{time_value:21.13E} "
                            + " ".join(f"{value:14.6E}" for value in values)
                            + "\n"
                        )

        profile_indices = np.flatnonzero(grid.y <= p.W_f)
        if profile_indices.size:
            stride = max(
                1,
                self._matlab_round_positive(500.0 / element_size),
            )
            selected = profile_indices[::stride]
            if selected[-1] != profile_indices[-1]:
                selected = np.append(selected, profile_indices[-1])
            self._write_bp3_profile(
                output_dir / "slip.dat", field_name="slip",
                description="Slip (m)", grid=grid, times=times,
                velocities=V, field=U, scale=-1.0, indices=selected,
                element_size=element_size,
            )
            self._write_bp3_profile(
                output_dir / "shear_stress.dat", field_name="shear_stress",
                description="Shear stress (MPa)", grid=grid, times=times,
                velocities=V, field=tau, scale=-1e-6, indices=selected,
                element_size=element_size,
            )
            self._write_bp3_profile(
                output_dir / "normal_stress.dat", field_name="normal_stress",
                description="Normal stress (MPa)", grid=grid, times=times,
                velocities=V, field=sigma, scale=1e-6, indices=selected,
                element_size=element_size,
            )

    def save_checkpoint(self, it: int, checkpointer: int,
                        fault: "FaultState", tauqs, sigmaqs,
                        uy, vy, ux, vx, dt: float, t: float):
        fname = self.out / f"data_{checkpointer + it}.npz"
        np.savez(fname,
                 U=fault.U, V=fault.V, tau=fault.tau, sigma=fault.sigma,
                 P=np.zeros_like(fault.U),  # placeholder; update as needed
                 theta=fault.theta, dt=dt, t=t,
                 tauqs=tauqs, sigmaqs=sigmaqs,
                 uy=uy, vy=vy, ux=ux, vx=vx)

    def save_all(self):
        fname = self.out / "dataall.npz"
        arrays = dict(
            Um=self.Um, Vm=self.Vm, taum=self.taum,
            sigmam=self.sigmam, Pm=self.Pm, thetam=self.thetam,
            dtm=self.dtm, tm=self.tm,
        )
        # Surface histories are small compared with the fault histories and
        # are especially useful when a long BP3 calculation is interrupted
        # before write_bp3_outputs() creates the ASCII station files.
        if self._bp3_surface_x is not None:
            arrays.update(
                bp3_surface_x=self._bp3_surface_x,
                bp3_surface_disp1=self._bp3_surface_disp1,
                bp3_surface_disp2=self._bp3_surface_disp2,
                bp3_surface_vel1=self._bp3_surface_vel1,
                bp3_surface_vel2=self._bp3_surface_vel2,
            )
        np.savez(fname, **arrays)

    def close(self):
        self._logfile.close()

    def load_checkpoint(self, checkpointer: int) -> dict:
        fname = self.out / f"data_{checkpointer}.npz"
        return dict(np.load(fname))

    def write_vtk_old(self, it: int, grid, ux, uy, vx, vy,
                  tauqs, sigmaqs, fault, t: float):
        """
        Write VTK files for ParaView visualisation.
 
        Two files per call (written to self.out/):
          fields_it{it:05d}.vtu  –  2-D quad mesh with stress / velocity
          fault_it{it:05d}.vtu   –  fault poly-line with slip / friction state
 
        Stress fields are stored as *cell data* (one value per quad) so
        ParaView renders discrete patches instead of interpolating between
        nodes.  Displacement / velocity fields are stored as *point data*
        and will be smoothly interpolated, which is appropriate for those
        quantities.
 
        Requires:  pip install meshio
        """
        try:
            import meshio
        except ImportError:
            print("[write_vtk] meshio not found – skipping VTK output. "
                  "Install with:  pip install meshio")
            return
 
        Ny, Nx = grid.p.Ny, grid.p.Nx
 
        # ── helper: node-centred field (Ny, Nx) → cell-centred (Ny-1, Nx-1) ──
        def to_cell(arr2d):
            return 0.25 * (arr2d[:-1, :-1] + arr2d[:-1, 1:]
                         + arr2d[1:,  :-1] + arr2d[1:,  1:])
 
        # ══════════════════════════════════════════════════════════════
        # 1.  2-D field mesh  (quad elements on the tau / stress grid)
        # ══════════════════════════════════════════════════════════════
 
        # Point coordinates – Xtau / Ytau already have shape (Ny, Nx)
        pts_2d = np.column_stack([
            grid.Xtau.ravel(order='C'),
            grid.Ytau.ravel(order='C'),
            np.zeros(Ny * Nx),
        ])
 
        # Quad connectivity: (Ny-1)*(Nx-1) cells
        ri, ci = np.meshgrid(np.arange(Ny - 1), np.arange(Nx - 1), indexing='ij')
        ri, ci = ri.ravel(), ci.ravel()
        idx    = lambda r, c: r * Nx + c
        quads  = np.column_stack([
            idx(ri,   ci),
            idx(ri,   ci+1),
            idx(ri+1, ci+1),
            idx(ri+1, ci),
        ])
 
        # ── point data: displacement / velocity (averaged to tau-grid shape) ──
        # ux is (Ny+1, Nx), uy is (Ny, Nx+1) → average to (Ny, Nx)
        ux_p = 0.5 * (ux[:Ny, :] + ux[1:Ny+1, :])
        uy_p = 0.5 * (uy[:, :Nx] + uy[:, 1:Nx+1])
        vx_p = 0.5 * (vx[:Ny, :] + vx[1:Ny+1, :])
        vy_p = 0.5 * (vy[:, :Nx] + vy[:, 1:Nx+1])
 
        n_pts = Ny * Nx
        zeros = np.zeros(n_pts)
 
        # Store as 3-component vectors (x, y, z=0) so ParaView's
        # 'Warp By Vector' filter can use them directly for deformation.
        displacement = np.column_stack([
            ux_p.ravel(order='C'),
            uy_p.ravel(order='C'),
            zeros,
        ])  # (n_pts, 3)
 
        velocity = np.column_stack([
            vx_p.ravel(order='C'),
            vy_p.ravel(order='C'),
            zeros,
        ])  # (n_pts, 3)
 
        point_data_2d = {
            "displacement_m": displacement,   # use with Warp By Vector
            "velocity_ms":    velocity,
        }
 
        # ── cell data: stress (no interpolation → true grid resolution) ──
        # tauqs is (Ny, Nx) → average to cell centres (Ny-1, Nx-1)
        # sigmaqs is already (Ny-1, Nx-1)
        tauqs_c   = to_cell(tauqs)
        sigmaqs_c = sigmaqs                      # (Ny-1, Nx-1) — native shape
 
        cell_data_2d = {
            "tauqs_Pa":   [tauqs_c.ravel(order='C')],
            "sigmaqs_Pa": [sigmaqs_c.ravel(order='C')],
        }
 
        mesh2d = meshio.Mesh(
            points=pts_2d,
            cells=[("quad", quads)],
            point_data=point_data_2d,
            cell_data=cell_data_2d,
        )
        meshio.write(str(self.out/ "vtu_results" / f"fields_it{it:05d}.vtu"), mesh2d)

    def write_vtk(self, it: int, grid, ux, uy, vx, vy,
                  tauqs, sigmaqs, fault, t: float):
        """
        Write VTK files for ParaView visualisation.
 
        Each field is written on its **own native staggered grid** — no
        cross-fault interpolation is ever performed.  The domain is split at
        the fault column so that Warp-By-Vector in ParaView reproduces the
        true discontinuous slip.
 
        Files written to self.out/ :
          left_it{it:05d}.vtu   – left half,  uy-grid  (Ny × mid+1 points)
          right_it{it:05d}.vtu  – right half, uy-grid  (Ny × mid+1 points)
          sigma_it{it:05d}.vtu  – sigma field on its native sigma-grid
          fault_it{it:05d}.vtu  – fault poly-line (slip, velocity, stress)
 
        Requires:  pip install meshio
        """
        try:
            import meshio
        except ImportError:
            print("[write_vtk] meshio not found – skipping VTK output. "
                  "Install with:  pip install meshio")
            return
 
        Ny, Nx = grid.p.Ny, grid.p.Nx
        mid = Nx // 2   # fault column (0-based) in the uy-node array (Nx+1 cols)
        #
        # Node layout reminder (from Grid.__init__):
        #   uy  : shape (Ny, Nx+1)  — y-displacement at  y[i], xp[j]
        #   ux  : shape (Ny+1, Nx)  — x-displacement at  yp[i], x[j]
        #   tauqs   : (Ny, Nx)      — on tau nodes  y[i],  x[j]
        #   sigmaqs : (Ny-1, Nx-1)  — on sigma nodes yp[1:Ny], xp[1:Nx]
        #
        # The fault sits between uy-columns  mid  and  mid+1  (xp[mid] = 0).
        # Left half  uses uy columns 0 … mid   (inclusive).
        # Right half uses uy columns mid … Nx  (inclusive, mid shared).
 
        # ══════════════════════════════════════════════════════════════
        # helper: build a quad mesh from 2-D coordinate + data arrays
        # ══════════════════════════════════════════════════════════════
        def _quad_mesh(X, Y, point_data=None, cell_data=None):
            """
            X, Y : (Nr, Nc) node coordinate arrays
            Returns a meshio.Mesh of (Nr-1)*(Nc-1) quads.
            point_data : dict  name → (Nr*Nc,) or (Nr*Nc, 3)
            cell_data  : dict  name → [(Nr-1)*(Nc-1),]
            """
            Nr, Nc = X.shape
            pts = np.column_stack([
                X.ravel(order="C"),
                Y.ravel(order="C"),
                np.zeros(Nr * Nc),
            ])
            ri, ci = np.meshgrid(np.arange(Nr - 1), np.arange(Nc - 1), indexing="ij")
            ri, ci = ri.ravel(), ci.ravel()
            node = lambda r, c: r * Nc + c
            quads = np.column_stack([
                node(ri,   ci),
                node(ri,   ci+1),
                node(ri+1, ci+1),
                node(ri+1, ci),
            ])
            return meshio.Mesh(
                points=pts,
                cells=[("quad", quads)],
                point_data=point_data or {},
                cell_data=cell_data  or {},
            )
 
        # ══════════════════════════════════════════════════════════════
        # 1a.  LEFT half  — uy columns 0 … mid  (fault col included)
        #      uy shape on this half: (Ny, mid+1)
        # ══════════════════════════════════════════════════════════════
        sl_L = slice(0, mid + 1)   # uy column slice for left side
 
        X_L = grid.Xuy[:, sl_L]   # (Ny, mid+1)
        Y_L = grid.Yuy[:, sl_L]
 
        uy_L = uy[:, sl_L]        # (Ny, mid+1) — left uy, unmodified
        vy_L = vy[:, sl_L]
 
        # tauqs on left tau-columns 0 … mid-1  → (Ny, mid) cell-centred later
        # For point data we keep tauqs at tau nodes; left columns: 0…mid-1
        # tauqs has shape (Ny, Nx); left tau columns cover x[0]…x[mid-1]
        # Interpolate tauqs to uy-node x-positions by averaging neighbours
        # (tau col j sits between uy cols j and j+1 for j=0…Nx-1)
        # uy col 0 → extrapolate from tau col 0
        # uy col j (1…mid-1) → average of tau cols j-1 and j
        # uy col mid (fault) → tau col mid-1 (left neighbour only)
        tauqs_L = np.zeros((Ny, mid + 1))
        tauqs_L[:, 0]        = tauqs[:, 0]
        tauqs_L[:, 1:mid]    = 0.5 * (tauqs[:, :mid-1] + tauqs[:, 1:mid])
        tauqs_L[:, mid]      = tauqs[:, mid - 1]   # one-sided at fault
 
        mesh_L = _quad_mesh(
            X_L, Y_L,
            point_data={
                "displacement_m": np.column_stack([
                    np.zeros(Ny * (mid + 1)),   # ux not defined on uy-grid; zero
                    uy_L.ravel(order="C"),
                    np.zeros(Ny * (mid + 1)),
                ]),
                "velocity_ms": np.column_stack([
                    np.zeros(Ny * (mid + 1)),
                    vy_L.ravel(order="C"),
                    np.zeros(Ny * (mid + 1)),
                ]),
                "tauqs_Pa": tauqs_L.ravel(order="C"),
            },
        )
        (self.out / "vtu_results").mkdir(exist_ok=True)
        meshio.write(str(self.out / "vtu_results" / f"left_it{it:05d}.vtu"), mesh_L)
 
        # ══════════════════════════════════════════════════════════════
        # 1b.  RIGHT half  — uy columns mid … Nx  (fault col shared)
        #      uy shape on this half: (Ny, Nx-mid+1)
        # ══════════════════════════════════════════════════════════════
        sl_R = slice(mid, Nx + 1)   # uy column slice for right side
        Nc_R = Nx + 1 - mid
 
        X_R = grid.Xuy[:, sl_R]   # (Ny, Nc_R)
        Y_R = grid.Yuy[:, sl_R]
 
        uy_R = uy[:, sl_R]
        vy_R = vy[:, sl_R]
 
        # tauqs interpolated to right uy-node positions
        # uy col mid   (local 0) → tau col mid (right neighbour only)
        # uy col mid+j (local j, j=1…Nc_R-2) → average tau cols mid+j-1, mid+j
        # uy col Nx    (local Nc_R-1) → extrapolate from tau col Nx-1
        tauqs_R = np.zeros((Ny, Nc_R))
        tauqs_R[:, 0]          = tauqs[:, mid]         # one-sided at fault
        tauqs_R[:, 1:Nc_R-1]   = 0.5 * (tauqs[:, mid:Nx-1] + tauqs[:, mid+1:Nx])
        tauqs_R[:, Nc_R - 1]   = tauqs[:, Nx - 1]
 
        mesh_R = _quad_mesh(
            X_R, Y_R,
            point_data={
                "displacement_m": np.column_stack([
                    np.zeros(Ny * Nc_R),
                    uy_R.ravel(order="C"),
                    np.zeros(Ny * Nc_R),
                ]),
                "velocity_ms": np.column_stack([
                    np.zeros(Ny * Nc_R),
                    vy_R.ravel(order="C"),
                    np.zeros(Ny * Nc_R),
                ]),
                "tauqs_Pa": tauqs_R.ravel(order="C"),
            },
        )
        meshio.write(str(self.out / "vtu_results" / f"right_it{it:05d}.vtu"), mesh_R)
 
        # ══════════════════════════════════════════════════════════════
        # 2.  Sigma field on its native sigma-grid
        #     sigmaqs : (Ny-1, Nx-1)  —  these ARE the cell-centre points.
        #     Store as point cloud (Vertices) with point_data so the
        #     count always matches: n_points == n_values.
        # ══════════════════════════════════════════════════════════════
        n_sig = (Ny - 1) * (Nx - 1)
        pts_sig = np.column_stack([
            grid.Xsigma.ravel(order="C"),
            grid.Ysigma.ravel(order="C"),
            np.zeros(n_sig),
        ])
        mesh_sig = meshio.Mesh(
            points=pts_sig,
            cells=[("vertex", np.arange(n_sig).reshape(-1, 1))],
            point_data={"sigmaqs_Pa": sigmaqs.ravel(order="C")},
        )
        meshio.write(str(self.out / "vtu_results" / f"sigma_it{it:05d}.vtu"), mesh_sig)
 
        # ══════════════════════════════════════════════════════════════
        # 3.  Fault poly-line
        # ══════════════════════════════════════════════════════════════
        X_f = grid.y * grid.cosa   # (Ny,)
        Y_f = grid.y * grid.sina
 
        pts_fault = np.column_stack([X_f, Y_f, np.zeros(Ny)])
        seg_i = np.arange(Ny - 1)
        lines = np.column_stack([seg_i, seg_i + 1])
 
        mesh_fault = meshio.Mesh(
            points=pts_fault,
            cells=[("line", lines)],
            point_data={
                "slip_U_m":         fault.U,
                "slip_rate_V_ms":   fault.V,
                "shear_stress_Pa":  fault.tau,
                "normal_stress_Pa": fault.sigma,
                "state_theta_s":    np.real(fault.theta).astype(float),
                "time_s":           np.full(Ny, t),
            },
        )
        meshio.write(str(self.out / "vtu_results" / f"fault_it{it:05d}.vtu"), mesh_fault)
