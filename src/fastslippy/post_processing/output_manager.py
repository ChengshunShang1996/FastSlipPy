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
from xml.etree import ElementTree as ET

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
        self._history_count = 0
        self._iteration_offset = 0
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
        idx = self._history_index(it)
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
        if self.p.case_type != "california" or (it + self._iteration_offset) % self.p.output_interval:
            return
        idx = self._history_index(it)
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
                        uy, vy, ux, vx, dt: float, t: float,
                        *, fault_velocity=None, fault_traction=None,
                        pressure=None, pressure_left=None, pressure_right=None):
        """Save a restartable, single-time-level state.

        The optional algebraic fields are used by coupled integrators to save
        the end-of-step friction solution while retaining their internal stage
        values.  Calls using the historical signature remain valid.
        """
        fname = self.out / f"data_{checkpointer + it}.npz"
        velocity = fault.V if fault_velocity is None else fault_velocity
        traction = fault.tau if fault_traction is None else fault_traction
        pore_pressure = (
            np.zeros_like(fault.U) if pressure is None else pressure
        )
        extra = {}
        if pressure_left is not None and pressure_right is not None:
            extra.update(Pl=pressure_left, Pr=pressure_right)
        np.savez(fname,
                 U=fault.U, V=velocity, tau=traction, sigma=fault.sigma,
                 P=pore_pressure,
                 theta=fault.theta, dt=dt, t=t,
                 tauqs=tauqs, sigmaqs=sigmaqs,
                 uy=uy, vy=vy, ux=ux, vx=vx,
                 time_integrator=np.asarray(self.p.time_integrator.value),
                 state_time_level=np.asarray("end"), **extra)

    def _history_index(self, it):
        interval = self.p.output_interval
        return (self._history_count
                + (it + self._iteration_offset) // interval
                - self._iteration_offset // interval - 1)

    def restore_history(self, checkpoint_time, iteration, tau0):
        """Retain accepted samples through the restart time, then reserve new slots."""
        self._iteration_offset = iteration
        names = ("Um", "Vm", "taum", "sigmam", "Pm", "thetam", "dtm", "tm", "tau0")
        surfaces = ("bp3_surface_disp1", "bp3_surface_disp2",
                    "bp3_surface_vel1", "bp3_surface_vel2")
        path = self.out / "dataall.npz"
        old = {}
        if path.exists():
            with np.load(path) as saved:
                old = {key: saved[key] for key in saved.files}
        times = old.get("tm", np.empty(0))
        # Older files include unused zero-filled columns.
        keep = np.flatnonzero(np.isfinite(times) & (times > 0) & (times <= checkpoint_time))
        if keep.size and np.any(np.diff(times[keep]) <= 0):
            raise ValueError("Checkpoint history times must be strictly increasing.")
        self._history_count = self._written_count = keep.size
        interval = self.p.output_interval
        capacity = keep.size + (iteration + self.p.Nt) // interval - iteration // interval
        for name in names:
            current = getattr(self, name)
            restored = np.zeros(current.shape[:-1] + (capacity,))
            if keep.size:
                if name == "tau0" and name not in old:
                    restored[..., :keep.size] = tau0[:, None]
                else:
                    restored[..., :keep.size] = old[name][..., keep]
            setattr(self, name, restored)
        for name in surfaces:
            restored = np.full((8, capacity), np.nan)
            if name in old:
                restored[:, :keep.size] = old[name][:, keep]
            setattr(self, "_" + name, restored)
        self._bp3_surface_x = old.get("bp3_surface_x")

    def save_all(self):
        fname = self.out / "dataall.npz"
        arrays = dict(
            Um=self.Um, Vm=self.Vm, taum=self.taum,
            sigmam=self.sigmam, Pm=self.Pm, thetam=self.thetam,
            dtm=self.dtm, tm=self.tm,
            tau0=self.tau0,
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
        arrays = {name: value if name == "bp3_surface_x"
                  else value[..., :self._written_count]
                  for name, value in arrays.items()}
        np.savez(fname, **arrays)

    def close(self):
        self._logfile.close()

    def load_checkpoint(self, checkpointer: int) -> dict:
        fname = self.out / f"data_{checkpointer}.npz"
        with np.load(fname) as checkpoint:
            return dict(checkpoint)

    @staticmethod
    def _checked_vtk_array(name, values, shape):
        """Return a finite array with the shape required by the VTK schema."""
        array = np.asarray(values)
        if array.shape != shape:
            raise ValueError(
                f"{name} has shape {array.shape}; expected {shape}."
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values.")
        return array

    @staticmethod
    def _vtk_quad_mesh(meshio, X, Y, point_data, cell_data):
        """Build a quad mesh whose points and cells use C-order indexing."""
        n_rows, n_cols = X.shape
        points = np.column_stack(
            (
                X.ravel(order="C"),
                Y.ravel(order="C"),
                np.zeros(n_rows * n_cols),
            )
        )
        row, col = np.meshgrid(
            np.arange(n_rows - 1),
            np.arange(n_cols - 1),
            indexing="ij",
        )
        row = row.ravel()
        col = col.ravel()
        lower_left = row * n_cols + col
        quads = np.column_stack(
            (
                lower_left,
                lower_left + 1,
                lower_left + n_cols + 1,
                lower_left + n_cols,
            )
        )
        return meshio.Mesh(
            points=points,
            cells=[("quad", quads)],
            point_data=point_data,
            cell_data=cell_data,
        )

    @staticmethod
    def _write_vtu(meshio, path: Path, mesh):
        """Write a compressed VTU atomically so failed writes are not exposed."""
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            meshio.write(
                str(temporary),
                mesh,
                file_format="vtu",
                binary=True,
                compression="zlib",
            )
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _vtk_vector(x_basis, y_basis, cosa, sina):
        """Convert the solver's skew-basis components to global vectors."""
        return np.column_stack(
            (
                (x_basis + cosa * y_basis).ravel(order="C"),
                (sina * y_basis).ravel(order="C"),
                np.zeros(x_basis.size),
            )
        )

    def _update_vtk_collection(self, it: int, t: float):
        """Atomically update the ParaView collection with physical times."""
        vtk_dir = self.out / "vtu_results"
        collection_path = vtk_dir / "results.pvd"
        steps = {}
        if collection_path.exists():
            try:
                root = ET.parse(collection_path).getroot()
            except (ET.ParseError, OSError) as exc:
                raise RuntimeError(
                    f"Cannot update invalid VTK collection {collection_path}."
                ) from exc
            for dataset in root.findall(".//DataSet"):
                if (
                    dataset.get("group") != "domain"
                    or dataset.get("part") != "0"
                ):
                    continue
                filename = Path(dataset.get("file", "")).stem
                try:
                    iteration = int(filename.rsplit("it", 1)[1])
                    time_value = float(dataset.get("timestep", "nan"))
                except (IndexError, ValueError):
                    continue
                if np.isfinite(time_value):
                    steps[iteration] = time_value

        # Restarting from an earlier checkpoint invalidates later frames in an
        # existing collection.  Leave their files untouched, but stop exposing
        # them through ParaView's active time series.
        steps = {
            iteration: time_value
            for iteration, time_value in steps.items()
            if iteration <= int(it)
        }
        steps[int(it)] = float(t)
        vtk_file = ET.Element(
            "VTKFile",
            type="Collection",
            version="0.1",
            byte_order="LittleEndian",
        )
        collection = ET.SubElement(vtk_file, "Collection")
        for iteration, time_value in sorted(steps.items()):
            stamp = format(time_value, ".17g")
            ET.SubElement(
                collection,
                "DataSet",
                timestep=stamp,
                group="domain",
                part="0",
                file=f"left_it{iteration:05d}.vtu",
            )
            ET.SubElement(
                collection,
                "DataSet",
                timestep=stamp,
                group="domain",
                part="1",
                file=f"right_it{iteration:05d}.vtu",
            )
            ET.SubElement(
                collection,
                "DataSet",
                timestep=stamp,
                group="fault",
                part="0",
                file=f"fault_it{iteration:05d}.vtu",
            )

        tree = ET.ElementTree(vtk_file)
        ET.indent(tree, space="  ")
        temporary = collection_path.with_name(".results.pvd.tmp")
        try:
            tree.write(
                temporary,
                encoding="utf-8",
                xml_declaration=True,
            )
            temporary.replace(collection_path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def write_vtk(
        self,
        it: int,
        grid,
        ux,
        uy,
        vx,
        vy,
        tauqs,
        sigmaqs,
        fault,
        t: float,
    ):
        """Write fault-aware field VTUs and a physical-time PVD collection.

        The two domain meshes share the geometric fault trace but own separate
        point arrays there.  This preserves the jump between ``uy[:, mid]`` and
        ``uy[:, mid + 1]`` when ParaView applies ``displacement_m``.  Both
        solver displacement components are interpolated to the tau grid
        without interpolating the y-basis motion across the fault.
        Quasi-static normal stress is cell data on the same quad topology.
        """
        try:
            import meshio
        except ImportError as exc:
            raise RuntimeError(
                "VTK output requires the declared runtime dependency meshio."
            ) from exc

        if not isinstance(it, (int, np.integer)) or int(it) < 0:
            raise ValueError("it must be a non-negative integer.")
        if not np.isfinite(t):
            raise ValueError("VTK time must be finite.")

        Ny, Nx = grid.p.Ny, grid.p.Nx
        mid = Nx // 2
        if Nx < 3 or mid < 1 or mid + 1 >= Nx:
            raise ValueError("VTK fault output requires at least three x nodes.")

        ux = self._checked_vtk_array("ux", ux, (Ny + 1, Nx))
        uy = self._checked_vtk_array("uy", uy, (Ny, Nx + 1))
        vx = self._checked_vtk_array("vx", vx, (Ny + 1, Nx))
        vy = self._checked_vtk_array("vy", vy, (Ny, Nx + 1))
        tauqs = self._checked_vtk_array("tauqs", tauqs, (Ny, Nx))
        sigmaqs = self._checked_vtk_array(
            "sigmaqs", sigmaqs, (Ny - 1, Nx - 1)
        )

        x = self._checked_vtk_array("grid.x", grid.x, (Nx,))
        y = self._checked_vtk_array("grid.y", grid.y, (Ny,))
        xp = self._checked_vtk_array("grid.xp", grid.xp, (Nx + 1,))
        yp = self._checked_vtk_array("grid.yp", grid.yp, (Ny + 1,))
        Xtau = self._checked_vtk_array(
            "grid.Xtau", grid.Xtau, (Ny, Nx)
        )
        Ytau = self._checked_vtk_array(
            "grid.Ytau", grid.Ytau, (Ny, Nx)
        )
        if np.any(np.diff(xp) <= 0.0) or np.any(np.diff(yp) <= 0.0):
            raise ValueError("VTK interpolation requires increasing grid axes.")

        fault_data = {}
        for name, values in (
            ("slip_U_m", fault.U),
            ("slip_rate_V_m_per_s", fault.V),
            ("shear_stress_Pa", fault.tau),
            ("normal_stress_Pa", fault.sigma),
        ):
            fault_data[name] = self._checked_vtk_array(
                name, values, (Ny,)
            )
        theta_values = self._checked_vtk_array(
            "fault.theta", fault.theta, (Ny,)
        )
        theta = np.real(theta_values)
        fault_data["state_theta_s"] = self._checked_vtk_array(
            "state_theta_s", theta, (Ny,)
        )
        fault_data["time_s"] = np.full(Ny, float(t))

        # Coordinate-aware interpolation from the two staggered displacement
        # grids to tau nodes.  The fault value is overwritten independently on
        # each split mesh below, so y-basis fields never cross the fault.
        weight_y = ((y - yp[:-1]) / (yp[1:] - yp[:-1]))[:, None]
        displacement_x_basis = (
            (1.0 - weight_y) * ux[:-1, :] + weight_y * ux[1:, :]
        )
        velocity_x_basis = (
            (1.0 - weight_y) * vx[:-1, :] + weight_y * vx[1:, :]
        )
        weight_x = ((x - xp[:-1]) / (xp[1:] - xp[:-1]))[None, :]
        displacement_y_basis = (
            (1.0 - weight_x) * uy[:, :-1] + weight_x * uy[:, 1:]
        )
        velocity_y_basis = (
            (1.0 - weight_x) * vy[:, :-1] + weight_x * vy[:, 1:]
        )

        left = slice(0, mid + 1)
        right = slice(mid, Nx)
        u_y_left = displacement_y_basis[:, left].copy()
        u_y_right = displacement_y_basis[:, right].copy()
        v_y_left = velocity_y_basis[:, left].copy()
        v_y_right = velocity_y_basis[:, right].copy()
        u_y_left[:, -1] = uy[:, mid]
        u_y_right[:, 0] = uy[:, mid + 1]
        v_y_left[:, -1] = vy[:, mid]
        v_y_right[:, 0] = vy[:, mid + 1]

        tau_left = tauqs[:, left].copy()
        tau_right = tauqs[:, right].copy()
        tau_left[:, -1] = tauqs[:, mid - 1]
        tau_right[:, 0] = tauqs[:, mid + 1]

        def domain_mesh(side, u_y, v_y, tau, sigma):
            u_x = displacement_x_basis[:, side]
            v_x = velocity_x_basis[:, side]
            return self._vtk_quad_mesh(
                meshio,
                Xtau[:, side],
                Ytau[:, side],
                point_data={
                    "displacement_m": self._vtk_vector(
                        u_x, u_y, grid.cosa, grid.sina
                    ),
                    "velocity_m_per_s": self._vtk_vector(
                        v_x, v_y, grid.cosa, grid.sina
                    ),
                    "ux_on_tau_m": u_x.ravel(order="C"),
                    "uy_on_tau_m": u_y.ravel(order="C"),
                    "vx_on_tau_m_per_s": v_x.ravel(order="C"),
                    "vy_on_tau_m_per_s": v_y.ravel(order="C"),
                    "shear_stress_quasistatic_Pa": tau.ravel(order="C"),
                },
                cell_data={
                    "normal_stress_quasistatic_Pa": [
                        sigma.ravel(order="C")
                    ]
                },
            )

        mesh_left = domain_mesh(
            left, u_y_left, v_y_left, tau_left, sigmaqs[:, :mid]
        )
        mesh_right = domain_mesh(
            right, u_y_right, v_y_right, tau_right, sigmaqs[:, mid:]
        )

        fault_x = y * grid.cosa
        fault_y = y * grid.sina
        fault_points = np.column_stack(
            (fault_x, fault_y, np.zeros(Ny))
        )
        segment = np.arange(Ny - 1)
        mesh_fault = meshio.Mesh(
            points=fault_points,
            cells=[("line", np.column_stack((segment, segment + 1)))],
            point_data=fault_data,
        )

        vtk_dir = self.out / "vtu_results"
        vtk_dir.mkdir(parents=True, exist_ok=True)
        self._write_vtu(
            meshio, vtk_dir / f"left_it{int(it):05d}.vtu", mesh_left
        )
        self._write_vtu(
            meshio, vtk_dir / f"right_it{int(it):05d}.vtu", mesh_right
        )
        self._write_vtu(
            meshio, vtk_dir / f"fault_it{int(it):05d}.vtu", mesh_fault
        )
        self._update_vtk_collection(int(it), float(t))
