#/////////////////////////////////////////////////
__author__      = "Chengshun Shang (Utrecht University)"
__copyright__   = "Copyright (C) 2026-present by Chengshun Shang"
__version__     = "0.1.3"
__maintainer__  = "Chengshun Shang"
__email__       = "c.shang@uu.nl"
__status__      = "development"
__date__        = "May 5, 2026"
__license__     = "MIT License"
#/////////////////////////////////////////////////

from cProfile import label
import time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastslippy import FastSlipPy
from fastslippy.pre_processing.model_parameters import FaultMode, ModelParameters, TimeIntegrator

class RunFastSlipPy(FastSlipPy):
    """
    This can be customized for specific runs.
    """
    def _comparison_point_indices(self):
        """Return staggered-grid indices near (Lx/4, Ly/2)."""
        x_centre = 0.5 * (self.grid.x[0] + self.grid.x[-1])
        x_target = 0.5 * (x_centre + self.grid.x[-1])
        y_target = 0.5 * (self.grid.y[0] + self.grid.y[-1])

        ix_ux = int(np.argmin(np.abs(self.grid.x - x_target)))
        iy_ux = int(np.argmin(np.abs(self.grid.yp - y_target)))
        ix_uy = int(np.argmin(np.abs(self.grid.xp - x_target)))
        iy_uy = int(np.argmin(np.abs(self.grid.y - y_target)))
        x_sigma = self.grid.xp[1:self.p.Nx]
        y_sigma = self.grid.yp[1:self.p.Ny]
        ix_sigma = int(np.argmin(np.abs(x_sigma - x_target)))
        iy_sigma = int(np.argmin(np.abs(y_sigma - y_target)))
        return iy_ux, ix_ux, iy_uy, ix_uy, iy_sigma, ix_sigma

    def _record_comparison_point(self, t):
        """Record displacement and stress at the selected interior location."""
        iy_ux, ix_ux, iy_uy, ix_uy, iy_sigma, ix_sigma = (
            self._comparison_point_indices()
        )
        self.comparison_times.append(t)
        self.comparison_ux.append(self.ux[iy_ux, ix_ux])
        self.comparison_uy.append(self.uy[iy_uy, ix_uy])
        self.comparison_sigma_xx.append(self.sigmaqs[iy_sigma, ix_sigma])

        dx = self.grid.x[ix_sigma + 1] - self.grid.x[ix_sigma]
        dy = self.grid.y[iy_sigma + 1] - self.grid.y[iy_sigma]
        strain_xx = (
            self.ux[iy_sigma + 1, ix_sigma + 1]
            - self.ux[iy_sigma + 1, ix_sigma]
        ) / dx
        strain_yy = (
            self.uy[iy_sigma + 1, ix_sigma + 1]
            - self.uy[iy_sigma, ix_sigma + 1]
        ) / dy
        sigma_yy = self.p.lam * strain_xx + (
            self.p.lam + 2.0 * self.p.G
        ) * strain_yy
        self.comparison_sigma_yy.append(sigma_yy)

        x_sample = self.grid.xp[1:self.p.Nx][ix_sigma]
        y_sample = self.grid.yp[1:self.p.Ny][iy_sigma]
        ix_tau = int(np.argmin(np.abs(self.grid.x - x_sample)))
        iy_tau = int(np.argmin(np.abs(self.grid.y - y_sample)))
        self.comparison_sigma_xy.append(self.tauqs[iy_tau, ix_tau])

    def _comparison_plot_indices(self, max_points=20):
        """Select sparse history samples while retaining both endpoints."""
        count = len(self.comparison_times)
        return np.unique(np.linspace(0, count - 1, min(count, max_points), dtype=int))

    def plot_displacement_comparison(self):
        """Compare axial and Poisson displacement histories."""
        _, ix_ux, iy_uy, _, _, _ = self._comparison_point_indices()
        times = np.asarray(self.comparison_times)
        plot_indices = self._comparison_plot_indices()
        length = self.grid.x[-1] - self.grid.x[0]
        strain_xx = (
            self.p.bc.right.ux.value - self.p.bc.left.ux.value
        ) * times / length

        x_sample = self.grid.x[ix_ux]
        u_left = self.p.bc.left.ux.value * times
        ux_analytical = u_left + strain_xx * (x_sample - self.grid.x[0])

        y_sample = self.grid.y[iy_uy]
        strain_yy = -self.p.nu * strain_xx / (1.0 - self.p.nu)
        uy_analytical = strain_yy * (y_sample - self.grid.y[0])

        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
        axes[0].plot(times, ux_analytical, "k-", 
                     label="Analytical")
        axes[0].plot(times[plot_indices],
                     np.asarray(self.comparison_ux)[plot_indices], "o",
                     color="tab:blue", label="FastSlipPy")
        axes[0].set_ylabel(r"$u_x$ [m]")
        axes[0].set_title(f"Axial displacement at x = {x_sample:g} m")

        axes[1].plot(times, uy_analytical, "k-",
                     label="Analytical")
        axes[1].plot(times[plot_indices],
                     np.asarray(self.comparison_uy)[plot_indices], "o",
                     color="tab:orange", label="FastSlipPy")
        axes[1].set_xlabel("Time [s]")
        axes[1].set_ylabel(r"$u_y$ [m]")
        axes[1].set_title(f"Poisson displacement at y = {y_sample:g} m")

        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.legend()
        fig.tight_layout()
        path = self.output.out / "displacement_comparison.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_stress_comparison(self):
        """Compare axial, transverse, and shear-stress histories."""
        _, _, _, _, _, ix_sigma = self._comparison_point_indices()
        times = np.asarray(self.comparison_times)
        plot_indices = self._comparison_plot_indices()
        length = self.grid.x[-1] - self.grid.x[0]
        strain_xx = (
            self.p.bc.right.ux.value - self.p.bc.left.ux.value
        ) * times / length
        sigma_xx_analytical = (
            4.0 * self.p.G * (self.p.lam + self.p.G)
            / (self.p.lam + 2.0 * self.p.G)
            * strain_xx
        )
        zero_stress = np.zeros_like(times)
        x_sample = self.grid.xp[1:self.p.Nx][ix_sigma]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
        analytical = (sigma_xx_analytical, zero_stress, zero_stress)
        numerical = (
            self.comparison_sigma_xx,
            self.comparison_sigma_yy,
            self.comparison_sigma_xy,
        )
        labels = (r"$\sigma_{xx}$", r"$\sigma_{yy}$", r"$\sigma_{xy}$")
        colors = ("tab:blue", "tab:orange", "tab:green")
        for ax, exact, computed, label, color in zip(
            axes, analytical, numerical, labels, colors
        ):
            if label == r"$\sigma_{yy}$":
                ax.plot(times, exact, "k-",
                                    label="Analytical")
                ax.plot(times[plot_indices],
                                np.asarray(computed)[plot_indices], "o",
                                color=color, label="FastSlipPy")
                ax.set_ylabel(f"{label} [Pa]")
                #ax.set_ylim(-0.1, 0.1)
            else:
                ax.plot(times, exact / 1e6, "k-", label="Analytical")
                ax.plot(times[plot_indices],
                        np.asarray(computed)[plot_indices] / 1e6, "o",
                        color=color, label="FastSlipPy")
                ax.set_ylabel(f"{label} [MPa]")
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_title(f"Stress histories near x = {x_sample:g} m")
            ax.set_xlabel("Time [s]")
        fig.tight_layout()
        path = self.output.out / "stress_comparison.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def run(self):
        t0_all = time.perf_counter()
        p = self.p
        Nx, Ny = p.Nx, p.Ny
        vtk_interval = p.vtk_interval
        if vtk_interval is None:
            raise RuntimeError("vtk_interval was not initialized.")

        # ── initialise / load checkpoint ──
        if not self.checkpointer:
            dPdt = p.loading.dPdt_pre
            dt = p.dt_init
            t = 0.0
            self._build_and_factor_LH(dPdt)
        else:
            ckpt = self.output.load_checkpoint(self.checkpointer)
            self.fault.U     = ckpt["U"]
            self.fault.V     = ckpt["V"]
            self.fault.tau   = ckpt["tau"]
            self.fault.sigma = ckpt["sigma"]
            self.fault.theta = ckpt["theta"]
            self.tauqs   = ckpt["tauqs"]
            self.sigmaqs = ckpt["sigmaqs"]
            self.uy = ckpt["uy"];  self.vy = ckpt["vy"]
            self.ux = ckpt["ux"];  self.vx = ckpt["vx"]
            dt = float(ckpt["dt"])
            t = float(ckpt["t"])
            if p.case_type == "groningen":
                if "Pl" in ckpt and "Pr" in ckpt:
                    self.stress.Pl = ckpt["Pl"]
                    self.stress.Pr = ckpt["Pr"]
                    self.stress.P = np.where(
                        self.grid.y < 1000, self.stress.Pl, self.stress.Pr
                    )
                else:
                    self.stress.update_pressure(min(t, p.loading.tload), p.loading.dPdt_pre)
                    self.stress.update_pressure(max(0.0, t - p.loading.tload), p.loading.dPdt_post)
            elif "P" in ckpt:
                self.stress.P = ckpt["P"]
            if t >= p.tfinal or p.Nt <= 0:
                self.output.close()
                print("No additional time steps requested; existing output retained.")
                return
            self.output.restore_history(t, self.checkpointer, self.stress.tau0)
            if p.case_type == "groningen" and t == p.loading.tload:
                dt = p.dt_init
            dPdt = (
                p.loading.dPdt_post
                if p.case_type == "groningen" and t >= p.loading.tload
                else p.loading.dPdt_pre
            )
            self._build_and_factor_LH(dPdt)

        self.comparison_times = []
        self.comparison_ux = []
        self.comparison_uy = []
        self.comparison_sigma_xx = []
        self.comparison_sigma_yy = []
        self.comparison_sigma_xy = []
        self._record_comparison_point(t)

        dt_max = p.dt_max
        t2 = max(0.0, t - p.loading.tload) if p.case_type == "groningen" else t
        phase = (
            0
            if p.case_type == "groningen" and t < p.loading.tload
            else 2
        )

        print(f"Setup complete in {time.perf_counter()-t0_all:.1f}s.  "
                f"Starting {p.Nt} time steps …")

        # ── time loop ────────────────────────────────────────────────
        for it in range(1, p.Nt + 1):
            global_it = self.checkpointer + it

            # Phase transition: pre → post depletion
            if phase == 1:
                dPdt  = p.loading.dPdt_post
                self._build_and_factor_LH(dPdt)
                dt    = p.dt_init
                dt_max = p.dt_max
                t2    = 0.0
                phase = 2

            # dt is constant
            dt = p.dt_init

            if p.time_integrator is TimeIntegrator.EULER:
                self._advance_euler_coupling(dt, dPdt)
            else:
                self._advance_rk2_midpoint_coupling(dt, dPdt)

            t += dt
            self._record_comparison_point(t)
            if phase == 2:
                t2 += dt

            reached_final_time = t >= p.tfinal
            needs_velocity_fields = (
                global_it % p.output_interval == 0
                or global_it % p.checkpoint_interval == 0
                or (
                    p.output_vtk_option
                    and global_it % vtk_interval == 0
                )
                or reached_final_time
                or it == p.Nt
            )
            output_V, output_tau, output_vx, output_vy = (
                self._synchronized_output_state(
                    dPdt, include_velocity_fields=needs_velocity_fields
                )
            )

            # ── logging ──
            self.output.log(it, t2 if phase == 2 else t, dt,
                            output_V, self.fault.U, self.checkpointer)

            if global_it % p.output_interval == 0:
                self.output.write_memory(
                    it, self.fault.U, output_V, output_tau,
                    self.fault.sigma, self.stress.P, self.fault.theta,
                    dt, t, self.tauqs, self.sigmaqs,
                    self.uy, output_vy, self.ux, output_vx, self.stress.tau0)
                if p.case_type == "california":
                    self.output.record_bp3_surface(
                        it, t, self.grid, self.ux, self.uy,
                        output_vx, output_vy
                    )

            if global_it % p.checkpoint_interval == 0:
                self.output.save_checkpoint(
                    it, self.checkpointer, self.fault,
                    self.tauqs, self.sigmaqs,
                    self.uy, output_vy, self.ux, output_vx, dt, t,
                    fault_velocity=output_V,
                    fault_traction=output_tau,
                    pressure=self.stress.P,
                    pressure_left=self.stress.Pl, pressure_right=self.stress.Pr)
                self.output.save_all()
                print(f"  Checkpoint it={global_it}, elapsed {time.perf_counter()-t0_all:.1f}s")
            should_write_vtk = p.output_vtk_option and (
                global_it % vtk_interval == 0
                or reached_final_time
                or it == p.Nt
            )
            if should_write_vtk:
                stage_velocity = self.fault.V
                stage_traction = self.fault.tau
                try:
                    self.fault.V = output_V
                    self.fault.tau = output_tau
                    self.output.write_vtk(
                        global_it, self.grid,
                        self.ux, self.uy, output_vx, output_vy,
                        self.tauqs, self.sigmaqs,
                        self.fault, t,
                    )
                finally:
                    self.fault.V = stage_velocity
                    self.fault.tau = stage_traction

            if reached_final_time:
                break

        # ── wrap up ──
        self.output.save_checkpoint(
                    it, self.checkpointer, self.fault,
                    self.tauqs, self.sigmaqs,
                    self.uy, output_vy, self.ux, output_vx, dt, t,
                    fault_velocity=output_V,
                    fault_traction=output_tau,
                    pressure=self.stress.P,
                    pressure_left=self.stress.Pl, pressure_right=self.stress.Pr)
        self.output.save_all()
        if p.case_type == "california":
            self.output.write_bp3_outputs(self.grid)
        self.output.close()
        self.plot_displacement_comparison()
        self.plot_stress_comparison()
        if p.case_type == "groningen":
            self.figure_creator.plot_results(Nx, shift_y=2000)
        elif p.case_type == "lab":
            self.figure_creator.plot_results_shear(Nx)
        elif p.case_type == "california":
            self.figure_creator.plot_results(Nx, shift_y=0)
        print(f"Done.  Total running time: {time.perf_counter()-t0_all:.1f}s")
    
    def after_run(self):
        pass

if __name__ == "__main__":
    # Customise parameters here or leave all defaults

    params = ModelParameters(
        case_type = "lab",
        fault_mode = FaultMode.NONE,
        alpha = 90.0,
        xsize = 0.1,
        ysize = 0.05,
        Nx=41, Ny=21,
        Nt=200,
        output_interval=10,
        #tfinal = 10,
        checkpoint_interval=10,
        #vtk_interval = 500,
        dt_init=0.01,
        dt_max = 0.01,
        nu=0.25,
        E=1e10, #according to k_critical = sigam * (b-a) / d_c, E = 1e10  ## E=0.55e10 for stick-slip pattern
        flash_heating_option = False
    )

    params.bc.left.ux.set_velocity(1e-4)
    params.bc.left.uy.set_free()
    params.bc.right.ux.set_velocity(-1e-4)
    params.bc.right.uy.set_free()
    params.bc.top.ux.set_traction_free()
    params.bc.top.uy.set_fixed()
    params.bc.bottom.ux.set_traction_free()
    params.bc.bottom.uy.set_traction_free()

    params.layers.set_homogeneous(top = 1, bottom = 2, a=params.a0, b=params.b0)

    model = RunFastSlipPy(params=params, output_dir="output/validation/elastic/2d_uniaxial_compression")
    model.run()
