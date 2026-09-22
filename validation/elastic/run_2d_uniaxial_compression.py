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

import time
import numpy as np

from fastslippy import FastSlipPy
from fastslippy.pre_processing.model_parameters import ModelParameters, TimeIntegrator

class RunFastSlipPy(FastSlipPy):
    """
    This can be customized for specific runs.
    """
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
        alpha = 90.0,
        xsize = 2.0,
        ysize = 1.0,
        Nx=41, Ny=21,
        Nt=100,
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

    params.bc.left.ux.set_velocity(1e-2)
    params.bc.left.uy.set_free()
    params.bc.right.ux.set_velocity(-1e-2)
    params.bc.right.uy.set_free()
    params.bc.top.ux.set_traction_free()
    params.bc.top.uy.set_traction_free()
    params.bc.bottom.ux.set_traction_free()
    params.bc.bottom.uy.set_fixed()

    params.layers.set_homogeneous(top = 1, bottom = 2, a=params.a0, b=params.b0)

    model = RunFastSlipPy(params=params, output_dir="output/validation/elastic/2d_uniaxial_compression")
    model.run()
