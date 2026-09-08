"""Run the fast 60-degree BP3 near-critical small-peak diagnostic."""

import numpy as np

from fastslippy.pre_processing.model_parameters import ModelParameters
from fastslippy.utilities.bp3_small_peak import (
    build_fault_traction_response,
    critical_stiffness,
    gaussian_fault_mode,
    rate_state_friction_coefficient,
    simulate_modal_pulse,
)


def build_small_bp3_parameters() -> ModelParameters:
    params = ModelParameters(
        case_type="california",
        alpha=60.0,
        motion_sign=-1,
        auto_motion_sign=True,
        # Compact domain with a 100 m core.  It resolves the BP3 critical
        # nucleation length while remaining small enough for pytest.
        xsize=20e3,
        ysize=15e3,
        Nx=81,
        Ny=101,
        rho=2670.0,
        cs=3464.0,
        nu=0.25,
        sigma0=50e6,
        mu0=0.6,
        V0=1e-6,
        a0=0.01,
        a_max=0.025,
        b0=0.015,
        L=0.008,
        Vi=1e-9,
        H=8e3,
        h=2e3,
        W_f=15e3,
        x_stretch_enabled=True,
        y_stretch_enabled=True,
        x_stretch_inner_size=6e3,
        y_stretch_inner_size=8e3,
        x_stretch_inner_points=61,
        y_stretch_inner_points=81,
        x_stretch_power=2,
        y_stretch_power=2,
        allow_nonuniform_solver=True,
        output_vtk_option=False,
    )
    params.loading.V_p = 1e-9
    params.loading.V_L = 1e-9
    params.bc.left.ux.set_fixed()
    params.bc.left.uy.set_fixed()
    params.bc.right.ux.set_fixed()
    params.bc.right.uy.set_fixed()
    params.bc.top.set_traction_free()
    params.bc.bottom.set_traction_free()
    return params


def main() -> None:
    params = build_small_bp3_parameters()
    response = build_fault_traction_response(params)
    theta_ss = params.L / params.loading.V_p
    mu_ss = rate_state_friction_coefficient(
        params.loading.V_p,
        theta_ss,
        a=params.a0,
        b=params.b0,
        mu0=params.mu0,
        V0=params.V0,
        L=params.L,
    )
    kc = critical_stiffness(
        sigma0=params.sigma0, a=params.a0, b=params.b0, L=params.L
    )

    candidates = []
    for width_km in np.geomspace(0.05, 8.0, 72):
        mode = gaussian_fault_mode(
            response.y, centre=4.0e3, width=width_km * 1e3
        )
        modal = response.project_mode(mode, mu_ss)
        candidates.append((width_km, modal.effective_stiffness / kc, modal))

    stable = min(candidates, key=lambda item: abs(item[1] - 1.2))
    unstable = min(candidates, key=lambda item: abs(item[1] - 0.8))

    print(f"critical stiffness: {kc / 1e6:.4f} MPa/m")
    for label, (width, ratio, modal) in (("stable", stable), ("unstable", unstable)):
        pulse = simulate_modal_pulse(modal, params, initial_velocity_factor=10.0)
        outcome = "event" if pulse.became_event else "decayed small peak"
        print(
            f"{label:8s}: width={width:.3f} km, keff/kc={ratio:.4f}, "
            f"k_tau={-modal.tau_coefficient / 1e6:.3f} MPa/m, "
            f"mu*k_sigma={mu_ss * modal.sigma_coefficient / 1e6:+.3f} MPa/m, "
            f"peak={pulse.peak_velocity:.3e} m/s, "
            f"final={pulse.final_velocity:.3e} m/s -> {outcome}"
        )


if __name__ == "__main__":
    main()
