from types import MethodType

import numpy as np
import pytest

from fastslippy import FastSlipPy
from fastslippy.pre_processing.model_parameters import ModelParameters, TimeIntegrator


def _small_bp3_parameters(**overrides):
    values = dict(
        case_type="california",
        alpha=60.0,
        motion_sign=-1,
        auto_motion_sign=True,
        xsize=1000.0,
        ysize=900.0,
        Nx=11,
        Ny=10,
        Nt=4,
        output_interval=1,
        checkpoint_interval=2,
        output_vtk_option=False,
        rho=2670.0,
        cs=3464.0,
        mu0=0.6,
        nu=0.25,
        V0=1e-6,
        a0=0.01,
        a_max=0.025,
        b0=0.015,
        L=0.008,
        Vi=1e-9,
        H=300.0,
        h=100.0,
        W_f=700.0,
        dt_init=1.0,
        dt_max=1.0,
        dt_growth=1.0,
        tfinal=4.0,
        friction_tolerance=5.0,
    )
    values.update(overrides)
    p = ModelParameters(**values)
    p.loading.dPdt_pre = 0.0
    p.loading.dPdt_post = 0.0
    p.loading.V_p = 1e-9
    p.loading.V_L = 1e-9
    p.bc.left.ux.set_fixed()
    p.bc.left.uy.set_velocity(-0.5e-9)
    p.bc.right.ux.set_fixed()
    p.bc.right.uy.set_velocity(0.5e-9)
    p.bc.top.set_traction_free()
    p.bc.bottom.set_traction_free()
    p.layers.set_homogeneous(
        top=p.ysize, bottom=2.0 * p.ysize, a=p.a0, b=p.b0
    )
    return p


def _disable_figures(model):
    model.figure_creator.plot_results = lambda *args, **kwargs: None


@pytest.mark.parametrize("integrator", ["euler", "rk2_midpoint"])
def test_checkpoint_restart_matches_uninterrupted_run(tmp_path, integrator):
    full_params = _small_bp3_parameters(time_integrator=integrator)
    full = FastSlipPy(full_params, output_dir=str(tmp_path / "full"))
    _disable_figures(full)
    full.run()

    first_params = _small_bp3_parameters(
        time_integrator=integrator, Nt=2, tfinal=2.0
    )
    restart_dir = tmp_path / "restart"
    first = FastSlipPy(first_params, output_dir=str(restart_dir))
    _disable_figures(first)
    first.run()

    resumed_params = _small_bp3_parameters(
        time_integrator=integrator, Nt=2, tfinal=4.0
    )
    resumed = FastSlipPy(
        resumed_params, output_dir=str(restart_dir), checkpointer=2
    )
    _disable_figures(resumed)
    resumed.run()

    with np.load(tmp_path / "full" / "data_4.npz") as uninterrupted, np.load(
        restart_dir / "data_4.npz"
    ) as restarted:
        for name in (
            "U", "V", "tau", "sigma", "theta", "tauqs", "sigmaqs",
            "uy", "vy", "ux", "vx",
        ):
            np.testing.assert_allclose(
                restarted[name], uninterrupted[name], rtol=2e-12, atol=2e-12
            )
        assert restarted["state_time_level"].item() == "end"
        assert restarted["time_integrator"].item() == integrator


@pytest.mark.parametrize("integrator", ["euler", "rk2_midpoint"])
def test_saved_fault_fields_are_at_the_same_time_level(tmp_path, integrator):
    p = _small_bp3_parameters(
        time_integrator=integrator, Nt=1, tfinal=1.0
    )
    model = FastSlipPy(p, output_dir=str(tmp_path / integrator))
    _disable_figures(model)
    model.run()

    with np.load(tmp_path / integrator / "data_1.npz") as checkpoint:
        np.testing.assert_allclose(checkpoint["V"], model.output.Vm[:, 0])
        np.testing.assert_allclose(checkpoint["tau"], model.output.taum[:, 0])
        expected_tau = (
            checkpoint["tauqs"][:, p.Nx // 2]
            + model.stress.tau0
            - p.eta * checkpoint["V"]
        )
        np.testing.assert_allclose(checkpoint["tau"], expected_tau)
        exponent = (
            p.mu0
            + model.fric.b * np.log(
                p.V0 * checkpoint["theta"] / p.L
            )
        ) / model.fric.a
        friction = (
            checkpoint["sigma"] * model.fric.a
            * np.arcsinh(
                checkpoint["V"] / (2.0 * p.V0) * np.exp(exponent)
            )
            + p.eta * checkpoint["V"]
        )
        creep_start = model.fault.california_loading_start_idx()
        residual = friction[1:creep_start] - (
            checkpoint["tauqs"][1:creep_start, p.Nx // 2]
            + model.stress.tau0[1:creep_start]
        )
        assert np.max(np.abs(residual)) <= p.friction_tolerance

        expected_vx, expected_vy = model._solve_elastic_velocity(
            0.0, checkpoint["V"]
        )
        np.testing.assert_allclose(checkpoint["vx"], expected_vx)
        np.testing.assert_allclose(checkpoint["vy"], expected_vy)


def _mock_scalar_coupling(tmp_path, integrator):
    """Create a vector-shaped model whose coupling reduces to u' = 1 + u."""
    p = _small_bp3_parameters(time_integrator=integrator)
    model = FastSlipPy(p, output_dir=str(tmp_path / integrator))
    model.output.close()

    def solve_fault(this, tauqs_col=None):
        if tauqs_col is None:
            tauqs_col = this.tauqs[:, this.p.Nx // 2]
        this.fault.V.fill(1.0 + float(tauqs_col[0]))

    def solve_elastic(this, _dPdt, velocity):
        vx = np.full_like(this.vx, float(velocity[0]))
        vy = np.zeros_like(this.vy)
        return vx, vy

    def recover_stress(this, uy, ux):
        tau = np.full_like(this.tauqs, float(ux[0, 0]))
        sigmaqs = np.zeros_like(this.sigmaqs)
        sigma = np.full_like(this.fault.sigma, this.p.sigma0)
        return tau, sigmaqs, sigma

    model._solve_fault_slip_rate = MethodType(solve_fault, model)
    model._solve_elastic_velocity = MethodType(solve_elastic, model)
    model._stress_from_displacement = MethodType(recover_stress, model)
    model.tauqs.fill(0.0)
    model.ux.fill(0.0)
    return model


def _integrate_mock_scalar(tmp_path, integrator, steps):
    model = _mock_scalar_coupling(tmp_path, integrator)
    dt = 1.0 / steps
    for _ in range(steps):
        model._solve_fault_slip_rate()
        if model.p.time_integrator is TimeIntegrator.EULER:
            model._advance_euler_coupling(dt, 0.0)
        else:
            model._advance_rk2_midpoint_coupling(dt, 0.0)
    return float(model.ux[0, 0])


def test_midpoint_coupling_has_second_order_on_coupled_scalar_problem(tmp_path):
    exact = np.e - 1.0
    euler_errors = []
    midpoint_errors = []
    for steps in (10, 20, 40):
        euler_errors.append(
            abs(_integrate_mock_scalar(tmp_path, "euler", steps) - exact)
        )
        midpoint_errors.append(
            abs(
                _integrate_mock_scalar(
                    tmp_path, "rk2_midpoint", steps
                ) - exact
            )
        )

    assert euler_errors[0] / euler_errors[1] == pytest.approx(2.0, rel=0.12)
    assert midpoint_errors[0] / midpoint_errors[1] == pytest.approx(
        4.0, rel=0.12
    )
    assert midpoint_errors[1] / midpoint_errors[2] == pytest.approx(
        4.0, rel=0.12
    )
    assert midpoint_errors[-1] < 0.02 * euler_errors[-1]
