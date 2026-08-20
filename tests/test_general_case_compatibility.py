import numpy as np
import pytest

from fastslippy import FastSlipPy
from fastslippy.pre_processing.frictional_zones import FrictionalZones
from fastslippy.pre_processing.grid import Grid
from fastslippy.pre_processing.model_parameters import CaseType, ModelParameters
from fastslippy.solver.fault_state import FaultState
from fastslippy.solver.stress_state import StressState


def _lab_parameters(*, stretched: bool = False, **overrides):
    values = dict(
        case_type=CaseType.LAB,
        alpha=90.0,
        xsize=1.0,
        ysize=1.0,
        Nx=11,
        Ny=11,
        Nt=3,
        dt_init=1e-4,
        dt_max=1e-4,
        tfinal=3e-4,
        output_interval=1,
        checkpoint_interval=100,
        output_vtk_option=False,
        Vi=1e-40,
        E=0.55e10,
        mu0=0.72,
        V0=1e-6,
        a0=0.012,
        b0=0.0135,
    )
    if stretched:
        values.update(
            x_stretch_enabled=True,
            y_stretch_enabled=True,
            x_stretch_inner_size=0.2,
            y_stretch_inner_size=0.2,
            x_stretch_inner_points=5,
            y_stretch_inner_points=3,
            x_stretch_power=2,
            y_stretch_power=2,
            allow_nonuniform_solver=True,
        )
    values.update(overrides)
    params = ModelParameters(**values)
    params.bc.left.set_fixed()
    params.bc.right.ux.set_fixed()
    params.bc.right.uy.set_velocity(1e-5)
    params.bc.top.ux.set_fixed()
    params.bc.top.uy.set_velocity(1e-5)
    params.bc.bottom.ux.set_fixed()
    params.bc.bottom.uy.set_velocity(1e-5)
    return params


@pytest.mark.parametrize("case_value", [CaseType.LAB, "lab", "LAB"])
def test_case_type_and_default_lab_friction_are_general(case_value):
    params = ModelParameters(case_type=case_value, Nx=11, Ny=11)
    grid = Grid(params)
    friction = FrictionalZones(params, grid.y)
    stress = StressState(params, grid.y)
    fault = FaultState(params, stress, friction, fault_y=grid.y)

    assert params.case_type is CaseType.LAB
    np.testing.assert_allclose(friction.a, params.a0)
    np.testing.assert_allclose(friction.b, params.b0)
    np.testing.assert_allclose(fault.theta, params.L / params.V0)


def test_newton_v2_solves_signed_friction_roots_for_lab_case():
    params = _lab_parameters(Ny=9)
    grid = Grid(params)
    friction = FrictionalZones(params, grid.y)
    stress = StressState(params, grid.y)
    fault = FaultState(params, stress, friction, fault_y=grid.y)

    target_velocity = np.logspace(-12, -7, params.Ny)
    target_velocity[1::2] *= -1.0
    exponent = (
        params.mu0
        + friction.b * np.log(params.V0 * fault.theta / params.L)
    ) / friction.a
    driving_stress = (
        fault.sigma * friction.a
        * np.arcsinh(
            target_velocity / (2.0 * params.V0) * np.exp(exponent)
        )
        + params.eta * target_velocity
    )
    tauqs = driving_stress - stress.tau0

    fault.solve_slip_rate_newton_v2(tauqs, stress, friction)

    residual = (
        fault.sigma * friction.a
        * np.arcsinh(fault.V / (2.0 * params.V0) * np.exp(exponent))
        + params.eta * fault.V
        - driving_stress
    )
    assert np.max(np.abs(residual)) <= params.friction_tolerance
    np.testing.assert_allclose(fault.V, target_velocity, rtol=5e-5, atol=0.0)


@pytest.mark.parametrize("stretched", [False, True])
def test_short_lab_run_uses_newton_v2_on_uniform_and_stretched_mesh(
    tmp_path, stretched
):
    params = _lab_parameters(stretched=stretched)
    model = FastSlipPy(
        params=params,
        output_dir=str(tmp_path / ("stretched" if stretched else "uniform")),
    )
    assert model.grid.is_nonuniform is stretched

    newton_v2 = model.fault.solve_slip_rate_newton_v2
    calls = []

    def tracked_newton_v2(*args, **kwargs):
        calls.append(True)
        return newton_v2(*args, **kwargs)

    model.fault.solve_slip_rate_newton_v2 = tracked_newton_v2
    model.fault.solve_slip_rate_newton = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("general cases should use Newton v2")
    )
    model.figure_creator.plot_results_shear = lambda *args, **kwargs: None
    model.run()

    np.testing.assert_allclose(model.output.tm, [1e-4, 2e-4, 3e-4])
    assert len(calls) == 3
    assert np.all(np.isfinite(model.fault.V))
    assert np.all(np.isfinite(model.fault.theta))
    assert np.all(np.isfinite(model.fault.sigma))


def test_short_groningen_run_remains_compatible_with_newton_v2(tmp_path):
    params = ModelParameters(
        case_type=CaseType.GRONINGEN,
        alpha=70.0,
        xsize=2000.0,
        ysize=2000.0,
        Nx=11,
        Ny=11,
        Nt=1,
        dt_init=1.0,
        dt_max=1.0,
        tfinal=1.0,
        output_interval=1,
        checkpoint_interval=100,
        output_vtk_option=False,
        rho=2400.0,
        rhof=1150.0,
        rhog=200.0,
        cs=1650.0,
        mu0=0.3,
        nu=0.15,
        V0=1e-6,
        L=0.5,
        Vw=1e90,
        Vi=1e-30,
        flash_heating_option=True,
    )
    params.loading.tload = 1000.0 * 365.0 * 24.0 * 3600.0
    params.loading.dPdt_pre = 0.0
    params.loading.dPdt_post = -0.0127
    params.bc.left.ux.set_fixed()
    params.bc.left.uy.set_free()
    params.bc.right.ux.set_fixed()
    params.bc.right.uy.set_free()
    params.bc.top.ux.set_free()
    params.bc.top.uy.set_fixed()
    params.bc.bottom.ux.set_free()
    params.bc.bottom.uy.set_fixed()
    params.layers.set_groningen()

    model = FastSlipPy(params=params, output_dir=str(tmp_path / "groningen"))
    model.figure_creator.plot_results = lambda *args, **kwargs: None
    model.run()

    np.testing.assert_allclose(model.output.tm, [1.0])
    assert np.all(np.isfinite(model.fault.V))
    assert np.all(np.isfinite(model.fault.theta))
    assert np.all(np.isfinite(model.fault.sigma))
