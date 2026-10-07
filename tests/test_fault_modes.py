import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from fastslippy.fast_slip_py import FastSlipPy
from fastslippy.pre_processing.grid import Grid
from fastslippy.pre_processing.model_parameters import FaultMode, ModelParameters
from fastslippy.solver.matrix_builder import MatrixBuilder


def test_fault_mode_parsing_and_even_grid_without_fault():
    params = ModelParameters(Nx=10, fault_mode="none")

    assert params.fault_mode is FaultMode.NONE
    assert ModelParameters().fault_mode is FaultMode.FRICTIONAL

    with pytest.raises(AssertionError, match="fault at centre column"):
        ModelParameters(Nx=10, fault_mode=FaultMode.LOCKED)

    with pytest.raises(ValueError, match="fault_mode must be one of"):
        ModelParameters(fault_mode="invalid")


@pytest.mark.parametrize("mode", [FaultMode.NONE, FaultMode.LOCKED])
def test_nonfrictional_modes_force_zero_fault_velocity(tmp_path, mode):
    model = FastSlipPy(
        ModelParameters(Nx=11, Ny=9, fault_mode=mode),
        output_dir=tmp_path / mode.value,
    )
    try:
        model.fault.V.fill(1.0)
        model._solve_fault_slip_rate()
        np.testing.assert_array_equal(model.fault.V, 0.0)
    finally:
        model.output.close()


def test_no_fault_matrix_recovers_continuous_uniaxial_velocity():
    params = ModelParameters(
        fault_mode=FaultMode.NONE,
        alpha=90.0,
        xsize=0.1,
        ysize=0.05,
        Nx=10,
        Ny=9,
        E=1.0e10,
    )
    params.bc.left.ux.set_velocity(1.0e-4)
    params.bc.right.ux.set_velocity(-1.0e-4)
    params.bc.top.ux.set_free()
    params.bc.bottom.ux.set_free()
    params.bc.left.uy.set_fixed()
    params.bc.right.uy.set_fixed()
    params.bc.top.uy.set_fixed()
    params.bc.bottom.uy.set_fixed()

    grid = Grid(params)
    builder = MatrixBuilder(params, grid)
    solution = spsolve(builder.build_LH().tocsc(), builder.build_RH(0.0, np.ones(params.Ny)))
    vx = np.reshape(
        solution[0::2], (params.Nx + 1, params.Ny + 1), order="C"
    ).T[:, :params.Nx]
    vy = np.reshape(
        solution[1::2], (params.Nx + 1, params.Ny + 1), order="C"
    ).T[:params.Ny, :]

    expected_vx = np.linspace(1.0e-4, -1.0e-4, params.Nx)
    np.testing.assert_allclose(
        vx, np.broadcast_to(expected_vx, vx.shape), rtol=1e-11, atol=1e-15
    )
    np.testing.assert_allclose(vy, 0.0, atol=1e-15)
