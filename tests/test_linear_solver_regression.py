#/////////////////////////////////////////////////
__author__      = "Chengshun Shang (Utrecht University)"
__copyright__   = "Copyright (C) 2026-present by Chengshun Shang"
__version__     = "0.0.1"
__maintainer__  = "Chengshun Shang"
__email__       = "c.shang@uu.nl"
__status__      = "development"
__date__        = "Aug 9, 2026"
__license__     = "MIT License"
#/////////////////////////////////////////////////

import numpy as np
import pytest

import fastslippy.fast_slip_py as fast_slip_py_module
from fastslippy.fast_slip_py import FastSlipPy, MPI_PETSC_AVAILABLE
from fastslippy.pre_processing.model_parameters import ModelParameters
from fastslippy.pre_processing.grid import Grid
from fastslippy.solver.matrix_builder import MatrixBuilder


@pytest.mark.filterwarnings("ignore:divide by zero encountered in divide:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
def test_iterative_solver_matches_direct_solution():
    common_kwargs = dict(
        case_type="lab",
        Nx=11,
        Ny=11,
        xsize=1.0,
        ysize=1.0,
        iterative_method="gmres",
        iterative_rtol=1e-10,
        iterative_atol=0.0,
        iterative_maxiter=300,
        ilu_drop_tol=1e-5,
        ilu_fill_factor=20.0,
    )

    params_direct = ModelParameters(linear_solver="direct", **common_kwargs)
    model_direct = FastSlipPy(params=params_direct, output_dir="output")
    model_direct._build_and_factor_LH(params_direct.loading.dPdt_pre)
    rhs = model_direct.RH_builder.build_RH(params_direct.loading.dPdt_pre, model_direct.fault.V)
    direct_solution = model_direct._solve(rhs)

    params_iter = ModelParameters(linear_solver="iterative", **common_kwargs)
    model_iter = FastSlipPy(params=params_iter, output_dir="output")
    model_iter._build_and_factor_LH(params_iter.loading.dPdt_pre)
    iterative_solution = model_iter._solve(rhs)

    np.testing.assert_allclose(iterative_solution, direct_solution, rtol=1e-7, atol=1e-9)


@pytest.mark.filterwarnings("ignore:divide by zero encountered in divide:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
def test_direct_solver_can_fallback_to_iterative_when_lu_oom(monkeypatch: pytest.MonkeyPatch):
    params = ModelParameters(
        case_type="lab",
        Nx=11,
        Ny=11,
        xsize=1.0,
        ysize=1.0,
        linear_solver="direct",
        iterative_method="gmres",
        fallback_to_iterative_on_oom=True,
    )
    model = FastSlipPy(params=params, output_dir="output")

    def raise_memory_error(_):
        raise MemoryError("simulated-lu-oom")

    monkeypatch.setattr(fast_slip_py_module, "factorized", raise_memory_error)

    model._build_and_factor_LH(params.loading.dPdt_pre)
    rhs = model.RH_builder.build_RH(params.loading.dPdt_pre, model.fault.V)
    solution = model._solve(rhs)

    assert np.all(np.isfinite(solution))


def test_mpi_direct_backend_solves_or_skips():
    if not MPI_PETSC_AVAILABLE:
        pytest.skip("mpi4py/petsc4py not available in this environment")

    params = ModelParameters(
        case_type="lab",
        Nx=11,
        Ny=11,
        xsize=1.0,
        ysize=1.0,
        solver_backend="mpi_direct",
    )
    model = FastSlipPy(params=params, output_dir="output")
    model._build_and_factor_LH(params.loading.dPdt_pre)
    rhs = model.RH_builder.build_RH(params.loading.dPdt_pre, model.fault.V)
    solution = model._solve(rhs)
    assert np.all(np.isfinite(solution))


def test_row_range_assembly_matches_full_matrix_and_rhs():
    params = ModelParameters(case_type="lab", Nx=11, Ny=11, xsize=1.0, ysize=1.0)
    grid = Grid(params)
    builder = MatrixBuilder(params, grid)
    model = FastSlipPy(params=params, output_dir="output")
    full_lh = builder.build_LH()
    row_range = (10, 30)
    local_lh = builder.build_LH(row_range=row_range)
    np.testing.assert_allclose(local_lh.toarray(), full_lh[row_range[0]:row_range[1], :].toarray())

    full_rh = builder.build_RH(params.loading.dPdt_pre, model.fault.V)
    local_rh = builder.build_RH(params.loading.dPdt_pre, model.fault.V, row_range=row_range)
    np.testing.assert_allclose(local_rh, full_rh[row_range[0]:row_range[1]])
