"""Tests for the static BP3 endpoint/interface-transfer diagnostic."""

import numpy as np

from examples.run_bp3_small_peak_diagnostic import build_small_bp3_parameters
from fastslippy.pre_processing.grid import Grid
from fastslippy.utilities.bp3_interface_transfer import (
    diagnose_bp3_interface_transfer,
    direct_fault_normal_stress,
)
from fastslippy.utilities.bp3_operator_consistency import pack_staggered_fields
from fastslippy.utilities.stress_cal_util import StressCalUtil


def test_direct_fault_normal_recovery_is_exact_for_affine_field():
    params = build_small_bp3_parameters()
    grid = Grid(params)
    a, b, c, d = 0.12, -0.07, 0.05, 0.09
    ux = a * grid.x[None, :] + b * grid.yp[:, None]
    uy = c * grid.xp[None, :] + d * grid.y[:, None]
    solution = pack_staggered_fields(params, ux, uy)

    result = direct_fault_normal_stress(params, grid, solution)
    expected = (
        (params.lam + 2.0 * params.G) * a
        + params.lam * d
        - 2.0 * params.G * grid.cosa * b
    )
    np.testing.assert_allclose(result.left, expected, rtol=2e-12, atol=1e-3)
    np.testing.assert_allclose(result.right, expected, rtol=2e-12, atol=1e-3)

    ux_out = solution[0::2].reshape(
        (params.Nx + 1, params.Ny + 1)
    ).T[:, : params.Nx]
    uy_out = solution[1::2].reshape(
        (params.Nx + 1, params.Ny + 1)
    ).T[: params.Ny, :]
    _, sigma = StressCalUtil(prefer_numba=False).compute_stress_fields(
        uy_out,
        ux_out,
        grid.dx,
        grid.dy,
        params.lam,
        params.G,
        grid.cosa,
        grid.sina,
        params.Ny,
        params.Nx,
        x=grid.x,
        y=grid.y,
        xp=grid.xp,
        yp=grid.yp,
    )
    left, right = StressCalUtil(prefer_numba=False).recover_fault_normal_stress(
        sigma,
        grid.x,
        grid.y,
        grid.xp,
        grid.yp,
        params.Nx // 2 - 1,
        params.Nx // 2,
    )
    np.testing.assert_allclose(left, expected, rtol=2e-12, atol=1e-3)
    np.testing.assert_allclose(right, expected, rtol=2e-12, atol=1e-3)


def test_interface_transfer_separates_wf_endpoint_conventions():
    params = build_small_bp3_parameters()
    # The shared small-peak fixture places W_f exactly at the bottom because it
    # does not otherwise need a creeping segment.  Move W_f into the domain for
    # this endpoint-specific diagnostic.
    params.W_f = 12e3
    params.loading.V_p = 1e-9
    params.loading.V_L = 1e-9
    params.bc.left.uy.set_velocity(-0.5e-9)
    params.bc.right.uy.set_velocity(0.5e-9)

    result = diagnose_bp3_interface_transfer(params)
    transfer = result.transfer
    wf = result.wf_loading

    assert transfer.tau.shape == (params.Ny, 5)
    assert transfer.projected_tau.shape == (4, 5)
    assert np.all(np.isfinite(transfer.projected_coulomb_current))
    assert np.all(np.isfinite(transfer.projected_coulomb_direct))

    grid = Grid(params)
    wf_index = int(np.searchsorted(grid.y, params.W_f, side="left"))
    differing = np.flatnonzero(wf.production_rate != wf.rs_endpoint_rate)
    np.testing.assert_array_equal(differing, np.array([wf_index]))
    assert np.linalg.norm(wf.delta_tau) > 0.0
    assert np.linalg.norm(wf.delta_sigma_effective_current) > 0.0


def test_custom_source_and_receiver_modes_are_supported():
    params = build_small_bp3_parameters()
    params.W_f = 12e3
    params.loading.V_L = 1e-9
    y = Grid(params).y
    source = np.sin(np.pi * y / y[-1])
    receiver = np.exp(-((y - 10e3) / 2e3) ** 2)

    result = diagnose_bp3_interface_transfer(
        params,
        source_labels=("source",),
        source_modes=source,
        receiver_labels=("receiver",),
        receiver_modes=receiver,
    )
    assert result.transfer.projected_tau.shape == (1, 1)
    assert np.isfinite(result.transfer.projected_tau[0, 0])
