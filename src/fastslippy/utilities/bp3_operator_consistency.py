"""Static consistency diagnostics for the BP3-QD elastic operator.

The earthquake-cycle update integrates the bulk velocity field, whereas a
quasi-static formulation may instead reconstruct displacement from accumulated
fault slip and far-field displacement.  This module compares those two views
at a checkpoint and tests whether the recovered fault shear traction satisfies
weighted Betti reciprocity on a selected modal subspace.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized

from fastslippy.pre_processing.grid import Grid
from fastslippy.pre_processing.model_parameters import CaseType, ModelParameters
from fastslippy.solver.matrix_builder import MatrixBuilder
from fastslippy.utilities.bp3_small_peak import (
    FaultModeResponse,
    FaultReciprocityResult,
    _fault_tractions,
    diagnose_fault_reciprocity,
)
from fastslippy.utilities.stress_cal_util import StressCalUtil


@dataclass(frozen=True)
class ResidualSummary:
    """Scale-aware statistics for a selected set of equilibrium rows."""

    relative_l2: float
    maximum_absolute: float
    maximum_rowwise_relative: float
    number_of_rows: int


@dataclass(frozen=True)
class CheckpointEquilibriumResult:
    """Stored-state residual and optional static projection at one checkpoint."""

    y: np.ndarray
    stored_solution: np.ndarray
    absolute_rhs: np.ndarray
    residual: np.ndarray
    residual_groups: dict[str, ResidualSummary]
    projected_solution: np.ndarray | None
    displacement_relative_l2: float | None
    fault_tau_stored: np.ndarray
    fault_sigma_effective_stored: np.ndarray
    fault_tau_projected: np.ndarray | None
    fault_sigma_effective_projected: np.ndarray | None


@dataclass(frozen=True)
class BP3OperatorConsistencyResult:
    """Combined checkpoint and reduced fault-operator diagnostics."""

    checkpoint: CheckpointEquilibriumResult
    mode_response: FaultModeResponse | None
    reciprocity: FaultReciprocityResult | None


def pack_staggered_fields(
    params: ModelParameters,
    ux: np.ndarray,
    uy: np.ndarray,
) -> np.ndarray:
    """Pack saved physical staggered fields into the full elastic vector.

    Checkpoints omit the extra ``ux`` column at ``ix=Nx`` and extra ``uy`` row
    at ``iy=Ny``.  Their matrix rows are identity ghost closures, so their
    values are exactly zero and can be restored without solving.
    """

    ux_array = np.asarray(ux, dtype=float)
    uy_array = np.asarray(uy, dtype=float)
    if ux_array.shape != (params.Ny + 1, params.Nx):
        raise ValueError(
            "ux must have shape "
            f"({params.Ny + 1}, {params.Nx}), got {ux_array.shape}."
        )
    if uy_array.shape != (params.Ny, params.Nx + 1):
        raise ValueError(
            "uy must have shape "
            f"({params.Ny}, {params.Nx + 1}), got {uy_array.shape}."
        )

    ux_full = np.zeros((params.Ny + 1, params.Nx + 1), dtype=float)
    uy_full = np.zeros_like(ux_full)
    ux_full[:, : params.Nx] = ux_array
    uy_full[: params.Ny, :] = uy_array
    solution = np.empty(2 * ux_full.size, dtype=float)
    solution[0::2] = ux_full.T.reshape(-1)
    solution[1::2] = uy_full.T.reshape(-1)
    return solution


def absolute_displacement_rhs(
    builder: MatrixBuilder,
    slip: np.ndarray,
    time: float,
) -> np.ndarray:
    """Construct ``b(U,t)`` from the velocity-form BP3 right-hand side.

    ``build_RH`` is linear in fault rate and prescribed side velocity.  The
    fault contribution is evaluated with accumulated slip ``U`` while only the
    external loading contribution is multiplied by time.
    """

    p = builder.p
    fault_slip = np.asarray(slip, dtype=float)
    if fault_slip.shape != (p.Ny,):
        raise ValueError(
            f"slip must have shape ({p.Ny},), got {fault_slip.shape}."
        )
    if not np.isfinite(time) or time < 0.0:
        raise ValueError("checkpoint time must be finite and non-negative.")
    zero = np.zeros(p.Ny, dtype=float)
    loading_rate_rhs = builder.build_RH(0.0, zero).copy()
    slip_plus_loading_rhs = builder.build_RH(0.0, fault_slip).copy()
    slip_rhs = slip_plus_loading_rhs - loading_rate_rhs
    return slip_rhs + float(time) * loading_rate_rhs


def _residual_summary(
    lhs: sparse.spmatrix,
    rhs: np.ndarray,
    solution: np.ndarray,
    residual: np.ndarray,
    rows: np.ndarray,
) -> ResidualSummary:
    rows = np.unique(np.asarray(rows, dtype=int))
    if rows.size == 0:
        return ResidualSummary(0.0, 0.0, 0.0, 0)
    lhs_rows = lhs[rows]
    rhs_rows = rhs[rows]
    selected = residual[rows]
    row_scale = np.asarray(abs(lhs_rows) @ np.abs(solution)).ravel()
    row_scale += np.abs(rhs_rows)
    # Use the absolute stencil action as the scale.  Normalising by
    # ||A*u||+||b|| is misleading for homogeneous continuity rows: both
    # signed vectors cancel to roundoff and can make a tiny residual appear to
    # have relative norm one.
    scale_l2 = max(np.linalg.norm(row_scale), np.finfo(float).tiny)
    significant = row_scale > max(float(np.max(row_scale)), 1.0) * 1e-14
    if np.any(significant):
        maximum_rowwise = float(
            np.max(np.abs(selected[significant]) / row_scale[significant])
        )
    else:
        maximum_rowwise = 0.0
    return ResidualSummary(
        relative_l2=float(np.linalg.norm(selected) / scale_l2),
        maximum_absolute=float(np.max(np.abs(selected))),
        maximum_rowwise_relative=maximum_rowwise,
        number_of_rows=int(rows.size),
    )


def equilibrium_residual_groups(
    lhs: sparse.spmatrix,
    rhs: np.ndarray,
    solution: np.ndarray,
    builder: MatrixBuilder,
) -> tuple[np.ndarray, dict[str, ResidualSummary]]:
    """Return the global residual and summaries for boundary/interface rows."""

    residual = np.asarray(lhs @ solution - rhs)
    p = builder.p
    mid = p.Nx // 2
    all_rows = np.arange(lhs.shape[0], dtype=int)
    physical = np.unique(
        np.concatenate((builder._kux.ravel(), builder._kuy.ravel()))
    )
    groups = {
        "all_rows": all_rows,
        "physical_rows": physical,
        "top": np.concatenate((builder._kux[0, :], builder._kuy[0, :])),
        "bottom": np.concatenate(
            (builder._kux[p.Ny, :], builder._kuy[p.Ny - 1, :])
        ),
        "fault_jump": builder._kuy[:, mid],
        "fault_traction_continuity": builder._kuy[:, mid + 1],
        "fault_ux_line": builder._kux[:, mid],
        "fault_top_corner": np.array(
            [builder._kux[0, mid], builder._kuy[0, mid],
             builder._kuy[0, mid + 1]],
            dtype=int,
        ),
        "fault_bottom_corner": np.array(
            [builder._kux[p.Ny, mid], builder._kuy[p.Ny - 1, mid],
             builder._kuy[p.Ny - 1, mid + 1]],
            dtype=int,
        ),
    }
    summaries = {
        name: _residual_summary(lhs, rhs, solution, residual, rows)
        for name, rows in groups.items()
    }
    return residual, summaries


def _validate_checkpoint(
    params: ModelParameters,
    checkpoint: dict[str, np.ndarray],
) -> None:
    required = {"U", "ux", "uy", "t"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing fields: {sorted(missing)}")
    if np.asarray(checkpoint["U"]).shape != (params.Ny,):
        raise ValueError("Checkpoint U does not match the configured Ny.")


def diagnose_bp3_operator_consistency(
    params: ModelParameters,
    checkpoint: dict[str, np.ndarray],
    *,
    modes: np.ndarray | None = None,
    project: bool = True,
) -> BP3OperatorConsistencyResult:
    """Diagnose one checkpoint, reusing one LU for projection and mode probes."""

    if params.case_type != CaseType.CALIFORNIA:
        raise ValueError("This diagnostic requires case_type='california'.")
    _validate_checkpoint(params, checkpoint)
    grid = Grid(params)
    builder = MatrixBuilder(params, grid)
    lhs = builder.build_LH().tocsc()
    slip = np.asarray(checkpoint["U"], dtype=float)
    time = float(np.asarray(checkpoint["t"]).item())
    rhs = absolute_displacement_rhs(builder, slip, time)
    stored = pack_staggered_fields(params, checkpoint["ux"], checkpoint["uy"])
    residual, groups = equilibrium_residual_groups(
        lhs, rhs, stored, builder
    )
    stress_util = StressCalUtil(prefer_numba=False)
    tau_stored, sigma_stored = _fault_tractions(
        params, grid, stress_util, stored
    )

    basis = None if modes is None else np.asarray(modes, dtype=float)
    if basis is not None:
        if basis.ndim == 1:
            basis = basis[:, None]
        if basis.ndim != 2 or basis.shape[0] != params.Ny:
            raise ValueError(
                f"modes must have shape ({params.Ny}, n_modes), got {basis.shape}."
            )
        if basis.shape[1] == 0 or np.any(~np.isfinite(basis)):
            raise ValueError("modes must contain at least one finite column.")

    solve = factorized(lhs) if project or basis is not None else None
    projected = None
    relative_difference = None
    tau_projected = None
    sigma_projected = None
    if project:
        projected = solve(rhs)
        physical = np.unique(
            np.concatenate((builder._kux.ravel(), builder._kuy.ravel()))
        )
        relative_difference = float(
            np.linalg.norm(projected[physical] - stored[physical])
            / max(np.linalg.norm(projected[physical]), np.finfo(float).tiny)
        )
        tau_projected, sigma_projected = _fault_tractions(
            params, grid, stress_util, projected
        )

    response = None
    reciprocity = None
    if basis is not None:
        zero = np.zeros(params.Ny, dtype=float)
        base_rhs = builder.build_RH(0.0, zero).copy()
        base = solve(base_rhs)
        base_tau, base_sigma = _fault_tractions(
            params, grid, stress_util, base
        )
        tau_response = np.empty_like(basis)
        sigma_response = np.empty_like(basis)
        for column in range(basis.shape[1]):
            modal = solve(builder.build_RH(0.0, basis[:, column]).copy())
            tau, sigma = _fault_tractions(params, grid, stress_util, modal)
            tau_response[:, column] = tau - base_tau
            sigma_response[:, column] = sigma - base_sigma
        response = FaultModeResponse(
            y=grid.y.copy(),
            modes=basis.copy(),
            tau=tau_response,
            sigma_effective=sigma_response,
        )
        reciprocity = diagnose_fault_reciprocity(response)

    checkpoint_result = CheckpointEquilibriumResult(
        y=grid.y.copy(),
        stored_solution=stored,
        absolute_rhs=rhs,
        residual=residual,
        residual_groups=groups,
        projected_solution=projected,
        displacement_relative_l2=relative_difference,
        fault_tau_stored=tau_stored,
        fault_sigma_effective_stored=sigma_stored,
        fault_tau_projected=tau_projected,
        fault_sigma_effective_projected=sigma_projected,
    )
    return BP3OperatorConsistencyResult(
        checkpoint=checkpoint_result,
        mode_response=response,
        reciprocity=reciprocity,
    )
