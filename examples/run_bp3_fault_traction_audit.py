r"""Audit BP3 fault shear traction against interface continuity and reciprocity.

The diagnostic performs static modal solves only.  It compares:

1. the production ``tauqs[:, mid]`` traction;
2. the two neighbouring stress columns used by the interface-continuity row;
3. an independent, one-sided quadratic extrapolation to the physical fault.

The third quantity is deliberately not labelled an exact reaction traction:
the current strong-form matrix replaces equilibrium rows and exposes no
Lagrange multiplier.  The comparison nevertheless separates a hidden
left/right recovery mismatch from a deeper interface/corner inconsistency.

Example::

    python run_bp3_fault_traction_audit.py \
      --output-dir artifacts/bp3_fault_traction_audit
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import factorized

from fastslippy.pre_processing.grid import Grid
from fastslippy.pre_processing.model_parameters import ModelParameters
from fastslippy.solver.matrix_builder import MatrixBuilder
from fastslippy.utilities.bp3_interface_transfer import (
    current_fault_shear_stress,
    deep_corner_source_modes,
    direct_fault_shear_stress,
)
from fastslippy.utilities.bp3_small_peak import (
    FaultModeResponse,
    _fault_tractions,
    diagnose_fault_reciprocity,
)
from fastslippy.utilities.stress_cal_util import StressCalUtil


@dataclass(frozen=True)
class DepthCase:
    name: str
    ysize_km: float
    ny: int


DEFAULT_CASES = (
    DepthCase("lz160", 160.0, 651),
    DepthCase("lz240", 240.0, 769),
)


def build_parameters(args: argparse.Namespace, case: DepthCase) -> ModelParameters:
    p = ModelParameters(
        case_type="california",
        alpha=args.alpha,
        motion_sign=-1,
        auto_motion_sign=True,
        xsize=args.xsize_km * 1e3,
        ysize=case.ysize_km * 1e3,
        Nx=args.nx,
        Ny=case.ny,
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
        H=15e3,
        h=3e3,
        W_f=args.wf_km * 1e3,
        x_stretch_enabled=True,
        y_stretch_enabled=True,
        x_stretch_inner_size=args.x_inner_km * 1e3,
        y_stretch_inner_size=args.y_inner_km * 1e3,
        x_stretch_inner_points=args.x_inner_points,
        y_stretch_inner_points=args.y_inner_points,
        x_stretch_power=args.stretch_power,
        y_stretch_power=args.stretch_power,
        allow_nonuniform_solver=True,
        output_vtk_option=False,
        fallback_to_iterative_on_oom=False,
        extrapolate_surface_fault_rate=False,
    )
    p.loading.V_p = args.plate_rate
    p.loading.V_L = args.plate_rate
    p.bc.left.ux.set_fixed()
    p.bc.left.uy.set_velocity(-0.5 * p.loading.V_p)
    p.bc.right.ux.set_fixed()
    p.bc.right.uy.set_velocity(0.5 * p.loading.V_p)
    p.bc.top.set_traction_free()
    p.bc.bottom.set_traction_free()
    p.layers.set_homogeneous(top=p.ysize, bottom=2.0 * p.ysize, a=p.a0, b=p.b0)
    p.apply_bp3_motion_sign()
    return p


def _relative_l2(numerator: np.ndarray, denominator: np.ndarray) -> float:
    return float(
        np.linalg.norm(numerator)
        / max(np.linalg.norm(denominator), np.finfo(float).tiny)
    )


def _reciprocity_payload(y, labels, modes, tau) -> dict[str, object]:
    result = diagnose_fault_reciprocity(
        FaultModeResponse(
            y=np.asarray(y),
            modes=np.asarray(modes),
            tau=np.asarray(tau),
            sigma_effective=np.zeros_like(tau),
        )
    )
    nucleation = labels.index("nucleation_10km")
    pairs = {}
    for index, label in enumerate(labels):
        forward = float(result.work_matrix[nucleation, index])
        reverse = float(result.work_matrix[index, nucleation])
        pairs[label] = {
            "nucleation_test_by_source": forward,
            "source_test_by_nucleation": reverse,
            "relative_difference": float(
                abs(forward - reverse)
                / max(abs(forward), abs(reverse), np.finfo(float).tiny)
            ),
        }
    return {
        "antisymmetric_fraction": result.antisymmetric_fraction,
        "maximum_pairwise_fraction": result.maximum_pairwise_fraction,
        "nucleation_pairs": pairs,
        "work_matrix": result.work_matrix,
    }


def _strip_work_matrix(payload: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in payload.items() if key != "work_matrix"}


def run_case(
    params: ModelParameters,
    *,
    fixed_deep_depth: float,
    equivalent_width: float,
):
    grid = Grid(params)
    labels, modes = deep_corner_source_modes(
        grid.y,
        params.W_f,
        fixed_deep_depth=fixed_deep_depth,
        equivalent_width=equivalent_width,
    )
    builder = MatrixBuilder(params, grid)
    solve = factorized(builder.build_LH().tocsc())
    stress_util = StressCalUtil(prefer_numba=False)

    def recover(solution):
        production, _ = _fault_tractions(params, grid, stress_util, solution)
        current = current_fault_shear_stress(
            params, grid, stress_util, solution
        )
        direct = direct_fault_shear_stress(params, grid, solution)
        return production, current, direct

    zero = np.zeros(params.Ny, dtype=float)
    base = solve(builder.build_RH(0.0, zero).copy())
    base_production, base_current, base_direct = recover(base)

    shape = modes.shape
    production = np.empty(shape)
    current_left = np.empty(shape)
    current_right = np.empty(shape)
    direct_left = np.empty(shape)
    direct_right = np.empty(shape)
    for column in range(shape[1]):
        solution = solve(builder.build_RH(0.0, modes[:, column]).copy())
        prod, current, direct = recover(solution)
        production[:, column] = prod - base_production
        current_left[:, column] = current.left - base_current.left
        current_right[:, column] = current.right - base_current.right
        direct_left[:, column] = direct.left - base_direct.left
        direct_right[:, column] = direct.right - base_direct.right

    current_average = 0.5 * (current_left + current_right)
    direct_average = 0.5 * (direct_left + direct_right)
    current_reciprocity = _reciprocity_payload(
        grid.y, labels, modes, current_average
    )
    direct_reciprocity = _reciprocity_payload(
        grid.y, labels, modes, direct_average
    )
    production_reciprocity = _reciprocity_payload(
        grid.y, labels, modes, production
    )

    band = (grid.y >= 5e3) & (grid.y <= 15e3)
    summary = {
        "dof_count": int(grid.N),
        "max_dx_m": float(np.max(grid.dx_edges)),
        "max_dy_m": float(np.max(grid.dy_edges)),
        "production_vs_current_average_relative_l2": _relative_l2(
            production - current_average, production
        ),
        "current_face_jump_relative_l2": _relative_l2(
            current_left - current_right, current_average
        ),
        "direct_trace_face_jump_relative_l2": _relative_l2(
            direct_left - direct_right, direct_average
        ),
        "direct_vs_production_relative_l2": _relative_l2(
            direct_average - production, production
        ),
        "direct_vs_production_nucleation_band_relative_l2": _relative_l2(
            direct_average[band] - production[band], production[band]
        ),
        "production_reciprocity": _strip_work_matrix(production_reciprocity),
        "current_face_average_reciprocity": _strip_work_matrix(
            current_reciprocity
        ),
        "direct_trace_average_reciprocity": _strip_work_matrix(
            direct_reciprocity
        ),
    }
    arrays = {
        "y": grid.y,
        "source_labels": np.asarray(labels),
        "source_modes": modes,
        "tau_production": production,
        "tau_current_left": current_left,
        "tau_current_right": current_right,
        "tau_direct_left": direct_left,
        "tau_direct_right": direct_right,
        "production_work": production_reciprocity["work_matrix"],
        "current_face_average_work": current_reciprocity["work_matrix"],
        "direct_trace_average_work": direct_reciprocity["work_matrix"],
    }
    return summary, arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/bp3_fault_traction_audit")
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=[case.name for case in DEFAULT_CASES],
        default=[case.name for case in DEFAULT_CASES],
    )
    parser.add_argument("--xsize-km", type=float, default=320.0)
    parser.add_argument("--nx", type=int, default=901)
    parser.add_argument("--alpha", type=float, default=60.0)
    parser.add_argument("--wf-km", type=float, default=40.0)
    parser.add_argument("--plate-rate", type=float, default=1e-9)
    parser.add_argument("--x-inner-km", type=float, default=20.0)
    parser.add_argument("--y-inner-km", type=float, default=20.0)
    parser.add_argument("--x-inner-points", type=int, default=401)
    parser.add_argument("--y-inner-points", type=int, default=401)
    parser.add_argument("--stretch-power", type=int, default=2)
    parser.add_argument("--fixed-deep-depth-km", type=float, default=150.0)
    parser.add_argument("--equivalent-width-km", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = [case for case in DEFAULT_CASES if case.name in args.cases]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for case in selected:
        print(f"[{case.name}] building Lz={case.ysize_km:g} km, Ny={case.ny}")
        params = build_parameters(args, case)
        summary, arrays = run_case(
            params,
            fixed_deep_depth=args.fixed_deep_depth_km * 1e3,
            equivalent_width=args.equivalent_width_km * 1e3,
        )
        summary["case"] = asdict(case)
        summaries[case.name] = summary
        np.savez_compressed(
            args.output_dir / f"{case.name}_traction_audit.npz", **arrays
        )
        (args.output_dir / f"{case.name}_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(
            f"[{case.name}] current face jump="
            f"{summary['current_face_jump_relative_l2']:.3e}, "
            f"direct face jump="
            f"{summary['direct_trace_face_jump_relative_l2']:.3e}"
        )
        direct_reciprocity = summary["direct_trace_average_reciprocity"]
        print(
            f"[{case.name}] reciprocity production/direct="
            f"{summary['production_reciprocity']['antisymmetric_fraction']:.3e}/"
            f"{direct_reciprocity['antisymmetric_fraction']:.3e}"
        )
        del params, arrays
        gc.collect()

    payload = {
        "configuration": {
            "xsize_km": args.xsize_km,
            "nx": args.nx,
            "alpha": args.alpha,
            "wf_km": args.wf_km,
            "fixed_deep_depth_km": args.fixed_deep_depth_km,
            "equivalent_width_km": args.equivalent_width_km,
        },
        "cases": summaries,
        "decision_rules": [
            (
                "Production matching the current-face average confirms that no "
                "post-processing branch changed tau."
            ),
            (
                "A tiny current-face jump confirms that the assembled interface-"
                "continuity row is satisfied; averaging is not hiding unequal "
                "face tractions."
            ),
            (
                "A much smaller direct-trace reciprocity defect implicates the "
                "location/recovery of fault traction."
            ),
            (
                "Comparable production and direct-trace reciprocity defects "
                "implicate the strong-form interface/free-boundary corner closure "
                "rather than the arithmetic average."
            ),
            (
                "Direct trace stress is an independent continuum recovery, not "
                "an exact jump-constraint reaction and is never symmetrized in "
                "this test."
            ),
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# BP3 fault-traction audit",
        "",
        (
            "No response matrix was symmetrized, and direct trace stress is not "
            "labelled an exact discrete reaction."
        ),
        "",
        (
            "| case | current face jump | direct face jump | production "
            "reciprocity | direct reciprocity |"
        ),
        "|---|---:|---:|---:|---:|",
    ]
    for case in selected:
        item = summaries[case.name]
        direct_reciprocity = item["direct_trace_average_reciprocity"]
        lines.append(
            f"| {case.name} | {item['current_face_jump_relative_l2']:.6e} | "
            f"{item['direct_trace_face_jump_relative_l2']:.6e} | "
            f"{item['production_reciprocity']['antisymmetric_fraction']:.6e} | "
            f"{direct_reciprocity['antisymmetric_fraction']:.6e} |"
        )
    (args.output_dir / "diagnostic_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
