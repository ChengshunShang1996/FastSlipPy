r"""Run the static BP3 fault-traction audit across x-core resolutions.

The 100, 50, and 25 m cases keep the full x domain, y mesh, and number of
stretched outer-x intervals fixed.  Only the uniformly resolved fault-normal
core is refined.  Each case is solved independently and released before the
next sparse factorization is built.

Example::

    python run_bp3_x_core_convergence.py \
      --output-dir artifacts/bp3_x_core_convergence
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:
    from examples.run_bp3_fault_traction_audit import (
        DepthCase,
        build_parameters,
        run_case,
    )
except ModuleNotFoundError:  # Support copying both runners into an HPC case.
    from run_bp3_fault_traction_audit import (  # type: ignore[no-redef]
        DepthCase,
        build_parameters,
        run_case,
    )
from fastslippy.pre_processing.grid import Grid


@dataclass(frozen=True)
class XCoreCase:
    name: str
    core_dx_m: float
    nx: int
    x_inner_points: int


CORE_CASES = (
    XCoreCase("dx100", 100.0, 701, 201),
    XCoreCase("dx50", 50.0, 901, 401),
    XCoreCase("dx25", 25.0, 1301, 801),
)

DEPTH_CASE = DepthCase("lz160", 160.0, 651)
ENDPOINT_LABELS = {"surface_endpoint", "bottom_endpoint"}


def _work_antisymmetric_fraction(work: np.ndarray) -> float:
    matrix = np.asarray(work, dtype=float)
    symmetric = 0.5 * (matrix + matrix.T)
    antisymmetric = 0.5 * (matrix - matrix.T)
    return float(
        np.linalg.norm(antisymmetric)
        / max(np.linalg.norm(symmetric), np.finfo(float).tiny)
    )


def _relative_change(value: float, reference: float) -> float:
    return float(
        (value - reference) / max(abs(reference), np.finfo(float).tiny)
    )


def _case_metrics(summary: dict, arrays: dict[str, np.ndarray]) -> dict:
    labels = list(np.asarray(arrays["source_labels"]).astype(str))
    nucleation = labels.index("nucleation_10km")
    keep = [
        index for index, label in enumerate(labels)
        if label not in ENDPOINT_LABELS
    ]
    production_work = np.asarray(arrays["production_work"])
    direct_work = np.asarray(arrays["direct_trace_average_work"])
    production_self = float(production_work[nucleation, nucleation])
    direct_self = float(direct_work[nucleation, nucleation])
    return {
        "dof_count": summary["dof_count"],
        "min_dx_m": summary["min_dx_m"],
        "max_dx_m": summary["max_dx_m"],
        "max_adjacent_dx_ratio": summary["max_adjacent_dx_ratio"],
        "production_nucleation_self_work": production_self,
        "direct_nucleation_self_work": direct_self,
        "direct_over_production_nucleation_self": float(
            direct_self / production_self
        ),
        "direct_vs_production_nucleation_band_relative_l2": summary[
            "direct_vs_production_nucleation_band_relative_l2"
        ],
        "production_full_antisymmetric_fraction": summary[
            "production_reciprocity"
        ]["antisymmetric_fraction"],
        "direct_full_antisymmetric_fraction": summary[
            "direct_trace_average_reciprocity"
        ]["antisymmetric_fraction"],
        "production_without_exact_endpoints_antisymmetric_fraction": (
            _work_antisymmetric_fraction(
                production_work[np.ix_(keep, keep)]
            )
        ),
        "direct_without_exact_endpoints_antisymmetric_fraction": (
            _work_antisymmetric_fraction(direct_work[np.ix_(keep, keep)])
        ),
        "current_face_jump_relative_l2": summary[
            "current_face_jump_relative_l2"
        ],
        "direct_trace_face_jump_relative_l2": summary[
            "direct_trace_face_jump_relative_l2"
        ],
    }


def _convergence_payload(results: dict[str, dict]) -> dict[str, dict]:
    ordered = [case for case in CORE_CASES if case.name in results]
    if len(ordered) < 2:
        return {}
    fields = (
        "production_nucleation_self_work",
        "direct_nucleation_self_work",
        "direct_over_production_nucleation_self",
        "direct_vs_production_nucleation_band_relative_l2",
        "production_without_exact_endpoints_antisymmetric_fraction",
        "direct_without_exact_endpoints_antisymmetric_fraction",
    )
    comparisons = {}
    for coarse, fine in zip(ordered[:-1], ordered[1:]):
        coarse_values = results[coarse.name]
        fine_values = results[fine.name]
        comparisons[f"{fine.name}_vs_{coarse.name}"] = {
            field: _relative_change(fine_values[field], coarse_values[field])
            for field in fields
        }

    if len(ordered) == 3:
        extrapolation = {}
        for field in fields[:3]:
            q0, q1, q2 = (results[case.name][field] for case in ordered)
            difference_coarse = q0 - q1
            difference_fine = q1 - q2
            if difference_coarse * difference_fine > 0.0:
                ratio = abs(difference_coarse / difference_fine)
                order = float(np.log2(ratio)) if ratio > 0.0 else np.nan
                denominator = 2.0**order - 1.0
                limit = (
                    float(q2 + (q2 - q1) / denominator)
                    if np.isfinite(order) and abs(denominator) > 1e-12
                    else np.nan
                )
            else:
                order = np.nan
                limit = np.nan
            extrapolation[field] = {
                "observed_order": order,
                "richardson_limit": limit,
                "monotone": bool(difference_coarse * difference_fine > 0.0),
            }
        comparisons["three_level_extrapolation"] = extrapolation
    return comparisons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/bp3_x_core_convergence"),
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=[case.name for case in CORE_CASES],
        default=[case.name for case in CORE_CASES],
    )
    parser.add_argument("--xsize-km", type=float, default=320.0)
    parser.add_argument("--alpha", type=float, default=60.0)
    parser.add_argument("--wf-km", type=float, default=40.0)
    parser.add_argument("--plate-rate", type=float, default=1e-9)
    parser.add_argument("--x-inner-km", type=float, default=20.0)
    parser.add_argument("--y-inner-km", type=float, default=20.0)
    parser.add_argument("--y-inner-points", type=int, default=401)
    parser.add_argument("--stretch-power", type=int, default=2)
    parser.add_argument("--fixed-deep-depth-km", type=float, default=150.0)
    parser.add_argument("--equivalent-width-km", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = [case for case in CORE_CASES if case.name in args.cases]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    for core in selected:
        print(
            f"[{core.name}] core dx={core.core_dx_m:g} m, "
            f"Nx={core.nx}, x-inner points={core.x_inner_points}"
        )
        args.nx = core.nx
        args.x_inner_points = core.x_inner_points
        params = build_parameters(args, DEPTH_CASE)
        summary, arrays = run_case(
            params,
            fixed_deep_depth=args.fixed_deep_depth_km * 1e3,
            equivalent_width=args.equivalent_width_km * 1e3,
        )
        # The saved mode response has no x coordinate. Reconstructing the grid
        # is cheap and avoids retaining it through the sparse factorization.
        grid_dx = np.diff(Grid(params).x)
        summary["min_dx_m"] = float(np.min(grid_dx))
        summary["max_adjacent_dx_ratio"] = float(
            max(
                np.max(grid_dx[1:] / grid_dx[:-1]),
                np.max(grid_dx[:-1] / grid_dx[1:]),
            )
        )
        metrics = _case_metrics(summary, arrays)
        metrics["case"] = asdict(core)
        results[core.name] = metrics
        np.savez_compressed(
            args.output_dir / f"{core.name}_traction_audit.npz", **arrays
        )
        (args.output_dir / f"{core.name}_summary.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        smooth_reciprocity = metrics[
            "production_without_exact_endpoints_antisymmetric_fraction"
        ]
        print(
            f"[{core.name}] Ktrace/Kprod="
            f"{metrics['direct_over_production_nucleation_self']:.8f}, "
            f"smooth reciprocity="
            f"{smooth_reciprocity:.3e}"
        )
        del params, arrays, summary, grid_dx
        gc.collect()

    convergence = _convergence_payload(results)
    payload = {
        "configuration": {
            "depth_case": asdict(DEPTH_CASE),
            "xsize_km": args.xsize_km,
            "alpha": args.alpha,
            "wf_km": args.wf_km,
            "x_inner_km": args.x_inner_km,
            "y_inner_km": args.y_inner_km,
            "y_inner_points": args.y_inner_points,
            "fixed_outer_intervals_per_side": 250,
        },
        "results": results,
        "convergence": convergence,
        "interpretation": [
            (
                "Convergence of direct_over_production_nucleation_self toward "
                "one supports a collocation truncation error."
            ),
            (
                "A non-unit limiting ratio supports testing a paired trace-"
                "continuity and trace-friction operator; changing only stress "
                "post-processing would be inconsistent."
            ),
            (
                "Reciprocity without exact endpoint modes is the relevant bulk "
                "consistency metric; isolated endpoint modes are reported "
                "separately by the underlying traction audit."
            ),
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    lines = [
        "# BP3 x-core static convergence",
        "",
        (
            "The full x domain, y mesh, and 250 stretched outer intervals per "
            "side are fixed; only the uniform x core is refined."
        ),
        "",
        (
            "| case | core dx | Nx | max dx | Ktrace/Kprod | band difference | "
            "production reciprocity without endpoints |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for core in selected:
        item = results[core.name]
        lines.append(
            f"| {core.name} | {core.core_dx_m:g} m | {core.nx} | "
            f"{item['max_dx_m']:.3f} m | "
            f"{item['direct_over_production_nucleation_self']:.8f} | "
            f"{item['direct_vs_production_nucleation_band_relative_l2']:.6e} | "
            f"{item['production_without_exact_endpoints_antisymmetric_fraction']:.6e} |"
        )
    (args.output_dir / "diagnostic_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
