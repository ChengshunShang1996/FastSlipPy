r"""Separate BP3 fault-bottom corner coupling from deep interior coupling.

The default run compares Lz=160 km and Lz=240 km while keeping the 50 m
near-fault y mesh and approximately 1.1 km maximum outer-y spacing.  All fault
sources have the same integrated absolute strength (1 km equivalent width).
No earthquake-cycle integration or checkpoint is required.

Example::

    python run_bp3_deep_corner_diagnostic.py \
      --output-dir artifacts/bp3_deep_corner
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from fastslippy.pre_processing.grid import Grid
from fastslippy.pre_processing.model_parameters import ModelParameters
from fastslippy.utilities.bp3_interface_transfer import (
    deep_corner_source_modes,
    diagnose_bp3_interface_transfer,
)
from fastslippy.utilities.bp3_small_peak import (
    FaultModeResponse,
    diagnose_fault_reciprocity,
)


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


def _sample(y: np.ndarray, values: np.ndarray, depth: float) -> float:
    return float(np.interp(depth, y, values))


def _source_metrics(
    y: np.ndarray,
    labels: tuple[str, ...],
    modes: np.ndarray,
    coulomb: np.ndarray,
) -> dict[str, dict[str, float]]:
    weights = np.empty_like(y)
    spacing = np.diff(y)
    weights[0] = 0.5 * spacing[0]
    weights[-1] = 0.5 * spacing[-1]
    weights[1:-1] = 0.5 * (spacing[:-1] + spacing[1:])
    shallow = (y >= 5e3) & (y <= 15e3)
    metrics = {}
    for column, label in enumerate(labels):
        shallow_values = coulomb[shallow, column]
        scale = max(float(np.max(np.abs(shallow_values))), np.finfo(float).tiny)
        metrics[label] = {
            "equivalent_width_km": float(
                np.sum(weights * np.abs(modes[:, column])) / 1e3
            ),
            "peak_source_depth_km": float(
                y[int(np.argmax(np.abs(modes[:, column])))] / 1e3
            ),
            "coulomb_at_10km_pa_per_m": _sample(y, coulomb[:, column], 10e3),
            "coulomb_at_11km_pa_per_m": _sample(y, coulomb[:, column], 11e3),
            "coulomb_at_15km_pa_per_m": _sample(y, coulomb[:, column], 15e3),
            "shallow_5_15km_range_over_peak": float(
                np.ptp(shallow_values) / scale
            ),
        }
    return metrics


def _pairwise_reciprocity_payload(labels, reciprocity):
    work = reciprocity.work_matrix
    nucleation = labels.index("nucleation_10km")
    pairs = {}
    for index, label in enumerate(labels):
        forward = float(work[nucleation, index])
        reverse = float(work[index, nucleation])
        pairs[label] = {
            "nucleation_test_by_source": forward,
            "source_test_by_nucleation": reverse,
            "relative_difference": float(
                abs(forward - reverse)
                / max(abs(forward), abs(reverse), np.finfo(float).tiny)
            ),
        }
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/bp3_deep_corner")
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
    parser.add_argument("--friction-coefficient", type=float, default=0.6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = [case for case in DEFAULT_CASES if case.name in args.cases]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, object]] = {}

    for case in selected:
        print(f"[{case.name}] building Lz={case.ysize_km:g} km, Ny={case.ny}")
        params = build_parameters(args, case)
        grid = Grid(params)
        labels, modes = deep_corner_source_modes(
            grid.y,
            params.W_f,
            fixed_deep_depth=args.fixed_deep_depth_km * 1e3,
            equivalent_width=args.equivalent_width_km * 1e3,
        )
        result = diagnose_bp3_interface_transfer(
            params,
            friction_coefficient=args.friction_coefficient,
            source_labels=labels,
            source_modes=modes,
            creep_velocity=params.loading.V_L,
        )
        transfer = result.transfer
        coulomb_current = (
            transfer.tau
            - args.friction_coefficient * transfer.sigma_effective_current
        )
        coulomb_direct = (
            transfer.tau
            - args.friction_coefficient * transfer.sigma_effective_direct
        )
        reciprocity = diagnose_fault_reciprocity(
            FaultModeResponse(
                y=transfer.y,
                modes=transfer.source_modes,
                tau=transfer.tau,
                sigma_effective=transfer.sigma_effective_current,
            )
        )
        # Persist the expensive solve before computing presentation metrics.
        # A later reporting failure must not require another sparse LU.
        np.savez_compressed(
            args.output_dir / f"{case.name}_profiles.npz",
            y=transfer.y,
            source_labels=np.asarray(labels),
            source_modes=transfer.source_modes,
            tau=transfer.tau,
            sigma_effective_current=transfer.sigma_effective_current,
            sigma_effective_direct=transfer.sigma_effective_direct,
            coulomb_current=coulomb_current,
            coulomb_direct=coulomb_direct,
            reciprocity_work=reciprocity.work_matrix,
        )
        case_summary = {
            "case": asdict(case),
            "dof_count": int(grid.N),
            "max_dx_m": float(np.max(grid.dx_edges)),
            "max_dy_m": float(np.max(grid.dy_edges)),
            "source_metrics": _source_metrics(
                transfer.y,
                labels,
                transfer.source_modes,
                coulomb_current,
            ),
            "reciprocity": {
                "antisymmetric_fraction": reciprocity.antisymmetric_fraction,
                "maximum_pairwise_fraction": (
                    reciprocity.maximum_pairwise_fraction
                ),
                "nucleation_pairs": _pairwise_reciprocity_payload(
                    labels, reciprocity
                ),
            },
            "wf_endpoint_relative_coulomb_change": None,
        }
        wf = result.wf_loading
        band = (transfer.y >= 8e3) & (transfer.y <= 15e3)
        production = wf.tau_production - (
            args.friction_coefficient
            * wf.sigma_effective_current_production
        )
        delta = wf.delta_tau - (
            args.friction_coefficient * wf.delta_sigma_effective_current
        )
        case_summary["wf_endpoint_relative_coulomb_change"] = float(
            np.linalg.norm(delta[band])
            / max(np.linalg.norm(production[band]), np.finfo(float).tiny)
        )
        summaries[case.name] = case_summary
        (args.output_dir / f"{case.name}_summary.json").write_text(
            json.dumps(case_summary, indent=2),
            encoding="utf-8",
        )
        print(
            f"[{case.name}] reciprocity antisymmetric/max-pair: "
            f"{reciprocity.antisymmetric_fraction:.6e} / "
            f"{reciprocity.maximum_pairwise_fraction:.6e}"
        )
        del result, transfer, params, grid
        gc.collect()

    comparison: dict[str, object] = {}
    if "lz160" in summaries and "lz240" in summaries:
        baseline = summaries["lz160"]["source_metrics"]
        deeper = summaries["lz240"]["source_metrics"]
        for label in baseline:
            comparison[label] = {}
            for station in (
                "coulomb_at_10km_pa_per_m",
                "coulomb_at_11km_pa_per_m",
                "coulomb_at_15km_pa_per_m",
            ):
                reference = float(baseline[label][station])
                value = float(deeper[label][station])
                comparison[label][f"lz240_over_lz160_{station}"] = float(
                    value / reference
                    if reference != 0.0
                    else np.nan
                )

    payload = {
        "configuration": {
            "xsize_km": args.xsize_km,
            "nx": args.nx,
            "alpha": args.alpha,
            "wf_km": args.wf_km,
            "fixed_deep_depth_km": args.fixed_deep_depth_km,
            "equivalent_width_km": args.equivalent_width_km,
            "friction_coefficient": args.friction_coefficient,
        },
        "cases": summaries,
        "depth_comparison": comparison,
        "interpretation": [
            "All source columns have identical integrated absolute strength.",
            "Shear-traction reciprocity, not Coulomb symmetry, tests elastic work consistency.",
            "A strong bottom_endpoint response absent from deep_fixed sources isolates the fault-bottom corner.",
            "A response that follows near_bottom_smooth when Lz changes implicates proximity to the traction-free bottom.",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    lines = [
        "# BP3 deep-corner transfer diagnostic",
        "",
        "All sources use the same integrated absolute strength.",
        "",
    ]
    for case in selected:
        summary = summaries[case.name]
        lines.extend(
            [
                f"## {case.name}",
                "",
                f"Lz={case.ysize_km:g} km, Ny={case.ny}, "
                f"max(dy)={summary['max_dy_m']:.3f} m.",
                "",
                "| source | peak depth (km) | C(10 km) | C(11 km) | C(15 km) | shallow range/peak |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for label, metric in summary["source_metrics"].items():
            lines.append(
                f"| {label} | {metric['peak_source_depth_km']:.3f} | "
                f"{metric['coulomb_at_10km_pa_per_m']:.6e} | "
                f"{metric['coulomb_at_11km_pa_per_m']:.6e} | "
                f"{metric['coulomb_at_15km_pa_per_m']:.6e} | "
                f"{metric['shallow_5_15km_range_over_peak']:.6e} |"
            )
        lines.extend(
            [
                "",
                "Reciprocity antisymmetric/max-pair fractions: "
                f"`{summary['reciprocity']['antisymmetric_fraction']:.6e}` / "
                f"`{summary['reciprocity']['maximum_pairwise_fraction']:.6e}`.",
                "",
            ]
        )
    (args.output_dir / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Saved: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
