r"""Low-rank spatial effective-stiffness diagnostic for BP3-QD nucleation.

The script reads a fault state from a checkpoint, applies a compact Gaussian
basis in the velocity-weakening nucleation band, and solves a reduced
generalized eigenproblem

    K_elastic phi = lambda K_critical phi.

It requires no earthquake-cycle integration.  Each mesh costs one sparse LU
factorization and ``number_of_basis_modes + 1`` backsolves.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fastslippy.pre_processing.frictional_zones import FrictionalZones
from fastslippy.pre_processing.grid import Grid
from fastslippy.pre_processing.model_parameters import ModelParameters
from fastslippy.utilities.bp3_small_peak import (
    NucleationStiffnessResult,
    diagnose_nucleation_stiffness,
    localized_gaussian_basis,
    signed_rate_state_friction_coefficient_profile,
    solve_fault_mode_response,
)


@dataclass(frozen=True)
class MeshCase:
    name: str
    xsize: float
    nx: int


DEFAULT_MESHES = (
    MeshCase("x320", 320e3, 901),
)


def parse_mesh(value: str) -> MeshCase:
    """Parse ``name:xsize_km:nx``."""
    try:
        name, size, nx = value.split(":")
        case = MeshCase(name=name, xsize=float(size) * 1e3, nx=int(nx))
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            "mesh must have the form name:xsize_km:nx"
        ) from exc
    if not case.name or case.xsize <= 20e3 or case.nx < 403 or case.nx % 2 == 0:
        raise argparse.ArgumentTypeError(
            "mesh requires a name, xsize_km > 20, and odd nx >= 403"
        )
    return case


def build_parameters(case: MeshCase) -> ModelParameters:
    p = ModelParameters(
        case_type="california",
        alpha=60.0,
        motion_sign=-1,
        auto_motion_sign=True,
        xsize=case.xsize,
        ysize=160e3,
        Nx=case.nx,
        Ny=651,
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
        W_f=40e3,
        x_stretch_enabled=True,
        y_stretch_enabled=True,
        x_stretch_inner_size=20e3,
        y_stretch_inner_size=20e3,
        x_stretch_inner_points=401,
        y_stretch_inner_points=401,
        x_stretch_power=2,
        y_stretch_power=2,
        allow_nonuniform_solver=True,
        output_vtk_option=False,
    )
    p.loading.V_p = 1e-9
    p.loading.V_L = 1e-9
    p.bc.left.ux.set_fixed()
    p.bc.left.uy.set_velocity(-0.5 * p.loading.V_p)
    p.bc.right.ux.set_fixed()
    p.bc.right.uy.set_velocity(0.5 * p.loading.V_p)
    p.bc.top.set_traction_free()
    p.bc.bottom.set_traction_free()
    return p


def load_checkpoint_state(path: Path, ny: int):
    with np.load(path) as data:
        required = {"sigma", "theta", "V", "t"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Checkpoint is missing fields: {sorted(missing)}")
        state = {
            name: np.asarray(data[name], dtype=float).copy()
            for name in ("sigma", "theta", "V")
        }
        state["time"] = float(data["t"])
    if any(state[name].shape != (ny,) for name in ("sigma", "theta", "V")):
        raise ValueError(f"Checkpoint fault fields must have shape ({ny},).")
    if np.any(state["sigma"] <= 0.0):
        raise ValueError("Checkpoint contains non-positive effective normal stress.")
    return state


def mode_metrics(result: NucleationStiffnessResult, count: int):
    weights = np.empty_like(result.y)
    spacing = np.diff(result.y)
    weights[0] = 0.5 * spacing[0]
    weights[-1] = 0.5 * spacing[-1]
    weights[1:-1] = 0.5 * (spacing[:-1] + spacing[1:])
    rows = []
    for index in range(min(count, result.spatial_modes.shape[1])):
        mode = result.spatial_modes[:, index]
        denominator = float(np.dot(weights, mode * mode))
        centre = float(np.dot(weights, result.y * mode * mode) / denominator)
        width = float(
            np.sqrt(
                np.dot(weights, (result.y - centre) ** 2 * mode * mode)
                / denominator
            )
        )
        tau_stiffness = float(
            -np.dot(weights * mode, result.tau_responses[:, index])
            / denominator
        )
        normal_stiffness = float(
            np.dot(
                weights * mode * result.friction_coefficient,
                result.sigma_effective_responses[:, index],
            ) / denominator
        )
        critical = float(
            np.dot(weights, result.critical_stiffness * mode * mode)
            / denominator
        )
        rows.append(
            {
                "mode": index + 1,
                "stiffness_ratio": float(result.stiffness_ratios[index]),
                "centre_depth_km": centre / 1e3,
                "rms_width_km": width / 1e3,
                "shear_stiffness_pa_per_m": tau_stiffness,
                "normal_coupling_stiffness_pa_per_m": normal_stiffness,
                "total_stiffness_pa_per_m": tau_stiffness + normal_stiffness,
                "critical_stiffness_pa_per_m": critical,
                "normal_fraction_of_total": normal_stiffness
                / max(abs(tau_stiffness + normal_stiffness), 1e-300),
            }
        )
    return rows


def save_case(
    path: Path,
    case: MeshCase,
    result: NucleationStiffnessResult,
    checkpoint_time: float,
    elapsed: float,
    mode_count: int,
):
    rows = mode_metrics(result, mode_count)
    np.savez_compressed(
        path,
        y=result.y,
        basis=result.basis,
        mass_matrix=result.mass_matrix,
        elastic_stiffness_matrix=result.elastic_stiffness_matrix,
        critical_stiffness_matrix=result.critical_stiffness_matrix,
        stiffness_ratios=result.stiffness_ratios,
        coefficient_modes=result.coefficient_modes,
        spatial_modes=result.spatial_modes,
        tau_responses=result.tau_responses,
        sigma_effective_responses=result.sigma_effective_responses,
        coulomb_responses=result.coulomb_responses,
        friction_coefficient=result.friction_coefficient,
        critical_stiffness=result.critical_stiffness,
        antisymmetric_fraction=result.antisymmetric_fraction,
        checkpoint_time=checkpoint_time,
        elapsed_seconds=elapsed,
        xsize=case.xsize,
        nx=case.nx,
    )
    return rows


def load_case(path: Path, mode_count: int):
    with np.load(path) as data:
        result = NucleationStiffnessResult(
            y=np.asarray(data["y"]),
            basis=np.asarray(data["basis"]),
            mass_matrix=np.asarray(data["mass_matrix"]),
            elastic_stiffness_matrix=np.asarray(data["elastic_stiffness_matrix"]),
            critical_stiffness_matrix=np.asarray(data["critical_stiffness_matrix"]),
            stiffness_ratios=np.asarray(data["stiffness_ratios"]),
            coefficient_modes=np.asarray(data["coefficient_modes"]),
            spatial_modes=np.asarray(data["spatial_modes"]),
            tau_responses=np.asarray(data["tau_responses"]),
            sigma_effective_responses=np.asarray(data["sigma_effective_responses"]),
            coulomb_responses=np.asarray(data["coulomb_responses"]),
            friction_coefficient=np.asarray(data["friction_coefficient"]),
            critical_stiffness=np.asarray(data["critical_stiffness"]),
            antisymmetric_fraction=float(data["antisymmetric_fraction"]),
        )
        metadata = {
            "checkpoint_time": float(data["checkpoint_time"]),
            "elapsed_seconds": float(data["elapsed_seconds"]),
        }
    return result, mode_metrics(result, mode_count), metadata


def plot_results(output_dir: Path, results: dict[str, NucleationStiffnessResult]):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for name, result in results.items():
        count = min(8, result.stiffness_ratios.size)
        axes[0].plot(
            np.arange(1, count + 1), result.stiffness_ratios[:count],
            marker="o", label=name,
        )
        for index in range(min(3, count)):
            axes[1].plot(
                result.spatial_modes[:, index], result.y / 1e3,
                label=f"{name} m{index + 1}: {result.stiffness_ratios[index]:.3f}",
            )
        mode = result.spatial_modes[:, 0]
        axes[2].plot(
            -result.tau_responses[:, 0], result.y / 1e3,
            label=f"{name}: shear",
        )
        axes[2].plot(
            result.friction_coefficient
            * result.sigma_effective_responses[:, 0],
            result.y / 1e3,
            linestyle="--", label=f"{name}: normal",
        )
        axes[2].plot(
            -result.coulomb_responses[:, 0], result.y / 1e3,
            linewidth=2, label=f"{name}: total",
        )
    axes[0].axhline(1.0, color="black", linestyle=":", linewidth=1)
    axes[0].set(xlabel="Reduced mode", ylabel=r"$k_{eff}/k_c$")
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Normalized mode amplitude", ylabel="Depth (km)")
    axes[1].invert_yaxis()
    axes[1].legend(fontsize=7)
    axes[2].set(xlabel="Restoring response (Pa/m)", ylabel="Depth (km)")
    axes[2].invert_yaxis()
    axes[2].legend(fontsize=7)
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "nucleation_stiffness_modes.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/bp3_nucleation_stiffness"),
    )
    parser.add_argument(
        "--meshes", nargs="+", type=parse_mesh, default=DEFAULT_MESHES,
        metavar="NAME:XSIZE_KM:NX",
    )
    parser.add_argument("--band-top-km", type=float, default=8.0)
    parser.add_argument("--band-bottom-km", type=float, default=15.0)
    parser.add_argument("--basis-spacing-km", type=float, default=1.0)
    parser.add_argument("--basis-width-km", type=float, default=0.75)
    parser.add_argument("--report-modes", type=int, default=5)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    source = args.checkpoint.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state = load_checkpoint_state(source, ny=651)
    all_results = {}
    summaries = {}
    table_rows = []

    for case in args.meshes:
        params = build_parameters(case)
        grid = Grid(params)
        basis = localized_gaussian_basis(
            grid.y,
            top=args.band_top_km * 1e3,
            bottom=args.band_bottom_km * 1e3,
            spacing=args.basis_spacing_km * 1e3,
            width=args.basis_width_km * 1e3,
        )
        saved_path = args.output_dir / f"{case.name}_stiffness_modes.npz"
        if args.reuse_existing and saved_path.exists():
            result, rows, metadata = load_case(
                saved_path, args.report_modes
            )
        else:
            started = time.perf_counter()
            response = solve_fault_mode_response(params, basis)
            friction = FrictionalZones(params, grid.y)
            mu = signed_rate_state_friction_coefficient_profile(
                state["V"], state["theta"],
                a=friction.a,
                b=friction.b,
                mu0=params.mu0,
                V0=params.V0,
                L=params.L,
            )
            critical = state["sigma"] * (friction.b - friction.a) / params.L
            result = diagnose_nucleation_stiffness(
                response,
                friction_coefficient=mu,
                critical_stiffness_profile=critical,
            )
            elapsed = time.perf_counter() - started
            rows = save_case(
                saved_path, case, result, state["time"], elapsed,
                args.report_modes,
            )
            metadata = {
                "checkpoint_time": state["time"],
                "elapsed_seconds": elapsed,
            }
        all_results[case.name] = result
        summaries[case.name] = {
            "mesh": asdict(case),
            "paper_Lx_km": case.xsize / 2e3,
            "max_dx_m": float(np.max(grid.dx_edges)),
            "basis_modes": int(basis.shape[1]),
            "antisymmetric_fraction": result.antisymmetric_fraction,
            **metadata,
            "modes": rows,
        }
        table_rows.extend({"case": case.name, **row} for row in rows)
        print(
            f"[{case.name}] softest k_eff/k_c={result.stiffness_ratios[0]:.6g}, "
            f"elapsed={metadata['elapsed_seconds']:.1f}s"
        )

    plot_results(args.output_dir, all_results)
    payload = {
        "source_checkpoint": str(source),
        "band_km": [args.band_top_km, args.band_bottom_km],
        "basis_spacing_km": args.basis_spacing_km,
        "basis_width_km": args.basis_width_km,
        "interpretation": (
            "k_eff/k_c < 1 is softer than the local aging-law critical "
            "stiffness; this is a quasistatic modal diagnostic, not a complete "
            "dynamic stability proof."
        ),
        "cases": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    if table_rows:
        with (args.output_dir / "modes.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table_rows[0]))
            writer.writeheader()
            writer.writerows(table_rows)
    lines = [
        "# BP3 nucleation-mode effective stiffness",
        "",
        f"Checkpoint: `{source}`",
        "",
        "| case | mode | k_eff/k_c | centre (km) | width (km) | normal fraction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        lines.append(
            f"| {row['case']} | {row['mode']} | {row['stiffness_ratio']:.6g} "
            f"| {row['centre_depth_km']:.3f} | {row['rms_width_km']:.3f} "
            f"| {row['normal_fraction_of_total']:.4g} |"
        )
    (args.output_dir / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Saved: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
