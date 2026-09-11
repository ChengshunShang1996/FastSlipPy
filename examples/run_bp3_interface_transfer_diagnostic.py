r"""Diagnose BP3 normal-stress recovery and endpoint-to-nucleation transfer.

This is a static diagnostic: it performs one sparse factorization and a small
number of backsolves, with no earthquake-cycle time integration.  A checkpoint
is used only to infer ``Nx``/``Ny`` and, when available, the instantaneous
friction coefficient from ``V`` and ``theta``.

Example::

    python run_bp3_interface_transfer_diagnostic.py output/data_23000.npz \
      --xsize-km 320 --ysize-km 160 \
      --x-inner-km 20 --y-inner-km 20 \
      --x-inner-points 401 --y-inner-points 401
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fastslippy.pre_processing.frictional_zones import FrictionalZones
from fastslippy.pre_processing.grid import Grid
from fastslippy.pre_processing.model_parameters import ModelParameters
from fastslippy.utilities.bp3_interface_transfer import (
    current_fault_normal_stress,
    diagnose_bp3_interface_transfer,
    direct_fault_normal_stress,
)
from fastslippy.utilities.bp3_operator_consistency import pack_staggered_fields
from fastslippy.utilities.stress_cal_util import StressCalUtil


def build_parameters(args: argparse.Namespace, nx: int, ny: int) -> ModelParameters:
    p = ModelParameters(
        case_type="california",
        alpha=args.alpha,
        motion_sign=args.motion_sign,
        auto_motion_sign=True,
        xsize=args.xsize_km * 1e3,
        ysize=args.ysize_km * 1e3,
        Nx=nx,
        Ny=ny,
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
        x_stretch_enabled=not args.uniform_x,
        y_stretch_enabled=not args.uniform_y,
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
    p.layers.set_homogeneous(
        top=p.ysize, bottom=2.0 * p.ysize, a=p.a0, b=p.b0
    )
    p.apply_bp3_motion_sign()
    return p


def load_checkpoint(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: np.asarray(data[name]).copy() for name in data.files}


def checkpoint_shape(checkpoint: dict[str, np.ndarray]) -> tuple[int, int]:
    if "U" not in checkpoint or "ux" not in checkpoint or "uy" not in checkpoint:
        raise ValueError("checkpoint must contain U, ux, and uy")
    ny = int(np.asarray(checkpoint["U"]).size)
    ux = np.asarray(checkpoint["ux"])
    uy = np.asarray(checkpoint["uy"])
    if ux.ndim != 2 or uy.ndim != 2:
        raise ValueError("checkpoint ux and uy must be two-dimensional")
    nx = int(ux.shape[1])
    if ux.shape != (ny + 1, nx) or uy.shape != (ny, nx + 1):
        raise ValueError(
            f"inconsistent checkpoint shapes: U={(ny,)}, ux={ux.shape}, "
            f"uy={uy.shape}"
        )
    return nx, ny


def _stable_asinh_exp(log_value: np.ndarray) -> np.ndarray:
    result = np.empty_like(log_value)
    large = log_value > 20.0
    result[large] = log_value[large] + np.log1p(
        np.sqrt(1.0 + np.exp(-2.0 * log_value[large]))
    )
    result[~large] = np.arcsinh(np.exp(log_value[~large]))
    return result


def checkpoint_friction_coefficient(
    checkpoint: dict[str, np.ndarray], params: ModelParameters, grid: Grid
) -> tuple[np.ndarray, str]:
    if "V" not in checkpoint or "theta" not in checkpoint:
        return np.full(params.Ny, params.mu0), "constant_mu0"
    velocity = np.asarray(checkpoint["V"], dtype=float)
    theta = np.asarray(checkpoint["theta"], dtype=float)
    if velocity.shape != (params.Ny,) or theta.shape != (params.Ny,):
        raise ValueError("checkpoint V and theta must match Ny")
    friction = FrictionalZones(params, grid.y)
    speed = np.maximum(np.abs(velocity), np.nextafter(0.0, 1.0))
    log_q = (
        np.log(speed)
        - np.log(2.0 * params.V0)
        + (
            params.mu0
            + friction.b * np.log(params.V0 * theta / params.L)
        )
        / friction.a
    )
    coefficient = friction.a * _stable_asinh_exp(log_q)
    return coefficient, "checkpoint_V_theta"


def station_rows(result, mu: np.ndarray) -> dict[str, dict[str, float]]:
    y = result.transfer.y
    wf = result.wf_loading
    changed = np.flatnonzero(wf.production_rate != wf.rs_endpoint_rate)
    wf_depth_km = float(y[changed[0]] / 1e3) if changed.size else 40.0
    rows = {}
    for depth_km in (5.0, 10.0, 11.0, 15.0, wf_depth_km):
        index = int(np.argmin(np.abs(y - depth_km * 1e3)))
        delta_tau = float(wf.delta_tau[index])
        delta_current = float(wf.delta_sigma_effective_current[index])
        delta_direct = float(wf.delta_sigma_effective_direct[index])
        rows[f"{depth_km:g}km"] = {
            "grid_depth_km": float(y[index] / 1e3),
            "delta_tau_pa_per_s": delta_tau,
            "delta_sigma_current_pa_per_s": delta_current,
            "delta_sigma_direct_pa_per_s": delta_direct,
            "delta_coulomb_current_pa_per_s": (
                delta_tau - float(mu[index]) * delta_current
            ),
            "delta_coulomb_direct_pa_per_s": (
                delta_tau - float(mu[index]) * delta_direct
            ),
        }
    return rows


def relative_norm(numerator: np.ndarray, denominator: np.ndarray) -> float:
    return float(
        np.linalg.norm(numerator)
        / max(np.linalg.norm(denominator), np.finfo(float).tiny)
    )


def checkpoint_normal_recovery(
    checkpoint: dict[str, np.ndarray],
    params: ModelParameters,
    grid: Grid,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Compare the production and direct recovery on a stored displacement."""

    solution = pack_staggered_fields(
        params,
        np.asarray(checkpoint["ux"], dtype=float),
        np.asarray(checkpoint["uy"], dtype=float),
    )
    current = (
        current_fault_normal_stress(
            params,
            grid,
            StressCalUtil(prefer_numba=False),
            solution,
        ).effective_average
        + params.sigma0
    )
    direct = direct_fault_normal_stress(
        params, grid, solution
    ).effective_average + params.sigma0
    difference = direct - current
    perturbation = current - params.sigma0
    creep_start = int(np.searchsorted(grid.y, params.W_f, side="left"))
    masks = {
        "all": np.ones(params.Ny, dtype=bool),
        "rate_state": np.arange(params.Ny) < creep_start,
        "nucleation_8_15km": (grid.y >= 8e3) & (grid.y <= 15e3),
    }
    metrics: dict[str, object] = {
        "first_creep_index": creep_start,
        "first_creep_depth_km": (
            float(grid.y[creep_start] / 1e3)
            if creep_start < params.Ny
            else None
        ),
    }
    for label, mask in masks.items():
        metrics[f"{label}_relative_l2"] = relative_norm(
            difference[mask], perturbation[mask]
        )
        metrics[f"{label}_maximum_absolute_kpa"] = float(
            np.max(np.abs(difference[mask])) / 1e3
        )
    if "sigma" in checkpoint:
        stored = np.asarray(checkpoint["sigma"], dtype=float)
        if stored.shape != (params.Ny,):
            raise ValueError("checkpoint sigma must match Ny")
        metrics["stored_vs_recomputed_current_maximum_absolute_pa"] = float(
            np.max(np.abs(stored - current))
        )

    stations: dict[str, dict[str, float]] = {}
    for depth_km in (5.0, 10.0, 11.0, 15.0, params.W_f / 1e3):
        index = int(np.argmin(np.abs(grid.y - depth_km * 1e3)))
        stations[f"{depth_km:g}km"] = {
            "grid_depth_km": float(grid.y[index] / 1e3),
            "current_mpa": float(current[index] / 1e6),
            "direct_mpa": float(direct[index] / 1e6),
            "direct_minus_current_kpa": float(difference[index] / 1e3),
        }
    metrics["stations"] = stations
    profiles = {
        "y": grid.y.copy(),
        "sigma_effective_current": current,
        "sigma_effective_direct": direct,
        "direct_minus_current": difference,
    }
    return metrics, profiles


def recovery_mismatch_payload(result) -> dict[str, dict[str, float]]:
    transfer = result.transfer
    changed = np.flatnonzero(
        result.wf_loading.production_rate
        != result.wf_loading.rs_endpoint_rate
    )
    creep_start = int(changed[0]) if changed.size else transfer.y.size
    difference = (
        transfer.sigma_effective_direct
        - transfer.sigma_effective_current
    )
    masks = {
        "all": np.ones(transfer.y.size, dtype=bool),
        "rate_state": np.arange(transfer.y.size) < creep_start,
        "nucleation_8_15km": (
            (transfer.y >= 8e3) & (transfer.y <= 15e3)
        ),
    }
    payload = {}
    for column, label in enumerate(transfer.source_labels):
        payload[label] = {}
        for mask_label, mask in masks.items():
            payload[label][f"{mask_label}_relative_l2"] = relative_norm(
                difference[mask, column],
                transfer.sigma_effective_current[mask, column],
            )
        payload[label]["maximum_absolute_pa_per_m"] = float(
            np.max(np.abs(difference[:, column]))
        )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/bp3_interface_transfer"),
    )
    parser.add_argument("--xsize-km", type=float, default=320.0)
    parser.add_argument("--ysize-km", type=float, default=160.0)
    parser.add_argument("--alpha", type=float, default=60.0)
    parser.add_argument("--motion-sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--wf-km", type=float, default=40.0)
    parser.add_argument("--plate-rate", type=float, default=1e-9)
    parser.add_argument("--x-inner-km", type=float, default=20.0)
    parser.add_argument("--y-inner-km", type=float, default=20.0)
    parser.add_argument("--x-inner-points", type=int, default=401)
    parser.add_argument("--y-inner-points", type=int, default=401)
    parser.add_argument("--stretch-power", type=int, default=2)
    parser.add_argument("--uniform-x", action="store_true")
    parser.add_argument("--uniform-y", action="store_true")
    parser.add_argument(
        "--checkpoint-recovery-only",
        action="store_true",
        help=(
            "only compare normal-stress recoveries on the stored displacement; "
            "skip sparse factorization and endpoint-transfer solves"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)
    nx, ny = checkpoint_shape(checkpoint)
    if not args.uniform_x and args.x_inner_points > nx:
        raise ValueError("x-inner-points cannot exceed checkpoint Nx")
    if not args.uniform_y and args.y_inner_points > ny:
        raise ValueError("y-inner-points cannot exceed checkpoint Ny")

    params = build_parameters(args, nx, ny)
    grid = Grid(params)
    mu, mu_source = checkpoint_friction_coefficient(checkpoint, params, grid)
    checkpoint_audit, checkpoint_profiles = checkpoint_normal_recovery(
        checkpoint, params, grid
    )
    configuration = {
        "nx": nx,
        "ny": ny,
        "xsize_km": params.xsize / 1e3,
        "ysize_km": params.ysize / 1e3,
        "alpha": params.alpha,
        "motion_sign": params.motion_sign,
        "wf_km": params.W_f / 1e3,
        "friction_coefficient_source": mu_source,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.checkpoint_recovery_only:
        payload = {
            "source_checkpoint": str(checkpoint_path),
            "configuration": configuration,
            "checkpoint_normal_recovery": checkpoint_audit,
            "interpretation": [
                "The checkpoint sigma must match the recomputed current recovery to exclude stale or unsynchronised output.",
                "Direct recovery is an independent one-sided fault-trace estimate, not a declared exact reaction traction.",
                "A mismatch localized at H or W_f motivates an interface-closure test; a small mismatch at 10--11 km argues against local interpolation as the direct trigger.",
            ],
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        np.savez_compressed(
            args.output_dir / "diagnostic_profiles.npz",
            **checkpoint_profiles,
        )
        lines = [
            "# BP3 checkpoint normal-traction recovery diagnostic",
            "",
            f"Checkpoint: `{checkpoint_path}`",
            "",
            "| station | grid depth (km) | current (MPa) | direct (MPa) | direct-current (kPa) |",
            "|---|---:|---:|---:|---:|",
        ]
        for label, row in checkpoint_audit["stations"].items():
            lines.append(
                f"| {label} | {row['grid_depth_km']:.6f} | "
                f"{row['current_mpa']:.9f} | {row['direct_mpa']:.9f} | "
                f"{row['direct_minus_current_kpa']:.6f} |"
            )
        lines.extend(
            [
                "",
                "8--15 km relative L2 mismatch: "
                f"`{checkpoint_audit['nucleation_8_15km_relative_l2']:.6e}`; "
                "maximum absolute mismatch: "
                f"`{checkpoint_audit['nucleation_8_15km_maximum_absolute_kpa']:.6e} kPa`.",
            ]
        )
        (args.output_dir / "summary.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        print(
            "Checkpoint normal-recovery mismatch in 8--15 km "
            f"(relative L2 / max kPa): "
            f"{checkpoint_audit['nucleation_8_15km_relative_l2']:.6e} / "
            f"{checkpoint_audit['nucleation_8_15km_maximum_absolute_kpa']:.6e}"
        )
        print(f"Saved: {args.output_dir.resolve()}")
        return

    result = diagnose_bp3_interface_transfer(
        params,
        friction_coefficient=mu,
        creep_velocity=params.loading.V_L,
    )
    transfer = result.transfer
    wf = result.wf_loading
    band = (transfer.y >= 8e3) & (transfer.y <= 15e3)
    production_coulomb_current = (
        wf.tau_production
        - mu * wf.sigma_effective_current_production
    )
    production_coulomb_direct = (
        wf.tau_production
        - mu * wf.sigma_effective_direct_production
    )
    delta_coulomb_current = (
        wf.delta_tau - mu * wf.delta_sigma_effective_current
    )
    delta_coulomb_direct = (
        wf.delta_tau - mu * wf.delta_sigma_effective_direct
    )

    payload = {
        "source_checkpoint": str(checkpoint_path),
        "configuration": configuration,
        "checkpoint_normal_recovery": checkpoint_audit,
        "source_labels": list(transfer.source_labels),
        "receiver_labels": list(transfer.receiver_labels),
        "projected_tau_pa_per_m": transfer.projected_tau.tolist(),
        "projected_sigma_current_pa_per_m": (
            transfer.projected_sigma_effective_current.tolist()
        ),
        "projected_sigma_direct_pa_per_m": (
            transfer.projected_sigma_effective_direct.tolist()
        ),
        "projected_coulomb_current_pa_per_m": (
            transfer.projected_coulomb_current.tolist()
        ),
        "projected_coulomb_direct_pa_per_m": (
            transfer.projected_coulomb_direct.tolist()
        ),
        "normal_recovery_mismatch": recovery_mismatch_payload(result),
        "wf_endpoint_convention": {
            "changed_grid_indices": np.flatnonzero(
                wf.production_rate != wf.rs_endpoint_rate
            ).tolist(),
            "stations": station_rows(result, mu),
            "nucleation_band_relative_coulomb_change_current": relative_norm(
                delta_coulomb_current[band],
                production_coulomb_current[band],
            ),
            "nucleation_band_relative_coulomb_change_direct": relative_norm(
                delta_coulomb_direct[band],
                production_coulomb_direct[band],
            ),
        },
        "interpretation": [
            "Direct recovery is an independent one-sided fault-trace estimate, not a declared exact reaction traction.",
            "The W_f A/B test changes one nodal loading value only; it does not reproduce sbplib's four-block SAT junction.",
            "Large transfer or recovery differences in 8--15 km justify a closure change before another long cycle run.",
        ],
    }

    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "diagnostic_profiles.npz",
        y=transfer.y,
        friction_coefficient=mu,
        source_modes=transfer.source_modes,
        receiver_modes=transfer.receiver_modes,
        tau=transfer.tau,
        sigma_effective_current=transfer.sigma_effective_current,
        sigma_effective_direct=transfer.sigma_effective_direct,
        production_rate=wf.production_rate,
        rs_endpoint_rate=wf.rs_endpoint_rate,
        delta_tau=wf.delta_tau,
        delta_sigma_effective_current=wf.delta_sigma_effective_current,
        delta_sigma_effective_direct=wf.delta_sigma_effective_direct,
        checkpoint_sigma_effective_current=(
            checkpoint_profiles["sigma_effective_current"]
        ),
        checkpoint_sigma_effective_direct=(
            checkpoint_profiles["sigma_effective_direct"]
        ),
        checkpoint_direct_minus_current=(
            checkpoint_profiles["direct_minus_current"]
        ),
    )

    lines = [
        "# BP3 interface/normal-traction transfer diagnostic",
        "",
        f"Checkpoint: `{checkpoint_path}`",
        f"Friction coefficient: `{mu_source}`",
        "",
        "## Endpoint-to-nucleation Coulomb transfer (current recovery)",
        "",
        "| receiver | " + " | ".join(transfer.source_labels) + " |",
        "|---|" + "---:|" * len(transfer.source_labels),
    ]
    for row, label in enumerate(transfer.receiver_labels):
        values = " | ".join(
            f"{value:.6e}"
            for value in transfer.projected_coulomb_current[row, :]
        )
        lines.append(f"| {label} | {values} |")
    lines.extend(
        [
            "",
            "## W_f endpoint convention",
            "",
            "Current/direct recovery relative Coulomb changes in 8--15 km: "
            f"`{payload['wf_endpoint_convention']['nucleation_band_relative_coulomb_change_current']:.6e}` / "
            f"`{payload['wf_endpoint_convention']['nucleation_band_relative_coulomb_change_direct']:.6e}`.",
            "",
            "Detailed normal-recovery mismatches and station values are in `summary.json`; full profiles are in `diagnostic_profiles.npz`.",
        ]
    )
    (args.output_dir / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        "W_f endpoint relative Coulomb change in 8--15 km "
        f"(current/direct): "
        f"{payload['wf_endpoint_convention']['nucleation_band_relative_coulomb_change_current']:.6e} / "
        f"{payload['wf_endpoint_convention']['nucleation_band_relative_coulomb_change_direct']:.6e}"
    )
    print(f"Saved: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
