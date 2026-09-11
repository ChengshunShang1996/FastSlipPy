r"""Diagnose BP3 checkpoint equilibrium and fault-traction reciprocity.

The production-size run performs one sparse LU factorization and reuses it for
an absolute-displacement checkpoint projection and a small set of localized
fault modes.  It is therefore intended for the same HPC environment as the
nucleation-stiffness diagnostic, but requires no earthquake-cycle integration.

Existing ``*_stiffness_modes.npz`` files can also be supplied with
``--mode-files``.  Their interior reciprocity metrics are evaluated without a
new elastic solve.

Example::

    python run_bp3_operator_consistency_diagnostic.py output/data_23000.npz \
      --mode-files ../BP3-nucleation-stiffness-2/output/*_stiffness_modes.npz
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from fastslippy.pre_processing.model_parameters import ModelParameters
from fastslippy.utilities.bp3_operator_consistency import (
    diagnose_bp3_operator_consistency,
)
from fastslippy.utilities.bp3_small_peak import (
    FaultModeResponse,
    diagnose_fault_reciprocity,
)


YEAR = 365.0 * 24.0 * 3600.0


def build_parameters(
    *,
    nx: int,
    ny: int,
    xsize: float,
    ysize: float,
    alpha: float,
    motion_sign: int,
    x_inner_size: float,
    y_inner_size: float,
    x_inner_points: int,
    y_inner_points: int,
) -> ModelParameters:
    """Build the stretched BP3 configuration used by the current runs."""

    p = ModelParameters(
        case_type="california",
        alpha=alpha,
        motion_sign=motion_sign,
        auto_motion_sign=True,
        xsize=xsize,
        ysize=ysize,
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
        W_f=40e3,
        x_stretch_enabled=True,
        y_stretch_enabled=True,
        x_stretch_inner_size=x_inner_size,
        y_stretch_inner_size=y_inner_size,
        x_stretch_inner_points=x_inner_points,
        y_stretch_inner_points=y_inner_points,
        x_stretch_power=2,
        y_stretch_power=2,
        allow_nonuniform_solver=True,
        output_vtk_option=False,
        fallback_to_iterative_on_oom=False,
        extrapolate_surface_fault_rate=False,
    )
    p.loading.V_p = 1e-9
    p.loading.V_L = 1e-9
    p.bc.left.ux.set_fixed()
    p.bc.left.uy.set_velocity(-0.5 * p.loading.V_p)
    p.bc.right.ux.set_fixed()
    p.bc.right.uy.set_velocity(0.5 * p.loading.V_p)
    p.bc.top.set_traction_free()
    p.bc.bottom.set_traction_free()
    p.layers.set_homogeneous(
        top=p.ysize, bottom=2.0 * p.ysize, a=p.a0, b=p.b0
    )
    # FastSlipPy.__init__ normally performs this mapping.  This standalone
    # static diagnostic constructs MatrixBuilder directly and must do it once.
    p.apply_bp3_motion_sign()
    return p


def load_checkpoint(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: np.asarray(data[name]).copy() for name in data.files}


def checkpoint_grid_shape(checkpoint: dict[str, np.ndarray]) -> tuple[int, int]:
    if "U" not in checkpoint or "ux" not in checkpoint or "uy" not in checkpoint:
        raise ValueError("Checkpoint must contain U, ux, and uy.")
    ny = int(np.asarray(checkpoint["U"]).size)
    ux_shape = np.asarray(checkpoint["ux"]).shape
    uy_shape = np.asarray(checkpoint["uy"]).shape
    if len(ux_shape) != 2 or len(uy_shape) != 2:
        raise ValueError("Checkpoint ux and uy must be two-dimensional.")
    nx = int(ux_shape[1])
    if ux_shape != (ny + 1, nx) or uy_shape != (ny, nx + 1):
        raise ValueError(
            f"Inconsistent checkpoint shapes: U={(ny,)}, ux={ux_shape}, "
            f"uy={uy_shape}."
        )
    return nx, ny


def gaussian(y: np.ndarray, centre: float, width: float) -> np.ndarray:
    values = np.exp(-0.5 * ((y - centre) / width) ** 2)
    return values / np.max(np.abs(values))


def diagnostic_modes(y: np.ndarray, w_f: float) -> tuple[list[str], np.ndarray]:
    """Modes that separate nucleation-band, endpoint, and long-wave behavior."""

    labels = [
        "surface",
        "depth_5km",
        "depth_10km",
        "depth_11km",
        "depth_15km",
        "wf_minus_1km",
        "rs_long_wave",
        "bottom_artificial",
    ]
    columns = [
        gaussian(y, 0.0, 0.75e3),
        gaussian(y, 5e3, 0.75e3),
        gaussian(y, 10e3, 0.75e3),
        gaussian(y, 11e3, 0.75e3),
        gaussian(y, 15e3, 0.75e3),
        gaussian(y, w_f - 1e3, 0.75e3),
        np.sin(np.pi * np.minimum(y, w_f) / w_f),
        gaussian(y, y[-1], 2e3),
    ]
    # Physical perturbations vanish on the prescribed creeping segment.  The
    # final artificial mode is retained solely to expose the bottom corner.
    for index in range(len(columns) - 1):
        columns[index] = columns[index].copy()
        columns[index][y >= w_f] = 0.0
    modes = np.column_stack(columns)
    modes /= np.max(np.abs(modes), axis=0)
    return labels, modes


def reciprocity_payload(response: FaultModeResponse) -> dict:
    result = diagnose_fault_reciprocity(response)
    return {
        "antisymmetric_fraction": result.antisymmetric_fraction,
        "maximum_pairwise_fraction": result.maximum_pairwise_fraction,
        "work_matrix": result.work_matrix.tolist(),
    }


def slice_response(response: FaultModeResponse, columns: slice) -> FaultModeResponse:
    return FaultModeResponse(
        y=response.y,
        modes=response.modes[:, columns],
        tau=response.tau[:, columns],
        sigma_effective=response.sigma_effective[:, columns],
    )


def analyze_saved_mode_file(path: Path) -> dict:
    with np.load(path) as data:
        required = {"y", "spatial_modes", "tau_responses", "sigma_effective_responses"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing fields: {sorted(missing)}")
        response = FaultModeResponse(
            y=np.asarray(data["y"], dtype=float),
            modes=np.asarray(data["spatial_modes"], dtype=float),
            tau=np.asarray(data["tau_responses"], dtype=float),
            sigma_effective=np.asarray(
                data["sigma_effective_responses"], dtype=float
            ),
        )
    return reciprocity_payload(response)


def station_differences(result, depths_km: tuple[float, ...]) -> dict:
    if result.checkpoint.projected_solution is None:
        return {}
    y = result.checkpoint.y
    tau_delta = (
        result.checkpoint.fault_tau_projected
        - result.checkpoint.fault_tau_stored
    )
    sigma_delta = (
        result.checkpoint.fault_sigma_effective_projected
        - result.checkpoint.fault_sigma_effective_stored
    )
    rows = {}
    for depth in depths_km:
        index = int(np.argmin(np.abs(y - depth * 1e3)))
        rows[f"{depth:g}km"] = {
            "grid_depth_km": float(y[index] / 1e3),
            "delta_tau_pa": float(tau_delta[index]),
            "delta_sigma_effective_pa": float(sigma_delta[index]),
        }
    return rows


def checkpoint_recovery_consistency(
    checkpoint: dict[str, np.ndarray], result, params: ModelParameters
) -> dict:
    """Compare saved fault stresses with recovery from saved displacement."""

    comparisons = {}
    if "tauqs" in checkpoint:
        saved_tau = np.asarray(checkpoint["tauqs"], dtype=float)[
            :, params.Nx // 2
        ]
        recovered_tau = result.checkpoint.fault_tau_stored
        difference = recovered_tau - saved_tau
        comparisons["tauqs"] = {
            "relative_l2": float(
                np.linalg.norm(difference)
                / max(np.linalg.norm(saved_tau), np.finfo(float).tiny)
            ),
            "maximum_absolute_pa": float(np.max(np.abs(difference))),
        }
    if "sigma" in checkpoint:
        saved_sigma = np.asarray(checkpoint["sigma"], dtype=float)
        recovered_sigma = (
            params.sigma0
            + result.checkpoint.fault_sigma_effective_stored
        )
        difference = recovered_sigma - saved_sigma
        comparisons["effective_normal_stress"] = {
            "relative_l2": float(
                np.linalg.norm(difference)
                / max(np.linalg.norm(saved_sigma), np.finfo(float).tiny)
            ),
            "maximum_absolute_pa": float(np.max(np.abs(difference))),
        }
    return comparisons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint", nargs="?", type=Path,
        help="Checkpoint to diagnose; omit for --mode-files-only analysis.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/bp3_operator_consistency"),
    )
    parser.add_argument("--mode-files", nargs="*", type=Path, default=())
    parser.add_argument("--xsize-km", type=float, default=320.0)
    parser.add_argument("--ysize-km", type=float, default=160.0)
    parser.add_argument("--alpha", type=float, default=60.0)
    parser.add_argument("--motion-sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--x-inner-km", type=float, default=20.0)
    parser.add_argument("--y-inner-km", type=float, default=20.0)
    parser.add_argument("--x-inner-points", type=int, default=401)
    parser.add_argument("--y-inner-points", type=int, default=401)
    parser.add_argument(
        "--residual-only", action="store_true",
        help="Assemble A and report A*u-b without LU projection or mode probes.",
    )
    parser.add_argument(
        "--no-probe-modes", action="store_true",
        help="Project the checkpoint but skip new endpoint/localized mode solves.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved_modes = {
        str(path.resolve()): analyze_saved_mode_file(path.resolve())
        for path in args.mode_files
    }
    if args.checkpoint is None:
        if not saved_modes:
            raise ValueError(
                "Supply a checkpoint, at least one --mode-files path, or both."
            )
        payload = {
            "source_checkpoint": None,
            "saved_interior_mode_files": saved_modes,
            "interpretation": (
                "Pure shear-traction reciprocity on the span of each saved "
                "nucleation-mode set; no new elastic solve was performed."
            ),
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        lines = ["# BP3 saved-mode shear reciprocity", ""]
        for path, values in saved_modes.items():
            lines.append(
                f"- `{path}`: antisymmetric fraction "
                f"`{values['antisymmetric_fraction']:.6e}`, maximum pairwise "
                f"fraction `{values['maximum_pairwise_fraction']:.6e}`."
            )
        (args.output_dir / "summary.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        print(f"Saved: {args.output_dir.resolve()}")
        return

    source = args.checkpoint.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    checkpoint = load_checkpoint(source)
    nx, ny = checkpoint_grid_shape(checkpoint)
    if args.x_inner_points > nx or args.y_inner_points > ny:
        raise ValueError("Inner-point counts cannot exceed checkpoint Nx/Ny.")

    params = build_parameters(
        nx=nx,
        ny=ny,
        xsize=args.xsize_km * 1e3,
        ysize=args.ysize_km * 1e3,
        alpha=args.alpha,
        motion_sign=args.motion_sign,
        x_inner_size=args.x_inner_km * 1e3,
        y_inner_size=args.y_inner_km * 1e3,
        x_inner_points=args.x_inner_points,
        y_inner_points=args.y_inner_points,
    )
    # Generate modes on the exact grid without independently constructing it:
    # the checkpoint fault grid and configured stretching have the same Ny.
    from fastslippy.pre_processing.grid import Grid

    y = Grid(params).y
    labels, modes = diagnostic_modes(y, params.W_f)
    use_modes = None if (args.residual_only or args.no_probe_modes) else modes
    result = diagnose_bp3_operator_consistency(
        params,
        checkpoint,
        modes=use_modes,
        project=not args.residual_only,
    )

    residuals = {
        name: asdict(summary)
        for name, summary in result.checkpoint.residual_groups.items()
    }
    payload = {
        "source_checkpoint": str(source),
        "checkpoint_time_seconds": float(np.asarray(checkpoint["t"]).item()),
        "checkpoint_time_years": float(np.asarray(checkpoint["t"]).item()) / YEAR,
        "configuration": {
            "alpha": params.alpha,
            "motion_sign": params.motion_sign,
            "xsize_km": params.xsize / 1e3,
            "ysize_km": params.ysize / 1e3,
            "nx": params.Nx,
            "ny": params.Ny,
        },
        "equilibrium_residuals": residuals,
        "checkpoint_recovery_consistency": checkpoint_recovery_consistency(
            checkpoint, result, params
        ),
        "projection": {
            "performed": result.checkpoint.projected_solution is not None,
            "displacement_relative_l2": result.checkpoint.displacement_relative_l2,
            "station_differences": station_differences(
                result, (5.0, 10.0, 11.0, 15.0)
            ),
        },
        "new_probe_modes": None,
        "saved_interior_mode_files": saved_modes,
        "interpretation": {
            "shear_reciprocity": (
                "The pure shear-traction work matrix should be symmetric. "
                "Coulomb traction is intentionally not used for this test."
            ),
            "bottom_artificial": (
                "The bottom mode perturbs the prescribed creeping segment and "
                "is a numerical corner probe, not an admissible BP3 evolution mode."
            ),
        },
    }
    if result.mode_response is not None:
        payload["new_probe_modes"] = {
            "labels": labels,
            "all": reciprocity_payload(result.mode_response),
            "physical_rs": reciprocity_payload(
                slice_response(result.mode_response, slice(0, -1))
            ),
            "nucleation_10_11_15km": reciprocity_payload(
                FaultModeResponse(
                    y=result.mode_response.y,
                    modes=result.mode_response.modes[:, 2:5],
                    tau=result.mode_response.tau[:, 2:5],
                    sigma_effective=result.mode_response.sigma_effective[:, 2:5],
                )
            ),
        }

    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    npz_payload = {
        "y": result.checkpoint.y,
        "fault_tau_stored": result.checkpoint.fault_tau_stored,
        "fault_sigma_effective_stored": (
            result.checkpoint.fault_sigma_effective_stored
        ),
    }
    if result.checkpoint.fault_tau_projected is not None:
        npz_payload.update(
            fault_tau_projected=result.checkpoint.fault_tau_projected,
            fault_sigma_effective_projected=(
                result.checkpoint.fault_sigma_effective_projected
            ),
        )
    if result.mode_response is not None:
        npz_payload.update(
            probe_modes=result.mode_response.modes,
            probe_tau=result.mode_response.tau,
            probe_sigma_effective=result.mode_response.sigma_effective,
        )
    np.savez_compressed(args.output_dir / "diagnostic_profiles.npz", **npz_payload)

    physical = residuals["physical_rows"]
    lines = [
        "# BP3 operator consistency diagnostic",
        "",
        f"Checkpoint: `{source}`",
        f"Time: `{payload['checkpoint_time_years']:.9f} yr`",
        "",
        "## Stored-state equilibrium residual",
        "",
        "| row group | relative L2 | max row-relative | max absolute |",
        "|---|---:|---:|---:|",
    ]
    for name, values in residuals.items():
        lines.append(
            f"| {name} | {values['relative_l2']:.6e} | "
            f"{values['maximum_rowwise_relative']:.6e} | "
            f"{values['maximum_absolute']:.6e} |"
        )
    lines.extend([
        "",
        "## Projection",
        "",
        f"Performed: `{payload['projection']['performed']}`",
        (
            "Stored/projected displacement relative L2: "
            f"`{payload['projection']['displacement_relative_l2']}`"
        ),
        "",
        "## Reciprocity",
        "",
    ])
    if result.reciprocity is not None:
        lines.append(
            "New physical RS probes: antisymmetric fraction "
            "`{:.6e}`.".format(
                payload["new_probe_modes"]["physical_rs"][
                    "antisymmetric_fraction"
                ]
            )
        )
        lines.append(
            "10/11/15 km probes: antisymmetric fraction "
            "`{:.6e}`.".format(
                payload["new_probe_modes"]["nucleation_10_11_15km"][
                    "antisymmetric_fraction"
                ]
            )
        )
    for path, values in saved_modes.items():
        lines.append(
            f"Saved modes `{path}`: antisymmetric fraction "
            f"`{values['antisymmetric_fraction']:.6e}`."
        )
    lines.extend([
        "",
        (
            "Raw matrices and station-wise projection differences are in "
            "`summary.json`; fault profiles are in `diagnostic_profiles.npz`."
        ),
    ])
    (args.output_dir / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        "Physical-row equilibrium residual: "
        f"{physical['relative_l2']:.6e}"
    )
    print(f"Saved: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
