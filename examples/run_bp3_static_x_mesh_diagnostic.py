"""Separate BP3 x-domain and stretched-outer-mesh effects without cycling.

The diagnostic reads a fault slip-rate profile from ``dataall.npz`` and solves
one instantaneous 60-degree BP3 elasticity problem for each x mesh:

* baseline:       current full width and outer spacing;
* wider_domain:   farther x boundaries with comparable maximum outer spacing;
* refined_outer:  original boundaries with a smaller maximum outer spacing.

The y domain and y grid remain identical, so the two comparisons isolate the
x-boundary location and x-outer-grid resolution respectively.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from fastslippy.pre_processing.grid import Grid
from fastslippy.pre_processing.model_parameters import ModelParameters
from fastslippy.utilities.bp3_small_peak import solve_fault_loading_response


YEAR = 365.25 * 24.0 * 3600.0


@dataclass(frozen=True)
class XMeshCase:
    name: str
    xsize: float
    nx: int


CASES = (
    XMeshCase("baseline", 320e3, 901),
    # Paper Lx increases from 160 to 200 km.  Nx keeps max(dx) near 1.2 km.
    XMeshCase("wider_domain", 400e3, 1009),
    # Same Lx=160 km as baseline, but max(dx) is reduced to about 0.6 km.
    XMeshCase("refined_outer", 320e3, 1325),
)


def build_parameters(case: XMeshCase) -> ModelParameters:
    """Return the production 60-degree BP3 setup for one x mesh."""

    params = ModelParameters(
        case_type="california",
        alpha=60.0,
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
        motion_sign=-1,
        auto_motion_sign=True,
    )
    params.loading.V_p = 1e-9
    params.loading.V_L = 1e-9
    params.bc.left.ux.set_fixed()
    params.bc.left.uy.set_velocity(-0.5e-9)
    params.bc.right.ux.set_fixed()
    params.bc.right.uy.set_velocity(0.5e-9)
    params.bc.top.set_traction_free()
    params.bc.bottom.set_traction_free()
    return params


def load_fault_profile(path: Path, target_year: float) -> tuple[np.ndarray, float]:
    """Load the nearest stored production-grid fault-rate profile."""

    with np.load(path) as data:
        valid = np.flatnonzero(data["tm"] > 0.0)
        if valid.size == 0:
            raise ValueError(f"{path} contains no valid output snapshots")
        times = data["tm"][valid]
        local = int(np.argmin(np.abs(times / YEAR - target_year)))
        index = int(valid[local])
        rate = np.asarray(data["Vm"][:, index], dtype=float).copy()
        actual_year = float(data["tm"][index] / YEAR)
    if rate.shape != (651,):
        raise ValueError(
            "The default diagnostic expects the 651-node production y grid; "
            f"got {rate.shape}."
        )
    return rate, actual_year


def sample_profile(y: np.ndarray, values: np.ndarray, depth: float) -> float:
    return float(np.interp(depth, y, values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataall", type=Path, help="60-degree dataall.npz")
    parser.add_argument("--profile-year", type=float, default=150.0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/bp3_static_x_mesh")
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=[case.name for case in CASES],
        default=[case.name for case in CASES],
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse previously saved case profiles and only rebuild summaries.",
    )
    args = parser.parse_args()

    fault_rate, actual_year = load_fault_profile(args.dataall, args.profile_year)
    selected = [case for case in CASES if case.name in args.cases]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    depths = (10e3, 11.9e3, 15e3)
    mu_reference = 0.6
    results = {}

    print(
        f"Loaded V(y) at {actual_year:.6f} yr from {args.dataall}; "
        f"running {len(selected)} static cases."
    )
    for case in selected:
        started = time.perf_counter()
        params = build_parameters(case)
        grid = Grid(params)
        print(
            f"[{case.name}] Lx={case.xsize / 2e3:.1f} km, Nx={case.nx}, "
            f"max(dx)={np.max(grid.dx_edges):.3f} m"
        )
        profile_path = args.output_dir / f"{case.name}_profiles.npz"
        if args.reuse_existing and profile_path.exists():
            with np.load(profile_path) as saved:
                response_y = np.asarray(saved["y"], dtype=float).copy()
                tau_rate = np.asarray(saved["tau_rate"], dtype=float).copy()
                sigma_effective_rate = np.asarray(
                    saved["sigma_effective_rate"], dtype=float
                ).copy()
            print(f"[{case.name}] reused {profile_path}")
        else:
            response = solve_fault_loading_response(params, fault_rate)
            response_y = response.y
            tau_rate = response.tau_rate
            sigma_effective_rate = response.sigma_effective_rate
        coulomb_rate = (
            tau_rate - mu_reference * sigma_effective_rate
        )
        samples = []
        for depth in depths:
            samples.append(
                {
                    "depth_km": depth / 1e3,
                    "tau_rate_mpa_per_year": sample_profile(
                        response_y, tau_rate, depth
                    ) * YEAR / 1e6,
                    "sigma_effective_rate_mpa_per_year": sample_profile(
                        response_y, sigma_effective_rate, depth
                    ) * YEAR / 1e6,
                    "coulomb_rate_mpa_per_year": sample_profile(
                        response_y, coulomb_rate, depth
                    ) * YEAR / 1e6,
                }
            )
        results[case.name] = {
            "case": asdict(case),
            "paper_Lx_km": case.xsize / 2e3,
            "max_dx_m": float(np.max(grid.dx_edges)),
            "max_adjacent_dx_ratio": float(
                max(
                    np.max(grid.dx_edges[1:] / grid.dx_edges[:-1]),
                    np.max(grid.dx_edges[:-1] / grid.dx_edges[1:]),
                )
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "samples": samples,
        }
        if not (args.reuse_existing and profile_path.exists()):
            np.savez_compressed(
                profile_path,
                y=response_y,
                tau_rate=tau_rate,
                sigma_effective_rate=sigma_effective_rate,
                coulomb_rate=coulomb_rate,
                fault_rate=fault_rate,
            )
        print(f"[{case.name}] completed in {results[case.name]['elapsed_seconds']:.1f} s")
        if "response" in locals():
            del response
        del response_y, tau_rate, sigma_effective_rate, coulomb_rate, grid, params
        gc.collect()

    baseline = results.get("baseline")
    rows = []
    for case_name, result in results.items():
        for sample_index, sample in enumerate(result["samples"]):
            row = {"case": case_name, **sample}
            if baseline is not None:
                reference = baseline["samples"][sample_index]
                for field in (
                    "tau_rate_mpa_per_year",
                    "sigma_effective_rate_mpa_per_year",
                    "coulomb_rate_mpa_per_year",
                ):
                    denominator = max(abs(reference[field]), 1e-30)
                    row[f"{field}_relative_to_baseline"] = (
                        sample[field] - reference[field]
                    ) / denominator
            rows.append(row)

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "source": str(args.dataall),
                "requested_profile_year": args.profile_year,
                "actual_profile_year": actual_year,
                "mu_reference": mu_reference,
                "results": results,
            },
            stream,
            indent=2,
        )
    if rows:
        with (args.output_dir / "samples.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    print(f"Saved diagnostic output to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
