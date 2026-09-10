r"""Short Euler/RK2 convergence runs restarted before the first BP3 event.

The default configuration reproduces the 320 km x 160 km, 50 m-core,
60-degree normal-fault case used in the current FastSlipPy investigation.  A
checkpoint is copied into an isolated output directory for every requested
integrator/ksi combination, so the source calculation is never modified.

Example (PowerShell)::

    python examples/run_bp3_checkpoint_integrator_convergence.py `
      "E:\...\x-small-...-60degree\output\data_6000.npz" `
      --integrators euler rk2_midpoint --ksi-scales 1 0.5 `
      --final-year 195
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from fastslippy import FastSlipPy
from fastslippy.pre_processing.model_parameters import ModelParameters


YEAR = 365.0 * 24.0 * 3600.0


def build_parameters(
    *, time_integrator: str, ksi_scale: float, final_year: float,
    max_steps: int, output_interval: int, checkpoint_interval: int,
) -> ModelParameters:
    """Build the production 320 x 160 km, 60-degree BP3 configuration."""
    p = ModelParameters(
        case_type="california",
        alpha=60.0,
        motion_sign=-1,
        auto_motion_sign=True,
        xsize=320e3,
        ysize=160e3,
        Nx=901,
        Ny=651,
        Nt=max_steps,
        output_interval=output_interval,
        checkpoint_interval=checkpoint_interval,
        rho=2670.0,
        cs=3464.0,
        mu0=0.6,
        nu=0.25,
        V0=1e-6,
        a0=0.01,
        a_max=0.025,
        b0=0.015,
        L=0.008,
        dt_init=1.0,
        dt_max=1e6,
        dt_growth=1.2,
        ksi_scale=ksi_scale,
        tfinal=final_year * YEAR,
        friction_tolerance=5.0,
        slip_rate_solver="newton_v2",
        time_integrator=time_integrator,
        output_vtk_option=False,
        Vi=1e-9,
        flash_heating_option=False,
        extrapolate_surface_fault_rate=False,
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
        fallback_to_iterative_on_oom=False,
    )
    p.loading.tload = 0.0
    p.loading.dPdt_pre = 0.0
    p.loading.dPdt_post = 0.0
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
    return p


def checkpoint_number(path: Path) -> int:
    try:
        return int(path.stem.removeprefix("data_"))
    except ValueError as exc:
        raise ValueError(
            f"Checkpoint must be named data_<iteration>.npz, got {path.name!r}."
        ) from exc


def validate_checkpoint(path: Path):
    required = {
        "U", "V", "tau", "sigma", "theta", "tauqs", "sigmaqs",
        "uy", "vy", "ux", "vx", "dt", "t",
    }
    with np.load(path) as data:
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Checkpoint is missing fields: {sorted(missing)}")
        if data["U"].shape != (651,) or data["uy"].shape != (651, 902):
            raise ValueError(
                "This entry point expects the 320 x 160 km production grid "
                "(Nx=901, Ny=651)."
            )
        return float(data["t"])


def final_checkpoint(run_dir: Path, restart_index: int) -> Path:
    candidates = sorted(
        run_dir.glob("data_*.npz"),
        key=lambda path: checkpoint_number(path),
    )
    candidates = [
        path for path in candidates if checkpoint_number(path) > restart_index
    ]
    if not candidates:
        raise RuntimeError(f"No continuation checkpoint was written in {run_dir}.")
    return candidates[-1]


def summarize_run(run_dir: Path, restart_index: int, threshold: float):
    checkpoint_path = final_checkpoint(run_dir, restart_index)
    with np.load(checkpoint_path) as checkpoint:
        final = {
            name: np.array(checkpoint[name])
            for name in ("U", "V", "tau", "sigma", "theta")
        }
        final_time = float(checkpoint["t"])

    with np.load(run_dir / "dataall.npz") as history:
        times = np.asarray(history["tm"])
        velocities = np.asarray(history["Vm"])
        valid = times > 0.0
        times = times[valid]
        velocities = velocities[:, valid]
        max_velocity = (
            np.max(np.abs(velocities), axis=0)
            if times.size else np.empty(0)
        )
        crossings = np.flatnonzero(max_velocity >= threshold)
        onset = float(times[crossings[0]]) if crossings.size else None
        peak = float(np.max(max_velocity)) if max_velocity.size else None

    return {
        "checkpoint": str(checkpoint_path),
        "final_time_year": final_time / YEAR,
        "event_onset_year": None if onset is None else onset / YEAR,
        "peak_abs_velocity": peak,
        "final": final,
    }


def write_summary(output_root: Path, results: list[dict], threshold: float):
    reference = min(
        results,
        key=lambda item: (
            item["integrator"] != "rk2_midpoint", item["ksi_scale"]
        ),
    )
    depths_km = (5.0, 10.0, 11.0, 15.0)
    dy = 50.0
    rows = []
    serializable = []
    for result in results:
        metrics = {}
        for field in ("U", "V", "tau", "sigma", "theta"):
            delta = result["final"][field] - reference["final"][field]
            metrics[f"{field}_relative_l2"] = float(
                np.linalg.norm(delta)
                / max(np.linalg.norm(reference["final"][field]), 1e-300)
            )
        metrics["station_delta"] = {
            f"{depth:g}km": {
                field: float(
                    result["final"][field][int(round(depth * 1000.0 / dy))]
                    - reference["final"][field][int(round(depth * 1000.0 / dy))]
                )
                for field in ("U", "V", "tau", "sigma", "theta")
            }
            for depth in depths_km
        }
        rows.append(
            "| {integrator} | {ksi:g} | {final:.9f} | {onset} | {peak:.6e} | "
            "{du:.3e} | {ds:.3e} |".format(
                integrator=result["integrator"],
                ksi=result["ksi_scale"],
                final=result["final_time_year"],
                onset=(
                    "n/a" if result["event_onset_year"] is None
                    else f"{result['event_onset_year']:.9f}"
                ),
                peak=(
                    float("nan") if result["peak_abs_velocity"] is None
                    else result["peak_abs_velocity"]
                ),
                du=metrics["U_relative_l2"],
                ds=metrics["sigma_relative_l2"],
            )
        )
        serializable.append(
            {
                key: value for key, value in result.items() if key != "final"
            } | {"errors_against_finest_rk2": metrics}
        )

    header = [
        "# BP3 checkpoint integrator convergence",
        "",
        f"Event threshold: `{threshold:.3e} m/s`.",
        (
            "Reference for reported errors: "
            f"`{reference['integrator']}`, ksi={reference['ksi_scale']:g}."
        ),
        "",
        "| integrator | ksi scale | final year | onset year | peak | rel L2 U | rel L2 sigma |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "Station-wise deltas at 5, 10, 11, and 15 km are in `summary.json`.",
    ]
    (output_root / "summary.md").write_text(
        "\n".join(header) + "\n", encoding="utf-8"
    )
    (output_root / "summary.json").write_text(
        json.dumps(serializable, indent=2), encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("artifacts/bp3_checkpoint_integrator_convergence"),
    )
    parser.add_argument(
        "--integrators", nargs="+", choices=("euler", "rk2_midpoint"),
        default=("euler", "rk2_midpoint"),
    )
    parser.add_argument(
        "--ksi-scales", nargs="+", type=float, default=(1.0, 0.5),
    )
    parser.add_argument("--final-year", type=float, default=195.0)
    parser.add_argument("--max-steps", type=int, default=40000)
    parser.add_argument("--output-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=5000)
    parser.add_argument("--event-threshold", type=float, default=1e-5)
    return parser.parse_args()


def main():
    args = parse_args()
    source = args.checkpoint.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    restart_index = checkpoint_number(source)
    restart_time = validate_checkpoint(source)
    if args.final_year * YEAR <= restart_time:
        raise ValueError(
            f"final-year must exceed checkpoint time {restart_time / YEAR:.9f} yr."
        )
    if any(scale <= 0.0 for scale in args.ksi_scales):
        raise ValueError("Every ksi scale must be positive.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for integrator in args.integrators:
        for ksi_scale in args.ksi_scales:
            label = f"{integrator}_ksi_{ksi_scale:g}".replace(".", "p")
            run_dir = args.output_root / label
            run_dir.mkdir(parents=True, exist_ok=True)
            restart_copy = run_dir / source.name
            stale = [
                path for path in run_dir.iterdir()
                if path.name != source.name
            ]
            if stale:
                raise FileExistsError(
                    f"{run_dir} already contains continuation results; "
                    "choose a new --output-root to avoid mixing runs."
                )
            if not restart_copy.exists():
                shutil.copy2(source, restart_copy)

            params = build_parameters(
                time_integrator=integrator,
                ksi_scale=ksi_scale,
                final_year=args.final_year,
                max_steps=args.max_steps,
                output_interval=args.output_interval,
                checkpoint_interval=args.checkpoint_interval,
            )
            model = FastSlipPy(
                params=params,
                output_dir=str(run_dir),
                checkpointer=restart_index,
            )
            model.figure_creator.plot_results = lambda *args, **kwargs: None
            model.run()
            result = summarize_run(
                run_dir, restart_index, args.event_threshold
            )
            result.update(integrator=integrator, ksi_scale=ksi_scale)
            results.append(result)

    write_summary(args.output_root, results, args.event_threshold)
    print(f"Summary: {args.output_root / 'summary.md'}")


if __name__ == "__main__":
    main()
