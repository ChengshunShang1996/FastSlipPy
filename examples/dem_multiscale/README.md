# DEM multiscale example

This directory is a self-contained Kratos DEM RVE case used by
`run_case_dem_multiscale.py`. It includes the particle geometry, empty FEM boundary model,
material law and Kratos project parameters expected by the FastSlipPy DEM
backend.

Run from the FastSlipPy repository root:

```powershell
python examples/dem_multiscale/run_case_dem_multiscale.py
```

Use `prepare_case.py` to regenerate the deterministic particle geometry after
changing the RVE dimensions or particle distribution. The committed `.mdpa`
files allow the example to run without a generation step.

The numerical values are a demonstration setup. Calibrate the contact law,
virtual wall mass, stress scale and coupling time-step limits before using it
as a physical fault-gouge model.

## Single-RVE wall and time-step check

Run from the repository root with Kratos DEMApplication available.

```powershell
$env:OMP_NUM_THREADS = '1'
python examples/dem_multiscale/verify_single_rve.py
```

This displacement-controlled experiment uses eight free cylinders between two
rough walls of four cylinders each, a 24 mm periodic width and the materials
in `MaterialsDEM.json`. The bottom wall stays fixed. The top wall moves down
30 µm, holds, then moves right 5.025 µm at constant height and holds again.
The prescribed hold intervals let the reactions settle for measurement.
Target-stress initialization is a separate experiment.

The script checks both wall reactions over the last 100 steps of each hold
(relative tolerance 0.1%, absolute tolerance 1 µN), nodal velocities and
coordinate changes, and every Kratos time increment. The shear interval takes
100 steps of 5 µs and one of 2.5 µs. For unit out-of-plane thickness,
compression-positive normal stress is `top_fy_N / 0.024` and positive shear
resistance is `-top_fx_N / 0.024`.

The per-substep trace is written to `output/single-rve/history.csv`. Use
`--output PATH` for another run directory. The trace reports wall-center
separation as `height_m`, actual wall displacements and velocities, forces
on both walls, tractions, Kratos time and the integration errors.
