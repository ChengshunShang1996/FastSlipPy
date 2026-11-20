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

## Target-controlled normal consolidation

```powershell
$env:OMP_NUM_THREADS = '1'
python examples/dem_multiscale/verify_normal_consolidation.py
```

The command creates separate 5 kPa, 15 kPa and step-limit runs of the same
16-cylinder packing. Each RVE writes `normal_consolidation.csv` under
`output/normal-consolidation/<case>/rve_00000/`. The final row records the
stop reason. The step-limit run must raise a consolidation error and leave
its diagnostic history. Use `--output PATH` to change the parent directory.

The coupled backend prepares each new RVE at its node's initial effective
normal stress. The bottom wall is fixed and the top wall can move vertically
in either direction. At every DEM substep both compression-positive wall
tractions must match the target within
`consolidation_stress_atol + consolidation_stress_rtol * target` (defaults
1 Pa and 0.001). The in-plane norm of the sum of both reactions per wall
area must satisfy the same bound. The largest wall normal speed, free-grain
translation speed and free-grain rim speed (radius times angular speed)
must each be at most `consolidation_speed_tol` (default 1e-5 m/s).

All checks must pass for `consolidation_window_steps` consecutive substeps
(default 100). A violation resets the count. `consolidation_max_steps`
(default 20000) bounds a failed attempt. The backend accepts the legacy
`consolidation_time` argument for compatibility and ignores its value.

## Single-RVE shear preload and handoff

```powershell
$env:OMP_NUM_THREADS = '1'
python examples/dem_multiscale/verify_shear_preload.py
```

Supply `initial_driving_shear_stress` and `radiation_damping` when constructing
a `KratosDEMRVE` to request a near-static shear preload after normal
consolidation. Zero is a valid shear target. The driver keeps the normal
servo active and ramps the shear drive smoothly from the measured initial
resistance to the target over `shear_preload_ramp_steps` (default 200).
The equilibrium window starts once the full drive is applied.
Upper and lower wall contact tractions must match the drive minus ηV within
the configured absolute-plus-relative shear tolerance. The damping contribution
ηV itself must fit that tolerance, and all wall and grain speeds and normal
equilibrium checks must pass. The shear phase has its own step count and uses
the same window and maximum steps as normal consolidation.

The same stage continues into loading with its accepted wall velocity and
contact forces. `loading_time` and `loading_slip` measure from the accepted
handoff. The original time and displacement are recorded in
`loading_time_origin` and `loading_slip_origin`. To check continuity, the
first call to `advance` uses the same drive, normal stress and damping.

The verification uses 10 kPa normal stress, drives of 0 and ±1 kPa, a wall
mass of 50 kg/m² and η = 50,000 Pa·s/m. Each accepted RVE is held for 200
production substeps. A separate 20 kPa run exercises the failure path.
At the step limit, `sustained_sliding` records same-sign wall motion above the
speed tolerance over the last full window at the target drive.
Under `output/shear-preload`, each case has a per-step `shear_preload.csv`.
Accepted cases also have `handoff.csv`, whose first row is the accepted
state at loading time zero. `handoff.png` compares the three accepted cases.
