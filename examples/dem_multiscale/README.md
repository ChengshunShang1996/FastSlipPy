# DEM multiscale example

This directory is a self-contained Kratos DEM RVE case used by
`run_case_dem_multiscale.py`. It includes the particle geometry, empty FEM boundary model,
material law and Kratos project parameters expected by the FastSlipPy DEM
backend.

Run from the FastSlipPy repository root:

```powershell
$env:OMP_NUM_THREADS = '1'
python examples/dem_multiscale/run_case_dem_multiscale.py --verify --output output/full-cycle-verified
```

The coupled example generates the small 16-cylinder packing and preloads one
RVE per constitutive node at 10 kPa normal stress and 500 Pa driving shear.
The four loading phases are initial hold, loading, hold and reload. With
`--verify` the example repeats the run with different output and checkpoint
intervals and compares the trajectories. The larger geometry in this directory
can be regenerated with `prepare_case.py`.

In `dem_multiscale` mode each constitutive node advances every Euler step.
The legacy activation and deactivation settings are ignored. Prescribed deep
creep and configured extrapolated surface rates retain their boundary roles.
Set `dem.initial_driving_shear_stress` in Pa to a scalar or a nodal profile.
Groningen defaults to its resolved initial shear, California DEM defaults to
zero, and the laboratory case defaults to its original near-zero shear.
The RVE supplies the slip increment and its interval mean drives the elastic
solve. In `dem_history.csv`, interval means and instantaneous end values are
separate. FDM fault V and tau outputs are end values. The FDM effective normal
stress and the measured DEM normal reaction are separate quantities. Legacy
NumPy theta arrays contain NaN for DEM and DEM VTK omits the RSF state field.

The numerical values are a demonstration setup. Calibrate the contact law,
virtual wall mass, stress scale and coupling time-step limits before using it
as a physical fault-gouge model.

`dem_history.csv` also separates macro wall kinematics from deformation inside
the free grains. `dem_slip_m` is the top-wall displacement used by the coupled
energy balance. `internal_slip_m` is the mean horizontal displacement of the
upper half of the free grains minus that of the lower half, both measured from
the accepted loading handoff. Wall reactions and volume-weighted free-grain
stress are reported independently. Contact diagnostics give the largest local
`|Ft|/|Fn|`, the material Coulomb limit, and the fraction of active contacts at
that limit. `normal_servo_saturated` identifies samples for which increasing
`normal_servo_velocity` or changing the loading rate should be considered.
The central-node comparison is plotted in `dem_diagnostics.png`.

## Full-cycle time-step sensitivity

For a full-cycle time-step sensitivity study, run the following from the
repository root.

```powershell
$env:OMP_NUM_THREADS = '1'
python examples/dem_multiscale/verify_time_steps.py
```

The study keeps the 10 MPa grain modulus, 5-node fault, particle packing and
0.3 ms physical loading phases of the coupled example. It prepares every RVE
at the original 5 µs DEM step, then refines the production DEM step twice at
fixed 25 µs Euler intervals. It refines the Euler interval twice at fixed
1.25 µs DEM steps. The normal servo retains its physical relaxation time
through the change of sampling interval. Each DEM step checks the discrete
wall momentum equation using the applied drive, previous contact traction and
previous wall velocity, matching the explicit wall update.

Under `output/time-step-baseline`, `responses.png` plots the central node;
`phases.csv` gives the largest motion and momentum residual over all nodes per
phase. `sensitivity.csv` compares all five nodes at common 25 µs times. Its
relative RMS denominator is the RMS variation of the finer solution from its
own handoff state. Layer height is the mean wall-center separation.

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
