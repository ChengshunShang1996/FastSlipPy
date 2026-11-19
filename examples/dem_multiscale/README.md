# DEM multiscale example

This directory is a self-contained Kratos DEM RVE case used by
`run_case.py`. It includes the particle geometry, empty FEM boundary model,
material law and Kratos project parameters expected by the FastSlipPy DEM
backend.

Run from the FastSlipPy repository root:

```powershell
python examples/dem_multiscale/run_case.py
```

Use `prepare_case.py` to regenerate the deterministic particle geometry after
changing the RVE dimensions or particle distribution. The committed `.mdpa`
files allow the example to run without a generation step.

The numerical values are a demonstration setup. Calibrate the contact law,
virtual wall mass, stress scale and coupling time-step limits before using it
as a physical fault-gouge model.
