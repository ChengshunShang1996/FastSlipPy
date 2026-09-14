# BP3 research tools

These runners document and reproduce the numerical investigations used while
developing the BP3 implementation. They are repository tools, not supported
quick-start examples and not part of the installed `fastslippy` API.

- `diagnostics/` contains focused static checks of fault tractions, interface
  transfer, corner behaviour, operator consistency, and compact transients.
- `studies/` contains convergence, checkpoint, mesh, nucleation, and frozen
  operator experiments. Several configurations require existing NPZ histories
  or checkpoints and production HPC memory.
- `plotting/` contains BP3/DFRA-specific reporting and animation runners.

Prefer module execution from the repository root so imports are unambiguous:

```console
python -m tools.bp3.diagnostics.run_bp3_fault_traction_audit --help
python -m tools.bp3.studies.run_bp3_x_core_convergence --help
python -m tools.bp3.plotting.plot_bp3_limit_cycle_evolution --help
```

Each runner documents its required inputs and writes generated material below
`artifacts/` by default. Reusable numerical kernels remain in
`src/fastslippy/utilities/`; the files here only orchestrate cases, I/O, and
reports.
