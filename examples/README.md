# Examples

This directory contains the supported, reasonably small entry points intended
for learning and routine verification:

- `run_case_BP3.py`: compact BP3 setup
- `run_case_groningen.py`: Groningen field-scale case
- `run_case_lab.py` and `run_case_lab_iterative.py`: laboratory cases
- `run_case_stretched_mesh.py`: stretched-mesh example

Run them from the repository root, for example:

```console
python examples/run_case_BP3.py
```

The production-size BP3 reproducibility case is kept in
[`benchmarks/bp3/`](../benchmarks/bp3). Development diagnostics, convergence
studies, and plotting runners are kept in [`tools/bp3/`](../tools/bp3), so they
are not presented as part of the public quick-start examples.
