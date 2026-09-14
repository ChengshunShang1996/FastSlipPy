# BP3 benchmark configurations

This directory contains reproducible, production-size SEAS BP3 configurations.
They are intentionally separate from `examples/`: a benchmark may require a
large sparse factorization, many time steps, and substantial output storage.

Run the 90-degree comparison case from the repository root:

```console
python -m benchmarks.bp3.run_case_bp3_real_case
```

The short, self-contained BP3 introduction remains in
`examples/run_case_BP3.py`.
