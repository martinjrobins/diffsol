# GPU Ensemble Performance

Compares solving many power-grid swing-equation simulations as one batched GPU
solve with solving one grid per Rayon task on the CPU. It reports ensemble
statistics and plots CPU/GPU timing across ensemble sizes; timing points use
the median of repeated runs.

Run it on a supported CUDA Oxide device with:

```sh
just gpu-ensemble
```

Plots are written to `book/src/performance/images/`.
