# CPU Ensemble Performance

Solves an ensemble of logistic-growth ODEs in parallel with Rayon, then plots
the 5th, 50th, and 95th percentiles. It also measures speed-up from one to the
available Rayon worker threads; timing points use the median of repeated runs.

Run it with:

```sh
cargo run -p performance-cpu-ensemble --release
```

Plots are written to `book/src/performance/images/`.
