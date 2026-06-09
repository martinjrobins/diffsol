# logistic-autodiff

Demonstrates using `std::autodiff` to generate derivative functions for diffsol's implicit BDF solver and adjoint sensitivity analysis.

## Build

Requires nightly Rust and `lto = "fat"`:

```bash
CARGO_PROFILE_DEV_LTO=fat RUSTFLAGS="-Z autodiff=Enable" \
  cargo +nightly run -p logistic-autodiff -j1
```

The `-j1` limits parallelism to avoid OOM during fat LTO codegen.
