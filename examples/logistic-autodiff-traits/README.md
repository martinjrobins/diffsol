# logistic-autodiff-traits

Demonstrates the `rhs_autodiff` and `init_autodiff` builder methods from diffsol.
The user passes closures directly to the builder — `std::autodiff` handles the
gradients automatically.

## Build

Requires nightly Rust, the `autodiff` feature, `lto = "fat"`, and **release mode**
to avoid Enzyme crashes on nalgebra types:

```bash
CARGO_PROFILE_RELEASE_LTO=fat RUSTFLAGS="-Z autodiff=Enable" \
  cargo +nightly run -p logistic-autodiff-traits --features autodiff --release -j1
```
