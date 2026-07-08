# autodiff-nalgebra-crash

Minimal reproducible example for a `std::autodiff` compiler crash in **debug mode**:
the Enzyme backend cannot differentiate through `nalgebra::DVector` operations.

## The crash

Applying `#[autodiff_*]` to a function taking `&DVector<f64>` / `&mut DVector<f64>`
in debug mode causes Enzyme to abort with:

```
Cannot deduce adding type
UNREACHABLE executed at DiffeGradientUtils.cpp:524
```

**Root cause:** nalgebra's `DVector` uses `VecStorage` with `get_address_unchecked_mut` — raw pointer arithmetic that Enzyme cannot trace through in debug mode.

**Workaround:** Compile in release mode.

## Build

```bash
# Debug build (CRASHES)
CARGO_PROFILE_DEV_LTO=fat RUSTFLAGS="-Z autodiff=Enable" \
  cargo +nightly run -p autodiff-nalgebra-crash -j1

# Release build (works)
CARGO_PROFILE_RELEASE_LTO=fat RUSTFLAGS="-Z autodiff=Enable" \
  cargo +nightly run -p autodiff-nalgebra-crash --release -j1
```
