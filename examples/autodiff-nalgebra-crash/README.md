# autodiff-nalgebra-crash

Minimal reproducible example for a `std::autodiff` compiler crash: the Enzyme backend cannot differentiate through `nalgebra::DVector` operations.

## The crash

Applying `#[autodiff_*]` to a function taking `&DVector<f64>` / `&mut DVector<f64>` causes Enzyme to abort with:

```
Cannot deduce adding type
UNREACHABLE executed at DiffeGradientUtils.cpp:524
```

**Root cause:** nalgebra's `DVector` uses `VecStorage` with `get_address_unchecked_mut` — raw pointer arithmetic that Enzyme cannot trace through.

**Workaround:** Write scalar functions (`f64`, `&[f64]`) and adapt from vectors to scalars in wrapper code (as done by diffsol's `ClosureAutodiff`).

## Build

```bash
CARGO_PROFILE_DEV_LTO=fat RUSTFLAGS="-Z autodiff=Enable" \
  cargo +nightly run -p autodiff-nalgebra-crash -j1
```
