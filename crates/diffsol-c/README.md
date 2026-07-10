# diffsol-c

A runtime-typed wrapper and C API around [diffsol](https://github.com/martinjrobins/diffsol).

Where the `diffsol` crate selects its scalar, matrix, and solver types at compile
time via generics, `diffsol-c` exposes them as runtime choices behind a C-compatible
interface, making diffsol usable from C and other languages via FFI.

The crate is built as both a C dynamic library and a Rust library
(`crate-type = ["cdylib", "rlib"]`).

## What's in this crate

- Runtime selection of scalar / matrix / linear-solver / ODE-solver types.
- A C API for building and solving ODE/DAE problems.
- Integration with the [DiffSL](https://martinjrobins.github.io/diffsl/) DSL for
  specifying equations.

## Features

- `diffsl-cranelift`: DiffSL with the Cranelift JIT backend.
- `diffsl-llvm15` … `diffsl-llvm21`: DiffSL with the corresponding LLVM backend.
- `diffsl-external-f64` / `diffsl-external-f32` / `diffsl-external-dynamic`:
  externally compiled DiffSL modules.
- `suitesparse`: enable the KLU sparse direct solver (via `diffsol/suitesparse`).
- `wasm`: WebAssembly support.

## Links

- Main crate: <https://crates.io/crates/diffsol>
- Documentation: <https://docs.rs/diffsol-c>
- Book: <https://martinjrobins.github.io/diffsol/>

## License

Licensed under the MIT license. See the [repository](https://github.com/martinjrobins/diffsol) for details.
