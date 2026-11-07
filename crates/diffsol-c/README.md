# diffsol-c

A runtime-typed wrapper and C API around [diffsol](https://github.com/martinjrobins/diffsol).

Where the `diffsol` crate selects its scalar, matrix, and solver types at compile
time via generics, `diffsol-c` exposes them as runtime choices behind a C-compatible
interface, making diffsol usable from C and other languages via FFI.

The crate is built as a C dynamic library and a Rust library
(`crate-type = ["cdylib", "rlib"]`). A C static library is made on request (see below).

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

## Use from C

On Debian 12+ and Ubuntu 22.04+, you can install the library as Debian packages
(`libdiffsol<version>` and `libdiffsol-dev`, which install the library as `libdiffsol.so`). See the book chapter
[The C API](https://martinjrobins.github.io/diffsol/use/c_api.html). The packaging is in
`debian/`.

To build the library yourself:

A build of this crate makes these files in `target/<profile>`:

- The dynamic library: `libdiffsol_c.so` (Linux), `libdiffsol_c.dylib` (macOS), or
  `diffsol_c.dll` with the import library `diffsol_c.dll.lib` (Windows).
- `include/diffsol.h`: the C header. The build script makes it with
  [cbindgen](https://github.com/mozilla/cbindgen). The header agrees with the features of the
  build. It defines a macro for each enabled feature, for example `DIFFSOL_DIFFSL_CRANELIFT`.

Link with the dynamic library:

```sh
cargo build -p diffsol-c --release --features diffsl-cranelift
cc main.c -I target/release/include -L target/release -ldiffsol_c -lm \
   -Wl,-rpath,target/release -o main
```

Or make and link the static library `libdiffsol_c.a` (`diffsol_c.lib` on Windows). Then also
link the system libraries that the static library needs. This command makes the static library,
and lists the system libraries for your platform and features:

```sh
cargo rustc -p diffsol-c --release --lib --features diffsl-cranelift --crate-type staticlib \
    -- --print native-static-libs
```

For example, on macOS:

```sh
cc main.c -I target/release/include target/release/libdiffsol_c.a -lc++ -liconv -lm -o main
```

The book chapter [The C API](https://martinjrobins.github.io/diffsol/use/c_api.html) gives
more details. The directory `examples/c` has two examples. The tests compile and run them.

- `logistic_callbacks.c`: give the equations as C callbacks (`diffsol_ode_new_callbacks_f64`).
- `logistic_diffsl_jit.c`: give the equations as DiffSL code, which the library compiles at run
  time (`diffsol_ode_new_jit`). This needs a JIT backend feature.

## Links

- Main crate: <https://crates.io/crates/diffsol>
- Documentation: <https://docs.rs/diffsol-c>
- Book: <https://martinjrobins.github.io/diffsol/>

## License

Licensed under the MIT license. See the [repository](https://github.com/martinjrobins/diffsol) for details.
