# The C API

The [`diffsol-c`](https://crates.io/crates/diffsol-c) crate wraps the
[dynamic dispatch API](./dynamic_dispatch_api.md) as `extern "C"` functions so that they can be called from C code.
A language that can call C functions (for example with `ctypes`, `cffi` or a foreign
function interface) can also be used.

## Install the Debian packages

On Debian 12 (bookworm) and 13 (trixie), and on Ubuntu 22.04 (jammy) and 24.04 (noble), for
amd64 and arm64, you can install the library with apt. You do not need Rust. There are two
packages:

- `libdiffsol<version>`, for example `libdiffsol0.7`: the shared library `libdiffsol.so`. The
  library has the Cranelift and LLVM JIT backends, and it can load DiffSL code that was compiled
  before. It uses the LLVM library of the distribution: `libllvm19` on Debian 12, Debian 13 and
  Ubuntu 24.04, and `libllvm15` on Ubuntu 22.04. apt installs it as a dependency.
- `libdiffsol-dev`: the header `diffsol.h`, the pkg-config file `diffsol.pc`, and the link to the
  shared library. Install this package to compile programs.

Add the APT repository of diffsol, then install the development package:

```sh
curl -1sLf 'https://dl.cloudsmith.io/public/martinjrobins/diffsol/setup.deb.sh' | sudo -E bash
sudo apt install libdiffsol-dev
```

You can also get the packages from the assets of the diffsol-c
[GitHub release](https://github.com/martinjrobins/diffsol/releases). There are packages for each
distribution: the codename of the distribution is part of the package version, for example
`libdiffsol0.7_0.7.0-1~noble_amd64.deb`. Install the two packages of your distribution and
architecture, for example on Ubuntu 24.04:

```sh
sudo apt install ./libdiffsol0.7_*~noble_amd64.deb ./libdiffsol-dev_*~noble_amd64.deb
```

Then compile and link a program with pkg-config:

```sh
cc main.c $(pkg-config --cflags --libs diffsol) -lm -o main
```

The packages install the library as `libdiffsol.so`, so a program links with `-ldiffsol`. A
library that you build with Cargo has the name `libdiffsol_c.so` (see below), so a program links
with `-ldiffsol_c`. The header and the functions are the same.

## Build the library

Build the crate with Cargo. By default the ODE equations can be supplied as C callbacks.
You can also supply the equations using DiffSL via the additional features:

- `diffsl-cranelift`: compile DiffSL code at runtime with the Cranelift JIT backend.
- `diffsl-llvm15` to `diffsl-llvm21`: compile DiffSL code at runtime with LLVM. LLVM of that
  version must be on the build machine. These features link LLVM statically.
- `diffsl-llvm15-prefer-dynamic` to `diffsl-llvm21-prefer-dynamic`: the same, but link the
  shared LLVM library (`libLLVM`) if it is available. Use these to depend on the LLVM package of
  a Linux distribution. The library is then much smaller, but the LLVM library must be installed
  where the program runs.
- `diffsl-external-dynamic` (default): load DiffSL code that was compiled before, from a dynamic
  library.
- `suitesparse`: add the KLU sparse linear solver.

```sh
cargo build -p diffsol-c --release --features diffsl-cranelift
```

The build makes these files in `target/release`:

| File | Description |
| --- | --- |
| `include/diffsol.h` | The C header. |
| `libdiffsol_c.so` (Linux), `libdiffsol_c.dylib` (macOS) | The dynamic library. |
| `diffsol_c.dll` and `diffsol_c.dll.lib` (Windows) | The dynamic library and its import library. |

The build script makes the header with [cbindgen](https://github.com/mozilla/cbindgen). The
header agrees with the features of the build. It defines a macro for each enabled feature, for
example `DIFFSOL_DIFFSL_CRANELIFT`. The functions of a feature are only in the header when the
macro of that feature is defined. Thus C code can use `#ifdef` to find which features the
library has.

## Compile and link a C program

Link with the dynamic library. On Linux and macOS:

```sh
cc main.c -I target/release/include -L target/release -ldiffsol_c -lm \
   -Wl,-rpath,$PWD/target/release -o main
```

The `-rpath` option tells the program where to find the library at runtime. You can also put
the library in a directory that the system searches, or use `LD_LIBRARY_PATH` (Linux) or
`DYLD_LIBRARY_PATH` (macOS).

On Windows, with MSVC:

```bat
cl main.c /I target\release\include target\release\diffsol_c.dll.lib
```

At runtime, `diffsol_c.dll` must be in the same directory as the program, or in a directory in
`PATH`.

### Static library

A static library is made only on request. This command makes it, and lists the system
libraries that a program must also link with:

```sh
cargo rustc -p diffsol-c --release --lib --features diffsl-cranelift --crate-type staticlib \
    -- --print native-static-libs
```

The static library is `target/release/libdiffsol_c.a` (`diffsol_c.lib` on Windows). Link it,
and the system libraries that the command lists. For example, on macOS:

```sh
cc main.c -I target/release/include target/release/libdiffsol_c.a -lc++ -liconv -lm -o main
```

## How to use the API

All C API functions use the same rules:

- A function that can fail returns an `int32_t` status: `DIFFSOL_OK` (0) on success, or a
  negative value (`DIFFSOL_ERR`, `DIFFSOL_BAD_ARG`) on failure. A constructor returns `NULL` on
  failure.
- After a failure, `diffsol_last_error_message()` gives a description of the error. The error is
  kept for each thread.
- The library owns each object that it returns, for example an `OdeWrapper`, a
  `SolutionWrapper` or a `HostArray`. Free each object with its free function, for example
  `diffsol_ode_free`.
- The enum values are `int32_t` constants in the header, for example
  `DIFFSOL_MATRIX_TYPE_NALGEBRA_DENSE`, `DIFFSOL_LINEAR_SOLVER_TYPE_DEFAULT`,
  `DIFFSOL_ODE_SOLVER_TYPE_BDF` and `DIFFSOL_JIT_BACKEND_TYPE_CRANELIFT`.
- A solution array (`HostArray`) gives a pointer to its data, its dimensions and its strides.
  The strides are in bytes.
- The arrays use the column-major order of diffsol: the first axis changes fastest.
- An ODE has `nbatch` lanes. The last axis of an array of lanes is
  always the lane, also when `nbatch` is 1. A state or a parameter vector is an `n x nbatch`
  array. The solution values `ys` are an `nout x nt x nbatch` array. The times `ts` are an `nt`
  array, because all the lanes use the same times.
- A function that takes lanes of values (for example `params`) takes a pointer, the number of
  values in each lane and the number of lanes. The values are a column-major `len x lanes`
  array: lane `b`, value `k` is at index `b * len + k`. The number of lanes is the `nbatch` of
  the ODE, or 1. One lane is used for all the lanes of the ODE.

The functions are in these modules of the crate documentation:

- [ode_c](https://docs.rs/diffsol-c/latest/diffsol_c/ode_c/index.html): configure, solve and
  free `OdeWrapper` handles.
- [ode_callbacks_c](https://docs.rs/diffsol-c/latest/diffsol_c/ode_callbacks_c/index.html):
  create an `OdeWrapper` from equations that C callbacks give.
- [ode_diffsl_c](https://docs.rs/diffsol-c/latest/diffsol_c/ode_diffsl_c/index.html): create an
  `OdeWrapper` from DiffSL code (JIT) or from DiffSL code that was compiled before.
- [solution_wrapper_c](https://docs.rs/diffsol-c/latest/diffsol_c/solution_wrapper_c/index.html):
  get the solution data, and free `SolutionWrapper` handles.
- [ode_options_c](https://docs.rs/diffsol-c/latest/diffsol_c/ode_options_c/index.html) and
  [initial_condition_options_c](https://docs.rs/diffsol-c/latest/diffsol_c/initial_condition_options_c/index.html):
  get and set the solver options.
- [matrix_type_c](https://docs.rs/diffsol-c/latest/diffsol_c/matrix_type_c/index.html),
  [scalar_type_c](https://docs.rs/diffsol-c/latest/diffsol_c/scalar_type_c/index.html),
  [linear_solver_type_c](https://docs.rs/diffsol-c/latest/diffsol_c/linear_solver_type_c/index.html),
  [ode_solver_type_c](https://docs.rs/diffsol-c/latest/diffsol_c/ode_solver_type_c/index.html)
  and [jit_c](https://docs.rs/diffsol-c/latest/diffsol_c/jit_c/index.html): the enum values,
  their names, and their checks.
- [host_array_c](https://docs.rs/diffsol-c/latest/diffsol_c/host_array_c/index.html): read and
  free `HostArray` data.
- [error_c](https://docs.rs/diffsol-c/latest/diffsol_c/error_c/index.html): get and clear the
  error of the thread.
- [string_c](https://docs.rs/diffsol-c/latest/diffsol_c/string_c/index.html): allocate and free
  strings that Rust owns.

## Example: equations from C callbacks

The example
[logistic_callbacks.c](https://github.com/martinjrobins/diffsol/blob/main/crates/diffsol-c/examples/c/logistic_callbacks.c)
solves the logistic equation \\(dy/dt = r \cdot y \cdot (1 - y)\\), with equations that C
functions give. It does not need DiffSL.

Each callback gets its inputs as arrays, writes its output to `y`, and returns `0` on success.
Its last argument is the `user_data` pointer of the descriptor. The
[ode_callbacks module](https://docs.rs/diffsol-c/latest/diffsol_c/ode_callbacks/index.html)
gives the array lengths of each callback, the sign conventions of the adjoint callbacks, and the
ownership rules of `user_data`.

```c
{{#include ../../../crates/diffsol-c/examples/c/logistic_callbacks.c:callbacks}}
```

Put the callbacks in a `DiffsolOdeCallbacksF64` descriptor, and create the solver. Only `rhs`
and `init` are necessary. A `NULL` callback is not given. If a solve needs a callback that is
not given, the solve returns an error that gives the name of the callback. For example, the
implicit methods (such as BDF) need `rhs_jac`.

```c
{{#include ../../../crates/diffsol-c/examples/c/logistic_callbacks.c:create}}
```

Solve, and read the solution:

```c
{{#include ../../../crates/diffsol-c/examples/c/logistic_callbacks.c:solve}}
```

## Example: DiffSL code with the JIT

The example
[logistic_diffsl_jit.c](https://github.com/martinjrobins/diffsol/blob/main/crates/diffsol-c/examples/c/logistic_diffsl_jit.c)
solves the same equation from DiffSL code. The library compiles the code at runtime. Thus the
library must have a JIT backend (`diffsl-cranelift` or `diffsl-llvm*`).

```c
{{#include ../../../crates/diffsol-c/examples/c/logistic_diffsl_jit.c:code}}
```

Select a JIT backend that the library has, and create the solver:

```c
{{#include ../../../crates/diffsol-c/examples/c/logistic_diffsl_jit.c:create}}
```

The solve is the same as in the callbacks example.

## Run the examples

Build the library, then compile and run an example. On Linux and macOS:

```sh
cargo build -p diffsol-c --features diffsl-cranelift
cc crates/diffsol-c/examples/c/logistic_diffsl_jit.c -I target/debug/include \
   -L target/debug -ldiffsol_c -lm -Wl,-rpath,$PWD/target/debug -o logistic_diffsl_jit
./logistic_diffsl_jit
```

The script `crates/diffsol-c/examples/c/run_examples.sh` compiles and runs the examples against
a Cargo build. It runs the JIT example only if the library has a JIT backend, and also runs it
with LLVM if the library has LLVM. The CI runs it on Linux, macOS and Windows (in Git Bash, with
MSVC).

```sh
cargo build -p diffsol-c --features diffsl-cranelift
crates/diffsol-c/examples/c/run_examples.sh target/debug
```

The JIT example uses Cranelift if the library has it. To select the backend, set the environment
variable `DIFFSOL_JIT_BACKEND` to `cranelift` or `llvm`.
