# C and other languages

The [`diffsol-c`](https://crates.io/crates/diffsol-c) crate provides two
higher-level APIs for using Diffsol from other languages with a
C-compatible FFI:

1. A **dynamic dispatch** API — a Rust API that wraps the core `diffsol` solver
   in runtime-dispatched types, allowing you to choose the matrix backend,
scalar type, linear solver, ODE solver method, and JIT backend at runtime
rather than at compile time.
2. A **C FFI** API — `extern "C"` functions that expose the dynamic dispatch
   API as a C-compatible interface, enabling direct use from C code or any
language with C bindings.

## Dynamic Dispatch API

The core `diffsol` crate requires you to specify the matrix type, code
generation backend, ODE equation type, linear solver, and solver method all as
generic type parameters at compile time. The dynamic dispatch API replaces
these generic parameters with runtime enums, storing the concrete
implementations behind trait objects.

### Key types

The central type is the
[OdeWrapper](https://docs.rs/diffsol-c/latest/diffsol_c/ode/struct.OdeWrapper.html),
which wraps the solver state in an `Arc<Mutex<...>>` for safe sharing across
threads and FFI boundaries. It provides methods to create, configure, solve,
and serialize the solver.

Configuration is done via runtime enum types:

- [OdeSolverType](https://docs.rs/diffsol-c/latest/diffsol_c/ode_solver_type/enum.OdeSolverType.html):
the ODE integration method
- [MatrixType](https://docs.rs/diffsol-c/latest/diffsol_c/matrix_type/enum.MatrixType.html):
the linear algebra backend
- [ScalarType](https://docs.rs/diffsol-c/latest/diffsol_c/scalar_type/enum.ScalarType.html):
the floating-point precision for the solver
- [LinearSolverType](https://docs.rs/diffsol-c/latest/diffsol_c/linear_solver_type/enum.LinearSolverType.html):
the linear solver for implicit methods
- [JitBackendType](https://docs.rs/diffsol-c/latest/diffsol_c/jit/enum.JitBackendType.html):
the JIT compilation backend for DiffSL code

Solution results are returned as a
[SolutionWrapper](https://docs.rs/diffsol-c/latest/diffsol_c/solution_wrapper/struct.SolutionWrapper.html),
from which you can extract the time points, state values, and sensitivities as
[HostArray](https://docs.rs/diffsol-c/latest/diffsol_c/host_array/struct.HostArray.html)
objects — read-only views of Rust-allocated data that can be accessed without
copying.

### Example: Logistic Equation

Below is a complete example of using the dynamic dispatch API to solve the
logistic ODE \\(dy/dt = r \cdot y \cdot (1 - y)\\) from the Rust side. The
solver is configured entirely at runtime using the `OdeWrapper` API.

The DiffSL code for the logistic equation:

```rust,ignore
{{#include ../../../examples/diffsol-c-logistic/src/main.rs:logistic_code}}
```

Creating the `OdeWrapper` with the Cranelift JIT backend, `f64` precision, and the BDF solver:

```rust,ignore
{{#include ../../../examples/diffsol-c-logistic/src/main.rs:create_wrapper}}
```

Configuring the solver tolerances:

```rust,ignore
{{#include ../../../examples/diffsol-c-logistic/src/main.rs:configure}}
```

Solving on a fixed time grid:

```rust,ignore
{{#include ../../../examples/diffsol-c-logistic/src/main.rs:solve_dense}}
```

Extracting the solution data from the `SolutionWrapper`:

```rust,ignore
{{#include ../../../examples/diffsol-c-logistic/src/main.rs:extract}}
```

See the full example at [examples/diffsol-c-logistic](https://github.com/martinjrobins/diffsol/tree/main/examples/diffsol-c-logistic).

## C FFI API

The C FFI API exposes the dynamic dispatch API as `extern "C"` functions,
making it callable from C or any language with C interop. The functions are
organized into modules suffixed `_c` in the crate documentation:

- [ode_c](https://docs.rs/diffsol-c/latest/diffsol_c/ode_c/index.html) —
create, configure, solve, and destroy `OdeWrapper` handles.
- [solution_wrapper_c](https://docs.rs/diffsol-c/latest/diffsol_c/solution_wrapper_c/index.html)
— extract solution data and destroy `SolutionWrapper` handles.
- [ode_options_c](https://docs.rs/diffsol-c/latest/diffsol_c/ode_options_c/index.html)
and
[initial_condition_options_c](https://docs.rs/diffsol-c/latest/diffsol_c/initial_condition_options_c/index.html)
— get/set solver options.
- [matrix_type_c](https://docs.rs/diffsol-c/latest/diffsol_c/matrix_type_c/index.html),
[scalar_type_c](https://docs.rs/diffsol-c/latest/diffsol_c/scalar_type_c/index.html),
[linear_solver_type_c](https://docs.rs/diffsol-c/latest/diffsol_c/linear_solver_type_c/index.html),
[ode_solver_type_c](https://docs.rs/diffsol-c/latest/diffsol_c/ode_solver_type_c/index.html),
[jit_c](https://docs.rs/diffsol-c/latest/diffsol_c/jit_c/index.html) — query
and convert enum values.
- [host_array_c](https://docs.rs/diffsol-c/latest/diffsol_c/host_array_c/index.html)
— inspect and free `HostArray` data.
- [error_c](https://docs.rs/diffsol-c/latest/diffsol_c/error_c/index.html) —
retrieve and clear thread-local error messages.
- [string_c](https://docs.rs/diffsol-c/latest/diffsol_c/string_c/index.html)
— allocate and free Rust-owned strings from C.

All C FFI functions follow a common pattern:

- Return `i32` (0 = success, negative = error).
- Store error details in a thread-local variable, retrievable via functions in `error_c`.
- Use raw pointers for ownership transfer — the caller allocates and frees via dedicated functions.
