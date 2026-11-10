# The dynamic dispatch API

The [`diffsol-c`](https://crates.io/crates/diffsol-c) crate has two APIs that let you use
diffsol from other languages:

1. The **dynamic dispatch** API, which this page describes. It is a Rust API. It wraps the core
   `diffsol` solver in types that select their implementation at runtime. Thus you can select
   the matrix backend, the scalar type, the linear solver, the ODE solver method and the JIT
   backend at runtime, not at compile time.
2. The **C API**. It gives the dynamic dispatch API as `extern "C"` functions, so that C code,
   or a language that can call C, can use it. See [The C API](./c_api.md).

## Key types

The central type is
[OdeWrapper](https://docs.rs/diffsol-c/latest/diffsol_c/ode/struct.OdeWrapper.html). It keeps
the solver state in an `Arc<Mutex<...>>`, so that threads and FFI code can share it safely. Its
methods create, configure, solve and serialize the solver.

You configure the solver with these runtime enum types:

- [OdeSolverType](https://docs.rs/diffsol-c/latest/diffsol_c/ode_solver_type/enum.OdeSolverType.html):
  the ODE integration method.
- [MatrixType](https://docs.rs/diffsol-c/latest/diffsol_c/matrix_type/enum.MatrixType.html):
  the linear algebra backend.
- [ScalarType](https://docs.rs/diffsol-c/latest/diffsol_c/scalar_type/enum.ScalarType.html):
  the floating-point precision of the solver.
- [LinearSolverType](https://docs.rs/diffsol-c/latest/diffsol_c/linear_solver_type/enum.LinearSolverType.html):
  the linear solver of the implicit methods.
- [JitBackendType](https://docs.rs/diffsol-c/latest/diffsol_c/jit/enum.JitBackendType.html):
  the JIT compilation backend for DiffSL code.

A solve returns a
[SolutionWrapper](https://docs.rs/diffsol-c/latest/diffsol_c/solution_wrapper/struct.SolutionWrapper.html).
From it, you can get the time points, the state values and the sensitivities as
[Tensor](https://docs.rs/diffsol-c/latest/diffsol_c/tensor/struct.Tensor.html)
objects. A `Tensor` shares the data of the solution, with no copy, and keeps the solution alive,
so it stays valid after the `SolutionWrapper` is dropped.

The inputs of the solve methods, for example the parameters and the times `t_eval`, are
[TensorView](https://docs.rs/diffsol-c/latest/diffsol_c/tensor/struct.TensorView.html) objects.
A `TensorView` borrows the data of the caller, with no copy. It is small and only on the stack,
and it has at most 3 axes. Make one from a slice with `TensorView::from_slice`, or from an
`Tensor` with `Tensor::view`.

The arrays use the column-major order of diffsol, and the last axis of an array of lanes is the
lane. For example, the parameters are an `nparams x nlanes`
array, and the state values of a solution are an `nout x nt x nbatch` array.

You can give the equations as DiffSL code (`OdeWrapper::new_jit`, `new_external` and
`new_external_dynamic`), or as C callbacks (`OdeWrapper::new_callbacks`, see
[The C API](./c_api.md)).

## Example: the logistic equation

This example uses the dynamic dispatch API to solve the logistic ODE
\\(dy/dt = r \cdot y \cdot (1 - y)\\) from Rust. All of the solver configuration occurs at
runtime, with the `OdeWrapper` API.

The DiffSL code of the logistic equation:

```rust,ignore
{{#include ../../../examples/diffsol-c-logistic/src/main.rs:logistic_code}}
```

Create the `OdeWrapper` with the Cranelift JIT backend, `f64` precision and the BDF solver:

```rust,ignore
{{#include ../../../examples/diffsol-c-logistic/src/main.rs:create_wrapper}}
```

Set the solver tolerances:

```rust,ignore
{{#include ../../../examples/diffsol-c-logistic/src/main.rs:configure}}
```

Solve on a fixed time grid:

```rust,ignore
{{#include ../../../examples/diffsol-c-logistic/src/main.rs:solve_dense}}
```

Get the solution data from the `SolutionWrapper`:

```rust,ignore
{{#include ../../../examples/diffsol-c-logistic/src/main.rs:extract}}
```

The full example is in
[examples/diffsol-c-logistic](https://github.com/martinjrobins/diffsol/tree/main/examples/diffsol-c-logistic).
