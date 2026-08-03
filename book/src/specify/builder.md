# The builder

The simplest way to create a new problem is to use the [`OdeBuilder`](https://docs.rs/diffsol/latest/diffsol/ode_solver/builder/struct.OdeBuilder.html) struct.
You can set many configuration options such as:

- initial time (`OdeBuilder::t0`),
- initial step size (`OdeBuilder::h0`)
- relative tolerance (`OdeBuilder::rtol`)
- absolute tolerance (`OdeBuilder::atol`)
- parameters (`OdeBuilder::p`)
- equations (`OdeBuilder::rhs`, `OdeBuilder::init`, `OdeBuilder::mass` etc.)

or leave them at their default values. Then, call the `OdeBuilder::build` method to create a [`OdeSolverProblem`](https://docs.rs/diffsol/latest/diffsol/ode_solver/problem/struct.OdeSolverProblem.html).

```rust,ignore
{{#include ../../../examples/intro-logistic-closures/src/main.rs:builder_intro}}
```

Note that the `OdeBuilder` struct has a generic parameter `M`, which defines the matrix type to use when solving the problem.
This matrix type must satisfy the trait [`Matrix`](https://docs.rs/diffsol/latest/diffsol/trait.Matrix.html), and could be
one of the matrix types included with diffsol:

- [`NalgebraMat`](https://docs.rs/diffsol/latest/diffsol/struct.NalgebraMat.html), this is a thin wrapper around the `nalgebra` crate dense matrix.
- [`FaerMat`](https://docs.rs/diffsol/latest/diffsol/struct.FaerMat.html), this is a thin wrapper around the `faer` crate dense matrix.
- [`FaerSparseMat`](https://docs.rs/diffsol/latest/diffsol/struct.FaerSparseMat.html), this is a thin wrapper around the `faer` sparse matrix.
- `CudaMat`, this is diffsol's CUDA matrix type (requires the `cuda` feature)

Each matrix type is parameterised by a scalar type that satisfies the [`Scalar`](https://docs.rs/diffsol/latest/diffsol/trait.Scalar.html) trait,
each matrix type also has an associated type that defines its corresponding vector type (bounded by the [`Vector`](https://docs.rs/diffsol/latest/diffsol/trait.Vector.html) trait).
So whenever you create a builder you are also defining what matrix, vector and scalar types to use for the problem that you will create using that builder.
