# The builder

The simplest way to create a new problem is to use the [`OdeBuilder`](https://docs.rs/diffsol/latest/diffsol/ode_solver/builder/struct.OdeBuilder.html) struct.
You can set many configuration options such as:

- initial time (`OdeBuilder::t0`),
- initial step size (`OdeBuilder::h0`)
- relative tolerance (`OdeBuilder::rtol`)
- absolute tolerance (`OdeBuilder::atol`)
- fixed-step operation (`OdeBuilder::disable_state_error_control`)
- maximum BDF order (`OdeBuilder::max_order`)
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
- `OxideMat`, this is diffsol's CUDA matrix type (requires the `cuda-oxide` feature)
- `CudaMat`, this is diffsol's deprecated C CUDA matrix type (requires the `cuda` feature), use `OxideMat` instead

Each matrix type is parameterised by a scalar type that satisfies the [`Scalar`](https://docs.rs/diffsol/latest/diffsol/trait.Scalar.html) trait,
each matrix type also has an associated type that defines its corresponding vector type (bounded by the [`Vector`](https://docs.rs/diffsol/latest/diffsol/trait.Vector.html) trait).
So whenever you create a builder you are also defining what matrix, vector and scalar types to use for the problem that you will create using that builder.

By default, state error control is enabled with relative and absolute tolerances of `1e-6`. To use fixed-step operation, call
`disable_state_error_control()` on the builder. Successful steps then keep the current step size and skip error estimates for
the state, outputs, and forward sensitivities. Nonlinear solve failures can still cause a retry with a smaller step. A step may
be temporarily shortened to land on a requested stop time, after which the previous step size is restored.

This differs from `turn_off_output_error_control()` and `turn_off_sensitivities_error_control()`: those methods remove only
the selected output or sensitivity contribution while state error control remains active. Output or sensitivity values are
still integrated when their error contribution is turned off.

For BDF, set an upper order with `max_order(order)`. The supported range is 1 through 5, and the default is 5. For example,
`.max_order(3)` limits BDF to order 3.
