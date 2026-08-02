# Parameters

The fundamental trait that diffsol uses to specify a mathematical operation is [`Op`](https://docs.rs/diffsol/latest/diffsol/op/trait.Op.html).
This trait defines the number of parameters in your ODE model via [`Op::nparams`](https://docs.rs/diffsol/latest/diffsol/op/trait.Op.html#tymethod.nparams).
When specifying your parameters with DiffSL or Rust closures, you are largely tied into
a specific notion of a parameter being *any* input parameter of your model that you want to
change between solves.
However, if you use a custom struct and the `OdeEquations` trait, you have much more
flexibility about defining and changing the parameters of your model, as we'll explore
below.

## What is a "parameter"

A parameter of your model does not continuously vary with time, like the state variables,
but is instead constant over the duration of an individual solve. If you were writing your
equations using a custom struct, you might have one or more model parameters as fields on
that struct, for example:

```rust,ignore
struct MyModel {
  pub parameter_a: f64,
  pub parameter_b: f64,
}
```

These parameters could then be used in your implementation of [`NonLinearOp`](https://docs.rs/diffsol/latest/diffsol/op/nonlinear_op/trait.NonLinearOp.html), or any other diffsol trait. However,
it is *entirely up to you* if you report these to diffsol as parameters or not. You could
set `nparams` in the [`Op`](https://docs.rs/diffsol/latest/diffsol/op/trait.Op.html) trait to
2 and implement the `set_params` and `get_params` methods to set/get both `parameter_a` and
`parameter_b`, or you can set `nparams` to 0 and just set `parameter_a` and `parameter_b` manually
via their `pub` fields on the struct, its up to you. This also gives you flexibility to
use parameter types that ain't supported by diffsol, for example `int`, `bool`, or arbitrary structs.

```rust,ignore
struct MyModel {
  pub parameter_a: f64,
  pub parameter_b: Hashmap<String, f64>,
}
```

## When to tell diffsol about a parameter

There are a few cases where you *do* need to report your parameter to diffsol, these are:

- You want to exclusively set/get parameters via the `OdeEquations` trait methods `set_params` and `get_params`.
  For example you might want to write generic code that can work over any set of equations.
- You want to calculate the gradient of your solution with respect to that parameter using a forward sensitivity solver.
- You want to calculate the gradient of a loss function with respect to that parameter using an adjoint sensitivity solver.

This decision can be made on a per-parameter basis. You can select which parameters of your model
you want to tell diffsol about, and which parameters you don't, on a case-by-case basis. Just make sure you are
consistent on the number of parameters that you are reporting to diffsol, and in which order, when you implement
`get/set_params` and the `*Sens` traits.

## Varying parameters

The rust borrow checker governs any change to your model. Any change in a parameter of your model, whether its a parameter you have reported to diffsol or otherwise, requires a mutation of your `OdeEquations` struct. Each solver that you create takes a reference to
[`OdeSolverProblem`](https://docs.rs/diffsol/latest/diffsol/ode_solver/problem/struct.OdeSolverProblem.html), and therefore to your custom `OdeEquations` struct, so you won't
be able to mutate it until you drop all the solvers with that reference. Given any instance of `OdeSolverProblem`, you can get a mutable reference to you custom `OdeEquations` struct using
[`OdeSolverProblem::eqn_mut`](https://docs.rs/diffsol/latest/diffsol/ode_solver/problem/struct.OdeSolverProblem.html#method.eqn_mut).
