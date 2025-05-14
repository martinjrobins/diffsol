# Solving the Problem

Each solver implements the [`OdeSolverMethod`](https://docs.rs/diffsol/latest/diffsol/ode_solver/method/trait.OdeSolverMethod.html) trait, which provides a number of methods to solve the problem.

## Solving the Problem

Diffsol has a few high-level solution functions on the `OdeSolverMethod` trait that are the easiest way to solve your equations:
- [`solve`](https://docs.rs/diffsol/latest/diffsol/ode_solver/method/trait.OdeSolverMethod.html#method.solve) - solve the problem from an initial state up to a specified time, returning the solution at all the internal timesteps used by the solver.
- [`solve_dense`](https://docs.rs/diffsol/latest/diffsol/ode_solver/method/trait.OdeSolverMethod.html#method.solve_dense) - solve the problem from an initial state, returning the solution at a `Vec` of times provided by the user.
- [`solve_dense_sensitivities`](https://docs.rs/diffsol/latest/diffsol/ode_solver/method/trait.OdeSolverMethod.html#method.solve_dense_sensitivities) - solve the forward sensitivity problem from an initial state, returning the solution at a `Vec` of times provided by the user.
- [`solve_adjoint`](https://docs.rs/diffsol/latest/diffsol/ode_solver/method/trait.OdeSolverMethod.html#method.solve_adjoint) - solve the adjoint sensitivity problem from an initial state to a final time, returning the integration of the output function over time as well as its gradient with respect to the initial state.

The following example shows how to solve a simple ODE problem using the `solve` method on the `OdeSolverMethod` trait. 

```rust,ignore
{{#include ../../examples/intro-logistic-closures/src/main.rs:1:21}}
{{#include ../../examples/intro-logistic-closures/src/main.rs:56:57}}
```

`solve_dense` will solve a problem from an initial state, returning the solution at a `Vec` of times provided by the user.

```rust,ignore
{{#include ../../examples/intro-logistic-closures/src/main.rs:60:62}}
```

## Stepping the Solution



The second way is to use the `set_stop_time` method on the `OdeSolverMethod` trait to stop the solver at a specific time, this will override the internal time step so that the solver stops at the specified time.
Note that this can be less efficient if you wish to continue stepping forward after the specified time, as the solver will need to be re-initialised.
The enum returned by `step` will indicate when the solver has stopped at the specified time.
Once the solver has stopped at the specified time, you can get the current state of the solution using the `state` method on the solver, which returns an [`OdeSolverState`](https://docs.rs/diffsol/latest/diffsol/ode_solver/method/struct.OdeSolverState.html) struct.

```rust,ignore
{{#include ../../examples/intro-logistic-closures/src/main.rs:81:91}}
```