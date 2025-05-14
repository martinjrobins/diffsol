# Solving the problem

Each solver implements the [`OdeSolverMethod`](https://docs.rs/diffsol/latest/diffsol/ode_solver/method/trait.OdeSolverMethod.html) trait, which provides a number of high-level methods to solve the problem.

- [`solve`](https://docs.rs/diffsol/latest/diffsol/ode_solver/method/trait.OdeSolverMethod.html#method.solve) - solve the problem from an initial state up to a specified time, returning the solution at all the internal timesteps used by the solver.
- [`solve_dense`](https://docs.rs/diffsol/latest/diffsol/ode_solver/method/trait.OdeSolverMethod.html#method.solve_dense) - solve the problem from an initial state, returning the solution at a `Vec` of times provided by the user.
- [`solve_dense_sensitivities`](https://docs.rs/diffsol/latest/diffsol/ode_solver/method/trait.OdeSolverMethod.html#method.solve_dense_sensitivities) - solve the forward sensitivity problem from an initial state, returning the solution at a `Vec` of times provided by the user.
- [`solve_adjoint`](https://docs.rs/diffsol/latest/diffsol/ode_solver/method/trait.OdeSolverMethod.html#method.solve_adjoint) - solve the adjoint sensitivity problem from an initial state to a final time, returning the integration of the output function over time as well as its gradient with respect to the initial state.

The following example shows how to solve a simple ODE problem up to \\(t=10\\) using the `solve` method on the `OdeSolverMethod` trait. The solver will return the solution at all the internal timesteps used by the solver (`_ys`), as well as the timesteps used by the solver (`_ts`). The solution is returned as a matrix whose columns are the solution at each timestep.

```rust,ignore
{{#include ../../../examples/intro-logistic-closures/src/solve.rs}}
```

`solve_dense` will solve a problem from an initial state, returning the solution as a matrix whose columns are the solution at each timestep in `times`.

```rust,ignore
{{#include ../../../examples/intro-logistic-closures/src/solve_dense.rs}}
```