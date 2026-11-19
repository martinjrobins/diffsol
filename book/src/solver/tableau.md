# Butcher Tableau

The Butcher tableau is a way of representing the coefficients of a Runge-Kutta method (see the [wikipedia page](https://en.wikipedia.org/wiki/Butcher_tableau)). diffsol uses the [`Tableau`](https://docs.rs/diffsol/latest/diffsol/ode_solver/tableau/struct.Tableau.html) struct to represent the tableau, and this is used to create any of the SDIRK or ERK solvers. diffsol has a few inbuilt tableaus that you can use, otherwise you can create your own by constructing an instance of `Tableau`.

To create an SDIRK or ERK solver using a pre-defined tableau, you can use methods on the `OdeSolverProblem` struct like so:

```rust,ignore
{{#include ../../../examples/intro-logistic-closures/src/create_solvers_tableau.rs}}
```

The `rk_state` method in this example does not make the initial state consistent with any algebraic constraints. If your problem has a mass matrix, use `rk_state_and_consistent` instead to create the state for an SDIRK solver. The explicit Runge-Kutta solver does not support a mass matrix.
