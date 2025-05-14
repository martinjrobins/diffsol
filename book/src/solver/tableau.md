# Butchers Tableau

The Butcher tableau is a way of representing the coefficients of a Runge-Kutta method (see the [wikipedia page](https://en.wikipedia.org/wiki/Butcher_tableau)). Diffsol uses the [`Tableau`](https://docs.rs/diffsol/latest/diffsol/ode_solver/tableau/struct.Tableau.html) struct to represent the tableau, and this is used to create any of the SDIRK or ERK solvers. Diffsol has a few inbuilt tableaus that you can use, otherwise you can create your own by constructing an instance of `Tableau`

To create an SDIRK or ERK solver using a pre-defined tableau, you can use methods on the `OdeSolverProblem` struct like so:

```rust,ignore
{{#include ../../../examples/intro-logistic-closures/src/create_solvers_tableau.rs}}
```