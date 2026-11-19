# Creating a solver

Once you have defined the problem, you need to create a solver to solve the problem. The available solvers are:
- [`diffsol::Bdf`](https://docs.rs/diffsol/latest/diffsol/ode_solver/bdf/struct.Bdf.html): A Backwards Difference Formulae solver, suitable for stiff problems. It supports mass matrices, including singular mass matrices (DAEs).
- [`diffsol::Sdirk`](https://docs.rs/diffsol/latest/diffsol/ode_solver/sdirk/struct.Sdirk.html): A Singly Diagonally Implicit Runge-Kutta (SDIRK or ESDIRK) solver, suitable for stiff problems. It supports mass matrices, including singular mass matrices (DAEs). You can define your own Butcher tableau using [`Tableau`](https://docs.rs/diffsol/latest/diffsol/ode_solver/tableau/struct.Tableau.html) or use one of the pre-defined tableaus.
- [`diffsol::ExplicitRk`](https://docs.rs/diffsol/latest/diffsol/ode_solver/explicit_rk/struct.ExplicitRk.html): An explicit Runge-Kutta solver, suitable for non-stiff problems. It does not support a mass matrix: if the problem has a mass matrix, creating the solver returns a `MassMatrixNotSupported` error. You can define your own Butcher tableau using [`Tableau`](https://docs.rs/diffsol/latest/diffsol/ode_solver/tableau/struct.Tableau.html) or use one of the pre-defined tableaus.

For the implicit solvers (BDF and SDIRK), you will need to specify the linear solver type to use. The explicit Runge-Kutta solver does not use a linear solver. The available linear solvers are:
- [`diffsol::NalgebraLU`](https://docs.rs/diffsol/latest/diffsol/struct.NalgebraLU.html): A LU decomposition solver using the [nalgebra](https://nalgebra.org) crate.
- [`diffsol::FaerLU`](https://docs.rs/diffsol/latest/diffsol/struct.FaerLU.html): A LU decomposition solver using the [faer](https://github.com/sarah-ek/faer-rs) crate.
- [`diffsol::FaerSparseLU`](https://docs.rs/diffsol/latest/diffsol/struct.FaerSparseLU.html): A sparse LU decomposition solver using the `faer` crate.

Each solver can be created directly, but it is generally easier to use the methods on the [`OdeSolverProblem`](https://docs.rs/diffsol/latest/diffsol/ode_solver/problem/struct.OdeSolverProblem.html) struct to create the solver.
For example:

```rust,ignore
{{#include ../../../examples/intro-logistic-closures/src/create_solvers.rs}}
```

