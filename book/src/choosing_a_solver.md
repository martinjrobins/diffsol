# Choosing a solver

Once you have defined the problem, you need to create a solver to solve the problem. The available solvers are:
- [`diffsol::Bdf`](https://docs.rs/diffsol/latest/diffsol/ode_solver/bdf/struct.Bdf.html): A Backwards Difference Formulae solver, suitable for stiff problems and singular mass matrices.
- [`diffsol::Sdirk`](https://docs.rs/diffsol/latest/diffsol/ode_solver/sdirk/struct.Sdirk.html) A Singly Diagonally Implicit Runge-Kutta (SDIRK or ESDIRK) solver. You can define your own butcher tableau using [`Tableau`](https://docs.rs/diffsol/latest/diffsol/ode_solver/tableau/struct.Tableau.html) or use one of the pre-defined tableaues.
    

Each of these solvers has a number of generic arguments, for example the `Bdf` solver has three generic arguments:
- `M`: The matrix type used to define the problem.
- `Eqn`: The type of the equations struct that defines the problem.
- `Nls`: The type of the non-linear solver used to solve the implicit equations in the solver.

In normal use cases, Rust can infer these from your code so you don't need to specify these explicitly. The `Bdf` solver implements the `Default` trait so can be easily created using:

```rust
# use diffsol::OdeBuilder;
# use nalgebra::DVector;
# type M = nalgebra::DMatrix<f64>;
use diffsol::{Bdf, OdeSolverState, OdeSolverMethod};
# fn main() {
# 
#   let problem = OdeBuilder::new()
#     .p(vec![1.0, 10.0])
#     .build_ode::<M, _, _, _>(
#        |x, p, _t, y| y[0] = p[0] * x[0] * (1.0 - x[0] / p[1]),
#        |x, p, _t, v , y| y[0] = p[0] * v[0] * (1.0 - 2.0 * x[0] / p[1]),
#        |_p, _t| DVector::from_element(1, 0.1),
#     ).unwrap();
let mut solver = Bdf::default();
let state = OdeSolverState::new(&problem, &solver).unwrap();
solver.set_problem(state, &problem);
# }
```

The `Sdirk` solver requires a tableu to be specified so you can use its `new` method to create a new solver, for example using the `tr_bdf2` tableau:

```rust
# use diffsol::{OdeBuilder};
# use nalgebra::DVector;
# type M = nalgebra::DMatrix<f64>;
use diffsol::{Sdirk, Tableau, NalgebraLU, OdeSolverState, OdeSolverMethod};
# fn main() {
#   let problem = OdeBuilder::new()
#     .p(vec![1.0, 10.0])
#     .build_ode::<M, _, _, _>(
#        |x, p, _t, y| y[0] = p[0] * x[0] * (1.0 - x[0] / p[1]),
#        |x, p, _t, v , y| y[0] = p[0] * v[0] * (1.0 - 2.0 * x[0] / p[1]),
#        |_p, _t| DVector::from_element(1, 0.1),
#     ).unwrap();
let mut solver = Sdirk::new(Tableau::<M>::tr_bdf2(), NalgebraLU::default());
let state = OdeSolverState::new(&problem, &solver).unwrap();
solver.set_problem(state, &problem);
# }
```

You can also use one of the helper functions to create a SDIRK solver with a pre-defined tableau, which will create it with the default linear solver:

```rust
# use diffsol::{OdeBuilder};
# use nalgebra::DVector;
# type M = nalgebra::DMatrix<f64>;
use diffsol::{Sdirk, Tableau, NalgebraLU, OdeSolverState, OdeSolverMethod};
# fn main() {
#   let problem = OdeBuilder::new()
#     .p(vec![1.0, 10.0])
#     .build_ode::<M, _, _, _>(
#        |x, p, _t, y| y[0] = p[0] * x[0] * (1.0 - x[0] / p[1]),
#        |x, p, _t, v , y| y[0] = p[0] * v[0] * (1.0 - 2.0 * x[0] / p[1]),
#        |_p, _t| DVector::from_element(1, 0.1),
#     ).unwrap();
let mut solver = Sdirk::tr_bdf2();
let state = OdeSolverState::new(&problem, &solver).unwrap();
solver.set_problem(state, &problem);
# }
```


