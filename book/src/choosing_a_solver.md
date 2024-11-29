# Choosing a solver

Once you have defined the problem, you need to create a solver to solve the problem. The available solvers are:
- [`diffsol::Bdf`](https://docs.rs/diffsol/latest/diffsol/ode_solver/bdf/struct.Bdf.html): A Backwards Difference Formulae solver, suitable for stiff problems and singular mass matrices.
- [`diffsol::Sdirk`](https://docs.rs/diffsol/latest/diffsol/ode_solver/sdirk/struct.Sdirk.html) A Singly Diagonally Implicit Runge-Kutta (SDIRK or ESDIRK) solver. You can define your own butcher tableau using [`Tableau`](https://docs.rs/diffsol/latest/diffsol/ode_solver/tableau/struct.Tableau.html) or use one of the pre-defined tableaues.

For each solver, you will need to specify the linear solver type to use. The available linear solvers are:
- [`diffsol::NalgebraLU`](https://docs.rs/diffsol/latest/diffsol/linear_solver/nalgebra_lu/struct.NalgebraLU.html): A LU decomposition solver using the [nalgebra](https://nalgebra.org) crate.
- [`diffsol::FaerLU`](https://docs.rs/diffsol/latest/diffsol/linear_solver/faer_lu/struct.FaerLU.html): A LU decomposition solver using the [faer](https://github.com/sarah-ek/faer-rs) crate.
- [`diffsol::FaerSparseLU`](https://docs.rs/diffsol/latest/diffsol/linear_solver/faer_sparse_lu/struct.FaerSparseLU.html): A sparse LU decomposition solver using the `faer` crate.

Each solver can be created directly, but it generally easier to use the methods on the [`OdeSolverProblem`](https://docs.rs/diffsol/latest/diffsol/ode_solver/problem/struct.OdeSolverProblem.html) struct to create the solver.
For example:

```rust
# use diffsol::OdeBuilder;
# use nalgebra::DVector;
use diffsol::{OdeSolverState, NalgebraLU, BdfState, Tableau, SdirkState};
# type M = nalgebra::DMatrix<f64>;
type LS = NalgebraLU<f64>;
# fn main() {
# 
#   let problem = OdeBuilder::<M>::new()
#     .p(vec![1.0, 10.0])
#     .rhs_implicit(
#        |x, p, _t, y| y[0] = p[0] * x[0] * (1.0 - x[0] / p[1]),
#        |x, p, _t, v , y| y[0] = p[0] * v[0] * (1.0 - 2.0 * x[0] / p[1]),
#     )
#     .init(|_p, _t| DVector::from_element(1, 0.1))
#     .build()
#     .unwrap();
// Create a BDF solver with an initial state
let solver = problem.bdf::<LS>();

// Create a non-initialised state and manually set the values before
// creating the solver
let state = BdfState::new_without_initialise(&problem).unwrap();
// ... set the state values manually
let solver = problem.bdf_solver::<LS>(state);

// Create a SDIRK solver with a pre-defined tableau
let tableau = Tableau::<M>::tr_bdf2();
let state = problem.sdirk_state::<LS, _>(&tableau).unwrap();
let solver = problem.sdirk_solver::<LS, _>(state, tableau);

// Create a tr_bdf2 or esdirk34 solvers directly (both are SDIRK solvers with different tableaus)
let solver = problem.tr_bdf2::<LS>();
let solver = problem.esdirk34::<LS>();

// Create a non-initialised state and manually set the values before
// creating the solver
let state = SdirkState::new_without_initialise(&problem).unwrap();
// ... set the state values manually
let solver = problem.tr_bdf2_solver::<LS>(state);
# }
```

# Initialisation

Each solver has an internal state that holds information like the current state vector, the gradient of the state vector, the current time, and the current step size. When you create a solver using the `bdf` or `sdirk` methods on the `OdeSolverProblem` struct, the solver will be initialised with an initial state based on the initial conditions of the problem as well as satisfying any algebraic constraints. An initial time step will also be chosen based on your provided equations.

Each solver's state struct implements the [`OdeSolverState`](https://docs.rs/diffsol/latest/diffsol/ode_solver/state/trait.OdeSolverState.html) trait, and if you wish to manually create and setup a state, you can use the methods on this trait to do so.

For example, say that you wish to bypass the initialisation of the state as you already have the algebraic constraints and so don't need to solve for them. You can use the `new_without_initialise` method on the `OdeSolverState` trait to create a new state without initialising it. You can then use the `as_mut` method to get a mutable reference to the state and set the values manually.

Note that each state struct has a [`as_ref`](https://docs.rs/diffsol/latest/diffsol/ode_solver/state/trait.OdeSolverState.html#tymethod.as_ref) and [`as_mut`](https://docs.rs/diffsol/latest/diffsol/ode_solver/state/trait.OdeSolverState.html#tymethod.as_mut) methods that return a [`StateRef`](https://docs.rs/diffsol/latest/diffsol/ode_solver/state/struct.StateRef.html) or ['StateRefMut`](https://docs.rs/diffsol/latest/diffsol/ode_solver/state/struct.StateRefMut.html) struct respectively. These structs provide a solver-independent way to access the state values so you can use the same code with different solvers.

```rust
# use diffsol::OdeBuilder;
# use nalgebra::DVector;
# type M = nalgebra::DMatrix<f64>;
use diffsol::{OdeSolverState, NalgebraLU, BdfState};
type LS = NalgebraLU<f64>;

# fn main() {
# 
#   let problem = OdeBuilder::<M>::new()
#     .p(vec![1.0, 10.0])
#     .rhs_implicit(
#        |x, p, _t, y| y[0] = p[0] * x[0] * (1.0 - x[0] / p[1]),
#        |x, p, _t, v , y| y[0] = p[0] * v[0] * (1.0 - 2.0 * x[0] / p[1]),
#     )
#     .init(|_p, _t| DVector::from_element(1, 0.1))
#     .build()
#     .unwrap();
let mut state = BdfState::new_without_initialise(&problem).unwrap();
state.as_mut().y[0] = 0.1;
let mut solver = problem.bdf_solver::<LS>(state);
# }
```