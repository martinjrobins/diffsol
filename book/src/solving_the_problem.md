# Solving the Problem

Each solver implements the [`OdeSolverMethod`](https://docs.rs/diffsol/latest/diffsol/ode_solver/method/trait.OdeSolverMethod.html) trait, which provides a number of methods to solve the problem.

## Solving the Problem

DiffSol has a few high-level solution functions on the `OdeSolverMethod` trait that are the easiest way to solve your equations:
- [`solve`](https://docs.rs/diffsol/latest/diffsol/ode_solver/method/trait.OdeSolverMethod.html#method.solve) - solve the problem from an initial state up to a specified time, returning the solution at all the internal timesteps used by the solver.
- [`solve_dense`](https://docs.rs/diffsol/latest/diffsol/ode_solver/method/trait.OdeSolverMethod.html#method.solve_dense) - solve the problem from an initial state, returning the solution at a `Vec` of times provided by the user.
- ['solve_dense_sensitivities`](https://docs.rs/diffsol/latest/diffsol/ode_solver/method/trait.OdeSolverMethod.html#method.solve_dense_sensitivities) - solve the forward sensitivity problem from an initial state, returning the solution at a `Vec` of times provided by the user.
- ['solve_adjoint'](https://docs.rs/diffsol/latest/diffsol/ode_solver/method/trait.OdeSolverMethod.html#method.solve_adjoint) - solve the adjoint sensitivity problem from an initial state to a final time, returning the integration of the output function over time as well as its gradient with respect to the initial state.

The following example shows how to solve a simple ODE problem using the `solve` method on the `OdeSolverMethod` trait. 

```rust
# use diffsol::OdeBuilder;
# use nalgebra::DVector;
use diffsol::{OdeSolverMethod, NalgebraLU};
type M = nalgebra::DMatrix<f64>;
type LS = NalgebraLU<f64>;

# fn main() {
#   let problem = OdeBuilder::<M>::new()
#     .p(vec![1.0, 10.0])
#     .rhs_implicit(
#        |x, p, _t, y| y[0] = p[0] * x[0] * (1.0 - x[0] / p[1]),
#        |x, p, _t, v , y| y[0] = p[0] * v[0] * (1.0 - 2.0 * x[0] / p[1]),
#     )
#     .init(|_p, _t| DVector::from_element(1, 0.1))
#     .build()
#     .unwrap();
let mut solver = problem.bdf::<LS>().unwrap();
let (ys, ts) = solver.solve(10.0).unwrap();
# }
```

`solve_dense` will solve a problem from an initial state, returning the solution at a `Vec` of times provided by the user.

```rust
# use diffsol::OdeBuilder;
# use nalgebra::DVector;
# type M = nalgebra::DMatrix<f64>;
use diffsol::{OdeSolverMethod, NalgebraLU};
type LS = NalgebraLU<f64>;

# fn main() {
#   let problem = OdeBuilder::<M>::new()
#     .p(vec![1.0, 10.0])
#     .rhs_implicit(
#        |x, p, _t, y| y[0] = p[0] * x[0] * (1.0 - x[0] / p[1]),
#        |x, p, _t, v , y| y[0] = p[0] * v[0] * (1.0 - 2.0 * x[0] / p[1]),
#     )
#     .init(|_p, _t| DVector::from_element(1, 0.1))
#     .build()
#     .unwrap();
let mut solver = problem.bdf::<LS>().unwrap();
let times = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
let _soln = solver.solve_dense(&times).unwrap();
# }
```

## Stepping the Solution

The fundamental method to step the solver through a solution is the [`step`](https://docs.rs/diffsol/latest/diffsol/ode_solver/method/trait.OdeSolverMethod.html#tymethod.step) method on the `OdeSolverMethod` trait, which steps the solution forward in time by a single step, with a step size chosen by the solver in order to satisfy the error tolerances in the `problem` struct. The `step` method returns a `Result` that contains the new state of the solution if the step was successful, or an error if the step failed.

```rust
# use diffsol::OdeBuilder;
# use nalgebra::DVector;
# type M = nalgebra::DMatrix<f64>;
use diffsol::{OdeSolverMethod, NalgebraLU};
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
let mut solver = problem.bdf::<LS>().unwrap();
while solver.state().t < 10.0 {
    if let Err(_) = solver.step() {
        break;
    }
}
# }
```

The `step` method will return an error if the solver fails to converge to the solution or if the step size becomes too small.

Often you will want to get the solution at a specific time \\(t_o\\). There are two ways to do this based on your particular needs, the most lightweight way is to step the solution forward
until you are beyond \\(t_o\\), and then interpolate the solution back to \\(t_o\\) using the `interpolate` method on the `OdeSolverMethod` trait. 

```rust
# use diffsol::OdeBuilder;
# use nalgebra::DVector;
# type M = nalgebra::DMatrix<f64>;
use diffsol::{OdeSolverMethod, NalgebraLU};
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
let mut solver = problem.bdf::<LS>().unwrap();
let t_o = 10.0;
while solver.state().t < t_o {
    solver.step().unwrap();
}
let _soln = solver.interpolate(t_o).unwrap();
# }
```

The second way is to use the `set_stop_time` method on the `OdeSolverMethod` trait to stop the solver at a specific time, this will override the internal time step so that the solver stops at the specified time.
Note that this can be less efficient if you wish to continue stepping forward after the specified time, as the solver will need to be re-initialised.
The enum returned by `step` will indicate when the solver has stopped at the specified time.
Once the solver has stopped at the specified time, you can get the current state of the solution using the `state` method on the solver, which returns an [`OdeSolverState`](https://docs.rs/diffsol/latest/diffsol/ode_solver/method/struct.OdeSolverState.html) struct.

```rust
# use diffsol::OdeBuilder;
# use nalgebra::DVector;
# type M = nalgebra::DMatrix<f64>;
use diffsol::{OdeSolverMethod, OdeSolverStopReason, NalgebraLU};
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
let mut solver = problem.bdf::<LS>().unwrap();
solver.set_stop_time(10.0).unwrap();
loop {
    match solver.step() {
        Ok(OdeSolverStopReason::InternalTimestep) => continue,
        Ok(OdeSolverStopReason::TstopReached) => break,
        Ok(OdeSolverStopReason::RootFound(_)) => panic!("Root finding not used"),
        Err(e) => panic!("Solver failed to converge: {}", e),
    }
}
let _soln = &solver.state().y;
# }
```

