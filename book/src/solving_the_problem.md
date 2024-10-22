# Solving the Problem

Each solver implements the [`OdeSolverMethod`](https://docs.rs/diffsol/latest/diffsol/ode_solver/method/trait.OdeSolverMethod.html) trait, which provides a number of methods to solve the problem.
The fundamental method to solve the problem is the `step` method on the `OdeSolverMethod` trait, which steps the solution forward in time by a single step, with a step size chosen by the solver
in order to satisfy the error tolerances in the `problem` struct. The `step` method returns a `Result` that contains the new state of the solution if the step was successful, or an error if the step failed.

```rust
# use diffsol::OdeBuilder;
# use nalgebra::DVector;
# type M = nalgebra::DMatrix<f64>;
use diffsol::{OdeSolverMethod, OdeSolverState, Bdf};

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
while solver.state().unwrap().t < 10.0 {
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
use diffsol::{OdeSolverMethod, OdeSolverState, Bdf};

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
let t_o = 10.0;
while solver.state().unwrap().t < t_o {
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
use diffsol::{OdeSolverMethod, OdeSolverStopReason, OdeSolverState, Bdf};

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
solver.set_stop_time(10.0).unwrap();
loop {
    match solver.step() {
        Ok(OdeSolverStopReason::InternalTimestep) => continue,
        Ok(OdeSolverStopReason::TstopReached) => break,
        Ok(OdeSolverStopReason::RootFound(_)) => panic!("Root finding not used"),
        Err(e) => panic!("Solver failed to converge: {}", e),
    }
}
let _soln = &solver.state().unwrap().y;
# }
```

DiffSol also has two convenience functions `solve` and `solve_dense` on the `OdeSolverMethod` trait. `solve` solve the problem from an initial state up to a specified time, returning the solution at all the 
internal timesteps used by the solver. This function returns a tuple that contains a `Vec` of 
the solution at each timestep, and a `Vec` of the times at each timestep.

```rust
# use diffsol::OdeBuilder;
# use nalgebra::DVector;
# type M = nalgebra::DMatrix<f64>;
use diffsol::{OdeSolverMethod, Bdf, OdeSolverState};

# fn main() {
#   let problem = OdeBuilder::new()
#     .p(vec![1.0, 10.0])
#     .build_ode::<M, _, _, _>(
#        |x, p, _t, y| y[0] = p[0] * x[0] * (1.0 - x[0] / p[1]),
#        |x, p, _t, v , y| y[0] = p[0] * v[0] * (1.0 - 2.0 * x[0] / p[1]),
#        |_p, _t| DVector::from_element(1, 0.1),
#     ).unwrap();
let mut solver = Bdf::default();
let state = OdeSolverState::new(&problem, &solver).unwrap();
let (ys, ts) = solver.solve(&problem, state, 10.0).unwrap();
# }
```

`solve_dense` will solve a problem from an initial state, returning the solution at a `Vec` of times provided by the user. This function returns a `Vec<V>`, where `V` is the vector type used to define the problem.

```rust
# use diffsol::OdeBuilder;
# use nalgebra::DVector;
# type M = nalgebra::DMatrix<f64>;
use diffsol::{OdeSolverMethod, Bdf, OdeSolverState};

# fn main() {
#   let problem = OdeBuilder::new()
#     .p(vec![1.0, 10.0])
#     .build_ode::<M, _, _, _>(
#        |x, p, _t, y| y[0] = p[0] * x[0] * (1.0 - x[0] / p[1]),
#        |x, p, _t, v , y| y[0] = p[0] * v[0] * (1.0 - 2.0 * x[0] / p[1]),
#        |_p, _t| DVector::from_element(1, 0.1),
#     ).unwrap();
let mut solver = Bdf::default();
let times = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
let state = OdeSolverState::new(&problem, &solver).unwrap();
let _soln = solver.solve_dense(&problem, state, &times).unwrap();
# }
```