# Initialisation

Before you can solve the problem, you need to generate an intitial state for the solution. DiffSol uses the [`OdeSolverState`](https://docs.rs/diffsol/latest/diffsol/ode_solver/method/struct.OdeSolverState.html)
struct to hold the current state of the solution, this is a struct that contains the state vector, the gradient of the state vector, the time, and the current step size.

You can create a new state for an ODE problem using the [`OdeSolverState::new`](https://docs.rs/diffsol/latest/diffsol/ode_solver/method/struct.OdeSolverState.html#method.new) method,
which takes as arguments the problem and solver instances. 
This method uses the \\(y_0(p, t)\\) closure to generate an intial state vector, and the \\(f(y, p, t)\\) closure to generate the gradient of the state vector. It will also set the time to the initial time
given by the `OdeSolverProblem` struct, and will guess a suitable step size based on the initial state vector and the gradient of the state vector. If you want to set the step size manually or have
more control over the initialisation of the state, you can use the [`OdeSolverState::new_without_initialise`](https://docs.rs/diffsol/latest/diffsol/ode_solver/method/struct.OdeSolverState.html#method.new_without_initialise) method.

Once the state is created then you can use the state and the problem to initialise the solver in preparation for solving the problem.

```rust
# use diffsol::OdeBuilder;
# use nalgebra::DVector;
# type M = nalgebra::DMatrix<f64>;
use diffsol::{OdeSolverState, OdeSolverMethod, Bdf};

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

