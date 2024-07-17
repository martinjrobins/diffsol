# Solution of ODEs

## Specifying the ODE problem

The simplest way to create a new ode problem in Rust is to use the [`OdeBuilder`](https://docs.rs/diffsol/latest/diffsol/ode_solver/builder/struct.OdeBuilder.html) struct. 
You can set the initial time, initial step size, relative tolerance, absolute tolerance, and parameters, or leave them at their default values. 
Then, call one of the `build_*` functions to create a new problem, for example the [`build_ode`](https://docs.rs/diffsol/latest/diffsol/ode_solver/builder/struct.OdeBuilder.html#method.build_ode)
function can be used to create an ODE problem of the form \\(dy/dt = f(t, y, p)\\), where \\(y\\) is the state vector, \\(t\\) is the time, and \\(p\\) are the parameters.

Below is an example of how to create a new ODE problem using the `OdeBuilder` struct. 
The specific problem we will solve is the logistic equation 

\\[dy/dt = r y (1 - y/K),\\] 

where \\(r\\) is the growth rate and \\(K\\) is the carrying capacity. 
To specify the problem, we need to provide the \\(dy/dt\\) function \\(f(y, p, t)\\), 
the jacobian of \\(f(y, p, t)\\) multiplied by a vector \\(v\\) function, which is

\\[f'(y, p, t, v) = rv (1 - 2y/K),\\]

and the initial state 

\\[y_0(p, t) = 0.1\\]

This can be done using the following code:

```rust
# fn main() {
use diffsol::OdeBuilder;
use nalgebra::DVector;
type M = nalgebra::DMatrix<f64>;

let problem = OdeBuilder::new()
    .t0(0.0)
    .rtol(1e-6)
    .atol([1e-6])
    .p(vec![1.0, 10.0])
    .build_ode::<M, _, _, _>(
       |x, p, _t, y| y[0] = p[0] * y[0] * (1.0 - y[0] / p[1]),
       |x, p, _t, v , y| y[0] = p[0] * v[0] * (1.0 - 2.0 * y[0] / p[1]),
       |_p, _t| DVector::from_element(1, 0.1),
    ).unwrap();
# }
```

Each `build_*` method requires the user to specify what matrix type they wish to use to define and solve the model (the other types are inferred from the closure types). 
Here we use the `nalgebra::DMatrix<f64>` type, which is a dense matrix type from the [nalgebra](https://nalgebra.org) crate. Other options are:
- `faer::Mat<T>` from [faer](https://github.com/sarah-ek/faer-rs), which is a dense matrix type.
- `diffsol::SparseColMat<T>`, which is a thin wrapper around `faer::sparse::SparseColMat<T>`, a sparse compressed sparse column matrix type.
    
Each of these matrix types have an associated vector type that is used to represent the vectors in the problem (i.e. the state vector \\(y\\), the parameter vector \\(p\\), and the gradient vector \\(v\\)).
You can see in the example above that the `DVector` type is explicitly used to create the initial state vector in the third closure.
For these matrix types the associated vector type is:
- `nalgebra::DVector<T>` for `nalgebra::DMatrix<T>`.
- `faer::Col<T>` for `faer::Mat<T>`.
- `faer::Coll<T>` for `diffsol::SparseColMat<T>`.

The arguments to the `build_ode` method are the equations that define the problem. 
The first closure is the function \\(f(y, p, t)\\) this is implemented as a closure that takes the time `t`, 
the parameter vector `p`, the state vector `y`, and a mutable reference that the closure can use to place the result (i.e. the derivative of the state vector \\(f(y, p, t)\\)).
The second closure is similar in structure in defines the jacobian multiplied by a vector \\(v\\) function \\(f'(y, p, t, v)\\).
The third closure returns the initial state vector \\(y_0(p, t)\\), this is done so that diffsol can infer the size of the state vector.


## Choosing a solver

Once you have defined the problem, you need to create a solver to solve the problem. The available solvers are:
    - [`diffsol::Bdf`](https://docs.rs/diffsol/latest/diffsol/ode_solver/bdf/struct.Bdf.html): A Backwards Difference Formulae solver, suitable for stiff problems and singular mass matrices.
    - [`diffsol::Sdirk`](https://docs.rs/diffsol/latest/diffsol/ode_solver/sdirk/struct.Sdirk.html) A Singly Diagonally Implicit Runge-Kutta (SDIRK or ESDIRK) solver. You can define your own butcher tableau using [`Tableau`](https://docs.rs/diffsol/latest/diffsol/ode_solver/tableau/struct.Tableau.html) or use one of the pre-defined tableaues.
    

Each of these solvers has a number of generic arguments, for example the `Bdf` solver has three generic arguments:
- `M`: The matrix type used to define the problem.
- `Eqn`: The type of the equations struct that defines the problem.
- `Nls`: The type of the non-linear solver used to solve the implicit equations in the solver.

In normal use cases, Rust can infer these from your code so you don't need to specify these explicitly. The `Bdf` solver implements the `Default` trait so can be easily created using:

```rust
# fn main() {
# use diffsol::{OdeBuilder, OdeSolverState};
# use nalgebra::DVector;
# type M = nalgebra::DMatrix<f64>;
# 
# let problem = OdeBuilder::new()
#     .p(vec![1.0, 10.0])
#     .build_ode::<M, _, _, _>(
#        |x, p, _t, y| y[0] = p[0] * x[0] * (1.0 - x[0] / p[1]),
#        |x, p, _t, v , y| y[0] = p[0] * v[0] * (1.0 - 2.0 * x[0] / p[1]),
#        |_p, _t| DVector::from_element(1, 0.1),
#     ).unwrap();
use diffsol::Bdf;
let solver = Bdf::default();
# let _state = OdeSolverState::new(&problem, &solver).unwrap();
# }
```

The `Sdirk` solver requires a tableu to be specified so you can use its `new` method to create a new solver, for example using the `tr_bdf2` tableau:

```rust
# fn main() {
# use diffsol::{OdeBuilder, OdeSolverState};
# use nalgebra::DVector;
# type M = nalgebra::DMatrix<f64>;
# 
# let problem = OdeBuilder::new()
#     .p(vec![1.0, 10.0])
#     .build_ode::<M, _, _, _>(
#        |x, p, _t, y| y[0] = p[0] * x[0] * (1.0 - x[0] / p[1]),
#        |x, p, _t, v , y| y[0] = p[0] * v[0] * (1.0 - 2.0 * x[0] / p[1]),
#        |_p, _t| DVector::from_element(1, 0.1),
#     ).unwrap();
use diffsol::{Sdirk, Tableau, NalgebraLU};
let solver = Sdirk::new(Tableau::<M>::tr_bdf2(), NalgebraLU::default());
# let _state = OdeSolverState::new(&problem, &solver).unwrap();
# }
```

## The initial state

Before you can solve the problem, you need to generate an intitial state for the solution. DiffSol uses the [`OdeSolverState`](https://docs.rs/diffsol/latest/diffsol/ode_solver/method/struct.OdeSolverState.html)
struct to hold the current state of the solution, this is a struct that contains the state vector, the gradient of the state vector, the time, and the current step size.

You can create a new state for an ODE problem using the [`OdeSolverState::new`](https://docs.rs/diffsol/latest/diffsol/ode_solver/method/struct.OdeSolverState.html#method.new) method,
which takes as arguments the problem and solver instances. 
This method uses the \\(y_0(p, t)\\) closure to generate an intial state vector, and the \\(f(y, p, t)\\) closure to generate the gradient of the state vector. It will also set the time to the initial time
given by the `OdeSolverProblem` struct, and will guess a suitable step size based on the initial state vector and the gradient of the state vector. If you want to set the step size manually or have
more control over the initialisation of the state, you can use the [`OdeSolverState::new_without_initialise`](https://docs.rs/diffsol/latest/diffsol/ode_solver/method/struct.OdeSolverState.html#method.new_without_initialise) method.

Once the state is created then you can use the state and the problem to initialise the solver in preparation for solving the problem.

```rust
# fn main() {
# use diffsol::OdeBuilder;
# use nalgebra::DVector;
# type M = nalgebra::DMatrix<f64>;
# 
# let problem = OdeBuilder::new()
#     .p(vec![1.0, 10.0])
#     .build_ode::<M, _, _, _>(
#        |x, p, _t, y| y[0] = p[0] * x[0] * (1.0 - x[0] / p[1]),
#        |x, p, _t, v , y| y[0] = p[0] * v[0] * (1.0 - 2.0 * x[0] / p[1]),
#        |_p, _t| DVector::from_element(1, 0.1),
#     ).unwrap();
use diffsol::{OdeSolverState, OdeSolverMethod, Bdf};
let mut solver = Bdf::default();
let state = OdeSolverState::new(&problem, &solver).unwrap();
solver.set_problem(state, &problem);
# }
```

## Solving the problem

Each solver implements the [`OdeSolverMethod`](https://docs.rs/diffsol/latest/diffsol/ode_solver/method/trait.OdeSolverMethod.html) trait, which provides a number of methods to solve the problem.
The fundamental method to solve the problem is the `step` method on the `OdeSolverMethod` trait, which steps the solution forward in time by a single step, with a step size chosen by the solver
in order to satisfy the error tolerances in the `problem` struct. The `step` method returns a `Result` that contains the new state of the solution if the step was successful, or an error if the step failed.

```rust
# fn main() {
# use diffsol::OdeBuilder;
# use nalgebra::DVector;
# type M = nalgebra::DMatrix<f64>;
# 
# let problem = OdeBuilder::new()
#     .p(vec![1.0, 10.0])
#     .build_ode::<M, _, _, _>(
#        |x, p, _t, y| y[0] = p[0] * x[0] * (1.0 - x[0] / p[1]),
#        |x, p, _t, v , y| y[0] = p[0] * v[0] * (1.0 - 2.0 * x[0] / p[1]),
#        |_p, _t| DVector::from_element(1, 0.1),
#     ).unwrap();
use diffsol::{OdeSolverMethod, OdeSolverState, Bdf};
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
# fn main() {
# use diffsol::OdeBuilder;
# use nalgebra::DVector;
# type M = nalgebra::DMatrix<f64>;
# 
# let problem = OdeBuilder::new()
#     .p(vec![1.0, 10.0])
#     .build_ode::<M, _, _, _>(
#        |x, p, _t, y| y[0] = p[0] * x[0] * (1.0 - x[0] / p[1]),
#        |x, p, _t, v , y| y[0] = p[0] * v[0] * (1.0 - 2.0 * x[0] / p[1]),
#        |_p, _t| DVector::from_element(1, 0.1),
#     ).unwrap();
use diffsol::{OdeSolverMethod, OdeSolverState, Bdf};
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

```rust
# fn main() {
# use diffsol::OdeBuilder;
# use nalgebra::DVector;
# type M = nalgebra::DMatrix<f64>;
# 
# let problem = OdeBuilder::new()
#     .p(vec![1.0, 10.0])
#     .build_ode::<M, _, _, _>(
#        |x, p, _t, y| y[0] = p[0] * x[0] * (1.0 - x[0] / p[1]),
#        |x, p, _t, v , y| y[0] = p[0] * v[0] * (1.0 - 2.0 * x[0] / p[1]),
#        |_p, _t| DVector::from_element(1, 0.1),
#     ).unwrap();
use diffsol::{OdeSolverMethod, OdeSolverStopReason, OdeSolverState, Bdf};
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

DiffSol also has two convenience functions `solve` and `solve_dense` on the `OdeSolverMethod` trait. `solve` will initialise the problem and solve the problem up to a specified time, returning the solution at all the 
internal timesteps used by the solver. This function returns a [`OdeSolution`](https://docs.rs/diffsol/latest/diffsol/ode_solver/solution/struct.OdeSolution.html) struct that contains both a `Vec` of 
the solution at each timestep, and a `Vec` of the times at each timestep.

```rust
# fn main() {
# use diffsol::OdeBuilder;
# use nalgebra::DVector;
# type M = nalgebra::DMatrix<f64>;
# 
# let problem = OdeBuilder::new()
#     .p(vec![1.0, 10.0])
#     .build_ode::<M, _, _, _>(
#        |x, p, _t, y| y[0] = p[0] * x[0] * (1.0 - x[0] / p[1]),
#        |x, p, _t, v , y| y[0] = p[0] * v[0] * (1.0 - 2.0 * x[0] / p[1]),
#        |_p, _t| DVector::from_element(1, 0.1),
#     ).unwrap();
use diffsol::{OdeSolverMethod, Bdf};
let mut solver = Bdf::default();
let _soln = solver.solve(&problem, 10.0).unwrap();
# }
```

`solve_dense` will initialise the problem and solve the problem, returning the solution at a `Vec` of times provided by the user. This function returns a `Vec<V>`, where `V` is the vector type used to define the problem.

```rust
# fn main() {
# use diffsol::OdeBuilder;
# use nalgebra::DVector;
# type M = nalgebra::DMatrix<f64>;
# 
# let problem = OdeBuilder::new()
#     .p(vec![1.0, 10.0])
#     .build_ode::<M, _, _, _>(
#        |x, p, _t, y| y[0] = p[0] * x[0] * (1.0 - x[0] / p[1]),
#        |x, p, _t, v , y| y[0] = p[0] * v[0] * (1.0 - 2.0 * x[0] / p[1]),
#        |_p, _t| DVector::from_element(1, 0.1),
#     ).unwrap();
use diffsol::{OdeSolverMethod, Bdf};
let mut solver = Bdf::default();
let times = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
let _soln = solver.solve_dense(&problem, &times).unwrap();
# }
```