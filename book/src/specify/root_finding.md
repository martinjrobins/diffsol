# Root finding

Root finding is the process of finding the values of the variables that make a set of equations equal to zero. This is a common problem where you want to
stop the solver or perform some action when a certain condition is met. 

## Specifying the root finding function

Using the logistic example, we can add a root finding function \\(r(y, p, t)\\) that will stop the solver when the value of \\(y\\) is such that \\(r(y, p, t) = 0\\).
For this example we'll use the root finding function \\(r(y, p, t) = y - 0.5\\), which will stop the solver when the value of \\(y\\) is 0.5.


\\[\frac{dy}{dt} = r y (1 - y/K),\\] 
\\[r(y, p, t) = y - 0.5,\\]

This can be done using the [`OdeBuilder`](https://docs.rs/diffsol/latest/diffsol/ode_solver/builder/struct.OdeBuilder.html) via the following code:

```rust
# fn main() {
use diffsol::OdeBuilder;
use nalgebra::DVector;
type M = nalgebra::DMatrix<f64>;

let problem = OdeBuilder::<M>::new()
    .t0(0.0)
    .rtol(1e-6)
    .atol([1e-6])
    .p(vec![1.0, 10.0])
    .rhs_implicit(
       |x, p, _t, y| y[0] = p[0] * x[0] * (1.0 - x[0] / p[1]),
       |x, p, _t, v , y| y[0] = p[0] * v[0] * (1.0 - 2.0 * x[0] / p[1]),
    )
    .init(|_p, _t| DVector::from_element(1, 0.1))
    .root(|x, _p, _t, y| y[0] = x[0] - 0.5, 1)
    .build()
    .unwrap();
# }
```

here we have added the root finding function \\(r(y, p, t) = y - 0.5\\), and also let DiffSol know that we have one root function by passing `1` as the last argument to the `root` method.
If we had specified more than one root function, the solver would stop when any of the root functions are zero.

## Detecting roots during the solve

To detect the root during the solve, we can use the return type on the [`step`](https://docs.rs/diffsol/latest/diffsol/ode_solver/method/trait.OdeSolverMethod.html#tymethod.step) method of the solver. 
If successful the `step` method returns an [`OdeSolverStopReason`](https://docs.rs/diffsol/latest/diffsol/ode_solver/method/enum.OdeSolverStopReason.html) enum that contains the reason the solver stopped.


```rust
# fn main() {
# use diffsol::OdeBuilder;
# use nalgebra::DVector;
# type M = nalgebra::DMatrix<f64>;
use diffsol::{OdeSolverMethod, OdeSolverStopReason, NalgebraLU};
type LS = NalgebraLU<f64>;

# let problem = OdeBuilder::<M>::new()
#     .p(vec![1.0, 10.0])
#     .rhs_implicit(
#        |x, p, _t, y| y[0] = p[0] * x[0] * (1.0 - x[0] / p[1]),
#        |x, p, _t, v , y| y[0] = p[0] * v[0] * (1.0 - 2.0 * x[0] / p[1]),
#     )
#     .init(|_p, _t| DVector::from_element(1, 0.1))
#     .root(|x, _p, _t, y| y[0] = x[0] - 0.5, 1)
#     .build()
#     .unwrap();
let mut solver = problem.bdf::<LS>().unwrap();
let t = loop {
    match solver.step() {
        Ok(OdeSolverStopReason::InternalTimestep) => continue,
        Ok(OdeSolverStopReason::TstopReached) => panic!("We didn't set a stop time"),
        Ok(OdeSolverStopReason::RootFound(t)) => break t,
        Err(e) => panic!("Solver failed to converge: {}", e),
    }
};
println!("Root found at t = {}", t);
let _soln = &solver.state().y;
# }
```

