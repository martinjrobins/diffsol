# Forward Sensitivities

The [`SensitivitiesOdeSolverMethod`](https://docs.rs/diffsol/latest/diffsol/ode_solver/sensitivities/trait.SensitivitiesOdeSolverMethod.html) trait provides a way to compute the forward sensitivities of the solution of an ODE problem, using its `solve_dense_sensitivities` method.

This method computes both the solution and the sensitivities at the same time, at the time-points provided by the user. These are returned as a tuple containing:

- A matrix of the solution at each time-point, where each column corresponds to a time-point and each row corresponds to a state variable.
- a `Vec` of matrices containing the sensitivities at each time-point. Each element of the outer vector corresponds to the sensitivities for each parameter, and each column of the inner matrix corresponds to a time-point. 

```rust,ignore
{{#include ../../../examples/intro-logistic-closures/src/solve_fwd_sens.rs}}
```