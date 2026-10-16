# Manual time-stepping

The fundamental method to step the solver through a solution is the [`step`](https://docs.rs/diffsol/latest/diffsol/ode_solver/method/trait.OdeSolverMethod.html#tymethod.step) method on the `OdeSolverMethod` trait, which steps the solution forward in time by a single step, with a step size chosen by the solver in order to satisfy the error tolerances in the `problem` struct. The `step` method returns a `Result` that contains the new state of the solution if the step was successful, or an error if the step failed.

To use fixed-step operation, call `disable_state_error_control()` when building the problem. Successful steps then keep the current step size without error estimates. A nonlinear solve failure can still make an implicit solver retry with a smaller step. A step may also be shortened to land on a requested stop time; the previous step size is restored after the stop time is reached.

```rust,ignore
{{#include ../../../examples/intro-logistic-closures/src/solve_step.rs}}
```

The `step` method will return an error if the solver fails to converge to the solution or if the step size becomes too small.
