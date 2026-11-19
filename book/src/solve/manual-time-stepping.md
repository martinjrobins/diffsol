# Manual time-stepping

The fundamental method to step the solver through a solution is the [`step`](https://docs.rs/diffsol/latest/diffsol/ode_solver/method/trait.OdeSolverMethod.html#tymethod.step) method on the `OdeSolverMethod` trait, which steps the solution forward in time by a single step, with a step size chosen by the solver in order to satisfy the error tolerances in the `problem` struct. The `step` method returns a `Result<OdeSolverStopReason<T>, DiffsolError>`. If the step is successful, the `OdeSolverStopReason` tells you why the step stopped:

- `InternalTimestep`: the solver took a normal internal step.
- `RootFound(t, idx)`: the root function with index `idx` changed sign at time `t`.
- `TstopReached`: the solver reached the stop time set with `set_stop_time`.

The `step` method does not return the new state. Use `solver.state()` to get the current time, state vector and other state values after the step. See [Stopping](./stopping.md) for how to use the stop reason.

```rust,ignore
{{#include ../../../examples/intro-logistic-closures/src/solve_step.rs}}
```

The `step` method returns an error if:

- the solver has already taken `ode_options.max_number_timesteps` steps (`TooManyTimesteps`). The default is 5000. The step count is not reset between calls, so successive `solve*` calls with the same solver share this limit.
- the step size becomes smaller than `ode_options.min_timestep` (`StepSizeTooSmall`).
- the error test fails more than `ode_options.max_error_test_failures` times in one step (`TooManyErrorTestFailures`).
- the nonlinear solver fails to converge more than `ode_options.max_nonlinear_solver_failures` times in one step (`TooManyNonlinearSolverFailures`, implicit solvers only).

See [Error handling](./error_handling.md) for how to handle these errors, and [Solver options](../solver/options.md) for how to change the limits.
