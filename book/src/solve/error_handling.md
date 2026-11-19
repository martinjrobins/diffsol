# Error handling

All the diffsol methods that can fail return a `Result<_, DiffsolError>`. [`DiffsolError`](https://docs.rs/diffsol/latest/diffsol/error/enum.DiffsolError.html) implements `std::error::Error`, so you can use the `?` operator in a function that returns a `Result<_, Box<dyn std::error::Error>>`.

`DiffsolError` has these variants:

| Variant | Cause |
| --- | --- |
| [`OdeSolverError`](https://docs.rs/diffsol/latest/diffsol/error/enum.DiffsolError.html#variant.OdeSolverError) | The ODE solver failed, or you used it incorrectly. It holds an [`OdeSolverError`](https://docs.rs/diffsol/latest/diffsol/error/enum.OdeSolverError.html), see below. |
| [`NonLinearSolverError`](https://docs.rs/diffsol/latest/diffsol/error/enum.DiffsolError.html#variant.NonLinearSolverError) | A Newton solve failed, for example the initial condition solve (`InitialConditionDidNotConverge`). |
| [`LaError`](https://docs.rs/diffsol/latest/diffsol/error/enum.DiffsolError.html#variant.LaError) | A linear algebra operation failed, for example an LU factorisation. |
| [`DiffslParserError`](https://docs.rs/diffsol/latest/diffsol/error/enum.DiffsolError.html#variant.DiffslParserError), [`DiffslCompilerError`](https://docs.rs/diffsol/latest/diffsol/error/enum.DiffsolError.html#variant.DiffslCompilerError) | The DiffSL code has an error. |
| [`Other`](https://docs.rs/diffsol/latest/diffsol/error/enum.DiffsolError.html#variant.Other) | Any other error. |

## Errors during a solve

These `OdeSolverError` variants tell you that the solver could not continue. The [Solver options](../solver/options.md) page gives the options that control each limit.

- **`TooManyTimesteps`**: the solver took `max_number_timesteps` steps (default 5000). Increase the limit. If the problem is stiff, use an implicit solver, which takes fewer steps.
- **`StepSizeTooSmall`**: the step size became smaller than `min_timestep`. The solution possibly has a singularity or a discontinuity. For a singularity, check the equations and the parameters. For a discontinuity, stop the solver at the discontinuity, then continue the solve (see [Discrete Events](../primer/discrete_events.md)).
- **`TooManyErrorTestFailures`**: one step was rejected more than `max_error_test_failures` times. Frequent causes are discontinuities, or functions that are not smooth, in the solution or in the right-hand side. Stop the solver at each discontinuity, then continue the solve.
- **`TooManyNonlinearSolverFailures`**: the Newton solve failed more than `max_nonlinear_solver_failures` times in one step. Check the Jacobian. An incorrect Jacobian is a frequent cause of Newton failures.

Each of these errors gives the time `time` at which the solver stopped.

## Errors from incorrect use

Other `OdeSolverError` variants tell you that the solver was used incorrectly. Some examples are:

- `StopTimeBeforeCurrentTime` and `StopTimeAtCurrentTime`: the stop time must be after the current time.
- `InvalidTEval`: the times given to `solve_dense` must increase, and they must not be before the current time.
- `InterpolationTimeAfterCurrentTime` and `InterpolationTimeOutsideCurrentStep`: you can interpolate only inside the last step.
- `MassMatrixNotSupported`: the explicit Runge-Kutta solver does not support a mass matrix.
- `FailedToGetMutableReference`: a solver still borrows the problem. Drop the solver, or call `take_state()`, before you change the problem.

## Example: continue after too many steps

A solver keeps the state of its last step after a `TooManyTimesteps` error. In this example, the limit is too small. The code matches on the error, increases the limit with `config_mut()`, and continues the solve from the last step:

```rust,ignore
{{#include ../../../examples/intro-logistic-closures/src/solve_error_handling.rs}}
```
