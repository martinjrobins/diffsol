# Stopping

There are two methods to halt the solver when a certain condition is met: you can either stop at a specific time or stop when a certain event occurs.

## Stopping with high-level solve methods

The high-level `solve` and `solve_dense` methods stop the solver when either the final time `tfinal` is reached, or when one of the elements of the root function returns a value that changes sign. The internal state of the solver (`t`, `y`, `dy` etc.) is set to the time that the zero-crossing occured.

If a `reset` function is also defined, `solve` and `solve_dense` apply the reset automatically at the root time and continue integrating to the requested final time or final evaluation point. In `solve_dense`, any user-specified evaluation time that lands exactly on the event receives the post-reset state.

The lower-level `step` method, staged solves via `solve_soln`, and sensitivity solve helpers such as `solve_dense_sensitivities` still report `RootFound` and leave it to the caller to decide how to resume.

## Manual time-stepping

Stopping at a specific time is straightforward, as you can use the `set_stop_time` method on the `OdeSolverMethod` trait and then just check if the return value of the `step` method is `Ok(OdeSolverStopReason::TstopReached)`

Stopping at a certain event requires you to set a root function in your system of equations, see [Root Finding](specify/closure/root_finding.md) for more information. During time-stepping, you can check if the solver has discovered a root by checking if the return value of the `step` method is `Ok(OdeSolverStopReason::RootFound)`. `RootFound` holds the time at which the root was found, and the index of the root that was found. You can use the `interpolate` method to obtain the solution at the root time.

```rust,ignore
{{#include ../../../examples/intro-logistic-closures/src/solve_match_step.rs}}
```
