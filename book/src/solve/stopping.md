# Stopping

There are two methods to halt the solver when a certain condition is met: you can either stop at a specific time or stop when a certain event occurs. 

Stopping at a specific time is straightforward, as you can use the `set_stop_time` method on the `OdeSolverMethod` trait and then just check if the return value of the `step` method is `Ok(OdeSolverStopReason::TstopReached)`

Stopping at a certain event requires you to set a root function in your system of equations, see [Root Finding](specify/closure/root_finding.md) for more information. During time-stepping, you can check if the solver has discovered a root by checking if the return value of the `step` method is `Ok(OdeSolverStopReason::RootFound)`. `RootFound` holds the time at which the root was found, and you can use the `interpolate` method to obtain the solution at that time.

```rust,ignore
{{#include ../../../examples/intro-logistic-closures/src/solve_match_step.rs}}
```