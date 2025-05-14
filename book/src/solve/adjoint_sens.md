# Adjoint Sensitivities

Solving the adjoint sensitivity problem requires a different approach than the forward sensitivity problem. The adjoint sensitivity problem is solved by first solving the original ODE problem and checkpointing the solution so it can be used later. The adjoint sensitivity problem is then solved backwards in time, using the solution of the original ODE problem as an input.

The [`AdjointOdeSolverMethod`](https://docs.rs/diffsol/latest/diffsol/ode_solver/adjoint/trait.AdjointOdeSolverMethod.html) trait provides a way to compute the adjoint sensitivities of the solution of an ODE problem, using its `solve_dense_adjoint_sensitivities` method.