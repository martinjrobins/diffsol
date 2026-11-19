# Solver options

The default options are suitable for most problems. You can change them at two levels:

- **The problem.** The fields `ode_options` ([`OdeSolverOptions`](https://docs.rs/diffsol/latest/diffsol/ode_solver/problem/struct.OdeSolverOptions.html)) and `ic_options` ([`InitialConditionSolverOptions`](https://docs.rs/diffsol/latest/diffsol/ode_solver/problem/struct.InitialConditionSolverOptions.html)) of the `OdeSolverProblem` apply to all the solvers that you create from the problem. Set them before you create a solver.
- **The solver.** When you create a solver, it copies its limits from `ode_options` into its config: [`BdfConfig`](https://docs.rs/diffsol/latest/diffsol/ode_solver/config/struct.BdfConfig.html), [`SdirkConfig`](https://docs.rs/diffsol/latest/diffsol/ode_solver/config/struct.SdirkConfig.html) or [`ExplicitRkConfig`](https://docs.rs/diffsol/latest/diffsol/ode_solver/config/struct.ExplicitRkConfig.html). Use `solver.config_mut()` to change the limits of one solver after you create it.

The tolerances are not in these options. See [Tolerances](../specify/tolerances.md).

```rust,ignore
{{#include ../../../examples/intro-logistic-closures/src/solver_options.rs}}
```

## ODE solver options

These are the main fields of `OdeSolverOptions`. The API documentation describes all the fields, including the step-size controller.

| Field | Default | Description |
|---|---|---|
| `max_number_timesteps` | 5000 | The maximum number of steps. All the `solve*` calls with the same solver share this limit. |
| `min_timestep` | machine epsilon | The minimum step size. |
| `max_error_test_failures` | 40 | The maximum number of rejected attempts in one step. |
| `max_nonlinear_solver_failures` | 10 | The maximum number of Newton failures in one step (implicit solvers only). |
| `max_nonlinear_solver_iterations` | 10 | The maximum number of Newton iterations in one nonlinear solve (implicit solvers only). |
| `nonlinear_solver_tolerance` | 0.2 | A scale factor for the Newton convergence test. A smaller value gives a tighter tolerance. |
| `derivative_solve_tolerance` | 0.2 | A scale factor for the convergence test of the linear solves of the forward sensitivities and the [discrete adjoint](../solve/adjoint_sens.md) (BDF and SDIRK only). |
| `max_timestep_growth` | 2 | The maximum increase of the step size in one step, as a ratio. |
| `min_timestep_shrink` | 0.5 | The maximum decrease of the step size in one step, as a ratio. |
| `update_jacobian_after_steps` | 20 | The maximum number of steps between two setups of the linear solver. |
| `update_rhs_jacobian_after_steps` | 50 | The maximum number of steps between two calculations of the Jacobian of the right-hand side. |
| `fixed_step` | `false` | Turn off error control and take steps of a constant size. You can use this only with manual time-stepping, and you cannot set a stop time. |

## Initial condition options

If the problem has a singular mass matrix, the implicit solvers solve for the algebraic variables when you create the solver (see [Initialisation](./initialisation.md)). `ic_options` controls this Newton solve:

| Field | Default | Description |
|---|---|---|
| `use_linesearch` | `true` | Use a backtracking line search. |
| `max_linesearch_iterations` | 10 | The maximum number of line-search iterations. |
| `max_newton_iterations` | 10 | The maximum number of Newton iterations. |
| `max_linear_solver_setups` | 4 | The maximum number of setups of the linear solver. |
| `step_reduction_factor` | 0.5 | The factor that decreases the step in the line search. |
| `armijo_constant` | 1e-4 | The Armijo constant of the line search. |

If this solve does not converge, the solver constructor returns an `InitialConditionDidNotConverge` error.

## Statistics

`solver.get_statistics()` returns an [`OdeSolverStatistics`](https://docs.rs/diffsol/latest/diffsol/ode_solver/struct.OdeSolverStatistics.html) struct, which counts what the solver did. The example above prints some of its fields:

- `number_of_steps`: the number of steps.
- `number_of_error_test_failures`: the number of rejected attempts.
- `number_of_nonlinear_solver_iterations` and `number_of_nonlinear_solver_fails`: the Newton iterations and failures.
- `number_of_linear_solver_setups`: the number of Jacobian and LU setups. Other fields give the cause of each setup.

Use these values to find why a solve is slow, or to compare solvers. For example, if there are many error test failures, the problem is possibly stiff, and an implicit solver can be better than an explicit solver (see [Stiff vs Non-Stiff](../performance/stiff_vs_non_stiff.md)).
