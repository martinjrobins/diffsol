# Adjoint Sensitivities

Adjoint sensitivity analysis calculates the gradient of a scalar cost function \\(G\\) with respect to the parameters \\(p\\). The cost of the backwards pass does not increase much with the number of parameters, so adjoints are a good choice for models with many parameters. The [Backwards Sensitivity Analysis](../primer/backwards_sensitivity_analysis.md) section of the primer gives the theory.

diffsol has two adjoint methods:

- **The continuous adjoint** ("optimise-then-discretise") solves the adjoint equations backwards in time with an ODE solver. The gradient is an approximation of the gradient of the exact solution, with an error that the solver tolerances control.
- **The discrete adjoint** ("discretise-then-optimise") differentiates the steps that the forward solver took, in reverse order. The gradient is the gradient of the numerical solution, with the step sizes held constant. It agrees with the gradient from the [forward sensitivities](./forward_sens.md).

We generally recommend the discrete adjoint. It does not solve a second ODE, so it is usually faster. Its gradient is also the exact gradient of the forward solve. With an implicit solver, the gradient is exact only to the tolerance of the linear solves in the backwards pass. The continuous adjoint is useful in the cases that the discrete adjoint does not support at the moment:

- the forward solve stops at root events.
- the cost function is the integral of the output.
- the forward solve has so many steps that the tape of the discrete adjoint uses too much memory.

For both methods, the problem must implement `OdeEquationsImplicitAdjoint`. For a problem made from closures, see [Adjoint Sensitivity](../specify/closure/adjoint_sensitivity.md). A problem made from DiffSL implements this trait automatically.

## The continuous adjoint

A continuous adjoint solve has two passes:

1. **The forward pass.** Solve the problem forwards in time with a checkpointed solve. This saves checkpoints of the solver state, so that the backwards pass can get the forward solution at any time.
   - [`solve_with_checkpointing`](https://docs.rs/diffsol/latest/diffsol/ode_solver/method/trait.OdeSolverMethod.html#method.solve_with_checkpointing) solves to a final time, like `solve`.
   - [`solve_dense_with_checkpointing`](https://docs.rs/diffsol/latest/diffsol/ode_solver/method/trait.OdeSolverMethod.html#method.solve_dense_with_checkpointing) solves at the times that you give, like `solve_dense`.

   The second argument of both methods is the maximum number of steps between two checkpoints. If it is `None`, the value is 500.
2. **The backwards pass.** Make an adjoint solver from the checkpoints, then call [`solve_adjoint_backwards_pass`](https://docs.rs/diffsol/latest/diffsol/ode_solver/adjoint/trait.AdjointOdeSolverMethod.html#method.solve_adjoint_backwards_pass). This method solves the adjoint equations backwards in time, from the end of the forward pass to the start, and returns the final adjoint state. The field `sg` of this state holds the gradient \\(dG/dp\\).

Each solver has a method that makes its adjoint solver: `bdf_solver_adjoint`, `tr_bdf2_solver_adjoint`, `esdirk34_solver_adjoint` and `tsit45_solver_adjoint`. These methods take three arguments:

- the checkpoints from the forward pass.
- an optional copy of the forward solver. The adjoint solver uses this copy to solve again between two checkpoints when it needs the forward solution at more times. You can use `None` only if the checkpoints already contain all the forward solution that the backwards pass needs.
- an optional number of adjoint outputs. Use `None` to use the number of outputs of the problem.

The adjoint solver can use a different method from the forward solver. For example, you can use a BDF solver for the forward pass and a TR-BDF2 solver for the backwards pass.

### Cost function: the integral of the output

If you give no data to the backwards pass, the cost function is the integral of the problem output \\(u(y)\\) over the solve:

\\[
G = \int_{t_0}^{t_f} u(y(t)) \\, dt.
\\]

If the problem has no output function, the output is the state. For this cost function, the problem must set [`integrate_out`](https://docs.rs/diffsol/latest/diffsol/ode_solver/builder/struct.OdeBuilder.html#method.integrate_out) to `true`. If it does not, the gradient is zero.

There is one cost function for each output, so `sg` holds one gradient for each output.

```rust,ignore
{{#include ../../../examples/intro-logistic-closures/src/solve_adjoint_sens.rs}}
```

### Cost function: a sum over data times

A cost function that compares the solution with data is often a sum of functions \\(g_i\\) at the data times \\(t_i\\):

\\[
G = \sum_{i} g_i(u(y(t_i))).
\\]

For example, the sum of squared errors uses \\(g_i = (u(y(t_i)) - d_i)^2\\), where \\(d_i\\) is the data at \\(t_i\\).

For this cost function, give the backwards pass the data times `t_eval` and the gradients \\(dg_i/du\\). Each item of the slice `dgdu_eval` is a dense matrix with one row for each output and one column for each data time. Column \\(i\\) is \\(dg_i/du\\) at \\(t_i\\). You can calculate it from the solution of the forward pass. Each matrix in the slice gives one cost function, so the number of adjoint outputs must be equal to the number of matrices.

In this example, the logistic problem is compared with data that comes from the same model with a different growth rate:

```rust,ignore
{{#include ../../../examples/intro-logistic-closures/src/solve_adjoint_sens_sum_squares.rs}}
```

### Tolerances

The adjoint equations and the parameter gradient have their own tolerances. See [Tolerances](../specify/tolerances.md).

### Events

If the forward pass stops at roots and applies resets, the checkpoints are split at each reset. The backwards pass applies the related correction to the adjoint state at each reset automatically.

## The discrete adjoint

The [`StepVjp`](https://docs.rs/diffsol/latest/diffsol/ode_solver/step_ad/trait.StepVjp.html) trait gives the discrete adjoint. The BDF, SDIRK and explicit Runge-Kutta solvers implement it.

The cost function is a sum over data times, as above:

\\[
G = \sum_{i} g_i(u(y(t_i))).
\\]

The [`solve_dense_vjp`](https://docs.rs/diffsol/latest/diffsol/ode_solver/step_ad/trait.StepVjp.html#method.solve_dense_vjp) method does the full calculation in one call. It solves forwards to the times `t_eval` and saves the solver state after each step. Then it goes back through the steps in reverse order, and returns the cost function \\(G\\) and the gradient \\(dG/dp\\). It takes these arguments:

- `t_eval`: the data times \\(t_i\\).
- `ndirections`: the number of cost functions. One backwards pass can calculate the gradients of more than one cost function.
- `g`: a closure that adds \\(g_i(u_i)\\) to the cost function. Its arguments are `(i, t_i, u_i, out)`.
- `dgdu`: a closure that writes \\(dg_i/du_i\\). Its arguments are also `(i, t_i, u_i, out)`.

This example uses the same problem and data as the continuous adjoint example above. It does not need a separate adjoint solver or checkpoints:

```rust,ignore
{{#include ../../../examples/intro-logistic-closures/src/solve_discrete_adjoint_sens.rs}}
```

If you want to calculate \\(dg_i/du\\) yourself, or use one forward solve for more than one cost function, use the two halves of `solve_dense_vjp`:

- [`solve_dense_vjp_fwd`](https://docs.rs/diffsol/latest/diffsol/ode_solver/step_ad/trait.StepVjp.html#method.solve_dense_vjp_fwd) solves forwards and returns the solution and a tape of the steps.
- [`solve_dense_vjp_bkwd`](https://docs.rs/diffsol/latest/diffsol/ode_solver/step_ad/trait.StepVjp.html#method.solve_dense_vjp_bkwd) takes the tape and a matrix of \\(dg_i/du\\) (the same as `dgdu_eval` for the continuous adjoint), and returns \\(dG/dp\\). The tape does not change, so you can use it again for a different cost function.

The [spring-mass fitting example](../primer/spring_mass_fitting.md) uses these two methods, and compares them with the continuous adjoint. For full control, `step_vjp` and `interpolate_vjp` differentiate one step and one interpolation.

The discrete adjoint has these limits:

- The forward solve must not stop at a root, and the problem must not set `integrate_out`. A cost function that is the integral of the output is not available.
- The tape holds the solver state after every step, so its memory increases with the number of steps. The continuous adjoint saves a checkpoint only every 500 steps (by default).
- If the mass matrix depends on the parameters, the gradient does not include that term.
