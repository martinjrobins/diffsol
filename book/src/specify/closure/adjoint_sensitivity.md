# Adjoint Sensitivity

Adjoint sensitivity analysis efficiently computes the gradient of a scalar objective with respect to many parameters.
For an implicit solver, you need to specify:

1. the [right-hand side](https://docs.rs/diffsol/latest/diffsol/ode_solver/builder/struct.OdeBuilder.html#method.rhs_adjoint_implicit) together with its state Jacobian-vector product and the two negative-transpose products used by the backward adjoint solver.
2. the [initial condition](https://docs.rs/diffsol/latest/diffsol/ode_solver/builder/struct.OdeBuilder.html#method.init_adjoint) together with the parameter sensitivity negative-transpose product.

For the logistic equation

$$\frac{dy}{dt} = r y (1 - y/K),$$

the state Jacobian-vector product is

$$Jv = r v (1 - 2y/K),$$

and the adjoint closure must return \\(-J^T v\\):

$$-J^T v = -r v (1 - 2y/K).$$

The parameter-adjoint closure similarly returns \\(-J_p^T v\\):

$$-J_p^T v = \begin{bmatrix}
-v y (1-y/K) \\\\
-v r y^2/K^2
\end{bmatrix}.$$

Use `rhs_adjoint_implicit` for these four right-hand-side operations and
`init_adjoint` for the initial state and its negative parameter transpose. The
logistic initial state below is constant, so its parameter-adjoint product is
zero.

```rust,ignore
{{#include ../../../../examples/intro-logistic-closures/src/problem_adjoint_sens.rs}}
```

The resulting problem implements `OdeEquationsImplicitAdjoint` and can be used
to create an adjoint solver after a checkpointed forward solve.
