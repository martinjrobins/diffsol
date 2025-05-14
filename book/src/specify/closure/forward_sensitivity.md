# Forward Sensitivity

In this section we will discuss how to compute the forward sensitivity of the solution of an ODE problem. The forward sensitivity is the derivative of the solution with respect to the parameters of the problem. This is useful for many applications, such as parameter estimation, optimal control, and uncertainty quantification.

## Specifying the sensitivity problem

We will again use the example of the logistic growth equation, but this time we will compute the sensitivity of the solution \\(y\\) with respect to the parameters \\(r\\) and \\(K\\) (i.e. \\(\frac{dy}{dr}\\) and \\(\frac{dy}{dK}\\)). 
The logistic growth equation is:

\\[\frac{dy}{dt} = r y (1 - y/K),\\]
\\[y(0) = 0.1\\]

Recall from [ODE equations](ode_equations.md) that we also need to provide the jacobian of the right hand side of the ODE with respect to the state vector \\(y\\) and the gradient vector \\(v\\), which we will call \\(J\\). This is:

\\[J v = \begin{bmatrix} r v (1 - 2 y/K) \end{bmatrix}.\\]

Using the logistic growth equation above, we can compute the partial derivative of the right hand side of the ODE with respect to the vector \\([r, K]\\) multiplied by a vector \\(v = [v_r, v_K]\\), which we will call \\(J_p v\\). This is:

\\[J_p v = v_r y (1 - y/K) + v_K r y^2 / K^2 .\\]

We also need the partial derivative of the initial state vector with respect to the parameters multiplied by a vector \\(v\\), which we will call \\(J_{y_0} v\\). Since the initial state vector is constant, this is just zero

\\[J_{y_0} v = 0.\\]

We can then use the `OdeBuilder` struct to specify the sensitivity problem. The `rhs_sens_implicit` and `init_sens` methods are used to create a new problem that includes the sensitivity equations.

```rust,ignore
{{#include ../../../../examples/intro-logistic-closures/src/problem_fwd_sens.rs}}
```

## Solving the sensitivity problem

Once the sensitivity problem has been specified, we can solve it using the [`OdeSolverMethod`](https://docs.rs/diffsol/latest/diffsol/ode_solver/method/trait.OdeSolverMethod.html) trait. 
Lets imagine we want to solve the sensitivity problem up to a time \\(t_o = 10\\). We can use the `OdeSolverMethod` trait to solve the problem as normal, stepping forward in time until we reach \\(t_o\\).

```rust,ignore
{{#include ../../../../examples/intro-logistic-closures/src/solve_fwd_sens_step.rs}}
```

We can then obtain the sensitivity vectors at time \\(t_o\\) using the `interpolate_sens` method on the `OdeSolverMethod` trait. 
This method returns a `Vec<DVector<f64>>` where each element of the vector is the sensitivity vector for element \\(i\\) of the parameter vector at time \\(t_o\\).
If we need the sensitivity at the current internal time step, we can get this from the `s` field of the [`OdeSolverState`](https://docs.rs/diffsol/latest/diffsol/ode_solver/method/struct.OdeSolverState.html) struct.