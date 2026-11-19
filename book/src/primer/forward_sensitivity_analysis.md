# Forward Sensitivity Analysis

Recall our general ODE system (we'll define it without the mass matrix for now):

\\[
\begin{align*}
\frac{dy}{dt} &= f(t, y, p) \\\\
y(t_0) &= y_0
\end{align*}
\\]

Solving this system gives us the solution \\(y(t)\\) for a given set of parameters \\(p\\). However, we often want to know how the solution changes with respect to the parameters (e.g. for model fitting). This is where forward sensitivity analysis comes in. If we take the derivative of the ODE system with respect to the parameters, we get the sensitivity equations:

\\[
\begin{align*}
\frac{d}{dt} \frac{dy}{dp} &= \frac{\partial f}{\partial y} \frac{dy}{dp} + \frac{\partial f}{\partial p} \\\\
\frac{dy}{dp}(t_0) &= \frac{dy_0}{dp}
\end{align*}
\\]

Here, \\(\frac{dy}{dp}\\) is the sensitivity of the solution with respect to the parameters. The sensitivity equations are solved alongside the ODE system to give us the solution and the sensitivity of the solution with respect to the parameters. The equations above are the continuous sensitivity equations. diffsol does not solve them with a separate discretisation. Instead, diffsol differentiates the discrete steps of the solver, in the same way as forward-mode automatic differentiation. These are the same steps that the reverse-mode gradients differentiate (see [Backwards Sensitivity Analysis](./backwards_sensitivity_analysis.md#discrete-gradients)). For implicit solvers, the sensitivities are solved to the `derivative_solve_tolerance` of the solver options. For explicit solvers, no solve is necessary, so the sensitivities are exact to floating point error.

By default, the sensitivities are not included in the error control. Thus they do not change the step sizes of the forward solve. To include the sensitivities in the error control, set both the `sens_rtol` and `sens_atol` tolerances on the `OdeBuilder`.

Forward sensitivity analysis and forward-mode automatic differentiation have the same scaling properties as the number of parameters increases: each additional parameter requires one additional solve, so the method is not efficient for large numbers of parameters (>100). In this case, adjoint sensitivity analysis is often preferred. 

To use forward sensitivity analysis in diffsol, more equations need to be specified that calculate the gradients with respect to the parameters. If you are using the `OdeBuilder` struct and rust closures, you need to supply additional closures that calculate the gradient of the right-hand side, and the gradient of the initial state vector with respect to the parameters. You can see an example of this in the [Forward Sensitivity API](../specify/closure/forward_sensitivity.md) section. If you are using the DiffSL language, these gradients are calculated automatically and you don't need to worry about them. An example of using forward sensitivity analysis in DiffSL is given in the [Fitting a Predator-Prey Model to Data](./population_dynamics_fitting.md) section next.