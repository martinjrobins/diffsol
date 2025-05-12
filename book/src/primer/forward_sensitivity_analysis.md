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

Here, \\(\frac{dy}{dp}\\) is the sensitivity of the solution with respect to the parameters. The sensitivity equations are solved alongside the ODE system to give us the solution and the sensitivity of the solution with respect to the parameters. Note that this is a similar concept to forward-mode automatic differentiation, but whereas automatic differentiation calculates the derivative of the code itself (e.g. the "discretised" ODE system), forward sensitivity analysis calculates the derivative of the continuous equations before they are discretised. This means that the error control for forward sensitivity analysis is decoupled from the forward solve, and the tolerances for both can be set independently. However, both methods have the same scaling properties as the number of parameters increases, each additional parameter requires one additional solve, so the method is not efficient for large numbers of parameters (>100). In this case, adjoint sensitivity analysis is often preferred. 

To use forward sensitvity analysis in Diffsol, more equations need to be specified that calculate the gradients with respect to the parameters. If you are using the `OdeBuilder` struct and rust closures, you need to supply additional closures that calculate the gradient of the right-hand side, and the gradient of the initial state vector with respect to the parameters. You can see an example of this in the [Forward Sensitivity API](../specify/forward_sensitivity.md) section. If you are using the DiffSL language, these gradients are calculated automatically and you don't need to worry about them. An example of using forward sensitivity analysis in DiffSL is given in the [Fitting a Preditor-Prey Model to Data](./population_dynamics_fitting.md) section next.