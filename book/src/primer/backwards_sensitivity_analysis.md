# Backwards Sensitivity Analysis

Backwards sensitivity analysis, as the name suggests, starts from a given cost or loss function that you want to minimise, and derives the gradient of this function with respect to the parameters of the ODE systems using a lagrangian multiplier approach.

diffsol supports two different classes of loss functions, the first being an integral of a model output function \\(g(t, u)\\) over time,

\\[
G(p) = \int_0^{t_f} g(t, u) dt
\\]

The second being a sum of \\(n\\) discrete functions \\(h_i(t, u)\\) at time points \\(t_i\\),

\\[
G(p) = \int_0^{t_f} \sum_{i=1}^n h_i(t_i, u) \delta(t - t_i) dt
\\]

Note that the \\(h_i\\) functions can be a combination of the (continuous) model output function \\(g\\) and a user-defined discrete function, such as the sum of squares difference between the model output and some observed data.

There are two main approaches to calculate the gradient of the loss function with respect to the parameters, \\(\frac{dG}{dp}\\). The first approach starts from the continuous ODE equations, and gives the *adjoint equations*. A standard ODE solver method then solves the adjoint equations backwards in time. The second approach starts from the *discrete* steps of a given solver method, and applies reverse-mode automatic differentiation directly to these steps. This gives the gradient of the numerical solution that the solver calculated. diffsol supports both of these approaches. The sections below describe them.

## Continuous adjoint equations

The derivation below is modified from that given in [(Rackauckas et al. 2021)](https://arxiv.org/abs/2001.04385), and uses the first form of the cost function, but the same approach can be used for the second form.

### Lagrangian Multiplier Approach

We wish to minimise the cost function \\(G(p)\\) with respect to the parameters \\(p\\), and subject to the constraints of the ODE system of equations,

\\[
M \frac{du}{dt} = f(t, u, p)
\\]

where \\(M\\) is the mass matrix, \\(u\\) is the state vector, and \\(f\\) is the right-hand side function. We can write the lagrangian as

\\[
L(u, \lambda, p) = G(p) - \int_0^{t_f} \lambda^T \left(M \frac{du}{dt} - f(t, u, p)\right) dt
\\]

where \\(\lambda\\) is the lagrangian multiplier. We already know we can generate a solution to the ODE system \\(u(t)\\) that will satisfy the constraint such that the last term in the lagrangian is zero, meaning that the gradient of the lagrangian is equal to the gradient of the cost function. Therefore, we can write the gradient of the cost function as

\\[
\frac{dG}{dp} = \int_0^{t_f} (g_p + g_u u_p) dt - \int_0^{t_f} \lambda^T (M \frac{du_p}{dt} - f_u u_p - f_p) dt
\\]

where \\(g_p\\) and \\(g_u\\) are the partial derivatives of the output function \\(g\\) with respect to the parameters and state variables, respectively, and \\(u_p\\) is the partial derivative of the state vector with respect to the parameters, also known as the sensitivities.

This equation can be simplified by using integration by parts, and requiring that the adjoint ODE system is satisfied, which is given by

\\[
\begin{aligned}
M^T \frac{d \lambda}{dt} &= -f_u^T \lambda - g_u^T \\\\
\lambda(t_f) &= 0
\end{aligned}
\\]

giving the gradient of the cost function as

\\[
\frac{dG}{dp} = \lambda^T(0) M u_p(0) + \int_0^{t_f} (g_p + \lambda^T f_p) dt
\\]

### Solving the Adjoint ODE System

Solving the adjoint ODE system is done in two stages. First, we solve the forward ODE system to get the state vector \\(u(t)\\). We require this solution to be valid the entire time interval \\([0, t_f]\\), so we use a checkpointing system to store the state vector at regular intervals and interpolate between them to get the state vector at any time point. The second stage is to solve the adjoint ODE system backwards in time, starting from the final time point \\(t_f\\) and using the interpolated state vector to supply \\(u(t)\\) as needed.

The gradient \\(\frac{dG}{dp}\\) can be calculated by performing a quadrature over the time interval \\([0, t_f]\\) to calculate the last term in the equation above. Special consideration needs to be taken for the second form of the cost function above, where the discrete functions are evaluated at specific time points. In this case, the solver will need to be stopped at each time point and the contribution \\(M^{-T} g_u^T\\) added to the state vector \\(\lambda\\), and the contribution \\(g_p\\) added to the gradient \\(\frac{dG}{dp}\\).

In the case that \\(M\\) is singular, but can be divided into an invertible block and zero blocks like so:

\\[
M = \begin{bmatrix} M_{dd} & 0 \\\\ 0 & 0 \end{bmatrix}
\\]

where \\(M_{dd}\\) is invertible. The subscript \\(d\\) refers to the differential variables and the subscript \\(a\\) refers to the algebraic variables. The corresponding block decomposition of the adjoint Jacobian and the partial derivative of the output function can be written as

\\[
f_u^T = \begin{bmatrix} f_{dd} & f_{da} \\\\ f_{ad} & f_{aa} \end{bmatrix}
\\]

and

\\[
g_u^T = \begin{bmatrix} g_{d} \\\\ g_{a} \end{bmatrix}.
\\]

The algebraic part of the adjoint equation has no time derivative. Thus the discrete function also changes \\(\lambda_a\\) at the time point, and this changes the contribution to \\(\lambda_d\\). The contribution to the differential part of \\(\lambda\\) is

\\[
\lambda_d \mathrel{+}= M_{dd}^{-T} \left(g_{d} - f_{da} f_{aa}^{-1} g_{a}\right)
\\]

The contribution to the gradient also has an extra term. Let \\(f_{p,a}\\) be the rows of \\(f_p\\) for the algebraic variables. The contribution to the gradient is

\\[
\left(\frac{dG}{dp}\right)^T \mathrel{+}= g_p^T - f_{p,a}^T f_{aa}^{-1} g_{a}
\\]

### Specifying the discrete functions

Here we consider the second form of the cost function. If we have a model output function \\(m(t, u)\\), and a set of discrete functions \\(h_i(t, u)\\) which only depend on the model output (i.e. \\(h_i(t, u) = h_i(m(t, u))\\)), then the partial derivatives of the cost function \\(g\\) with respect to the parameters can be calculated as

\\[
\begin{aligned}
g_u &= g_m m_u \\\\
g_p &= g_m m_p
\end{aligned}
\\]

where \\(m_u\\) and \\(m_p\\) are the partial derivatives of the model output function with respect to the state variables and parameters, respectively. Therefore, a user only has to supply \\(g_m\\) at each of the time points \\(t_i\\) and diffsol will be able to calculate the correct gradients as part of the backwards solution of the adjoint ODE system.

## Discrete gradients

An ODE solver method calculates the solution as a sequence of steps. Each step is a function \\(\Phi\\) that maps the solver state \\(s_i\\) at step \\(i\\) to the solver state at the next step:

\\[
s_{i+1} = \Phi(s_i, p)
\\]

The solver state \\(s_i\\) contains the time \\(t_i\\), the state vector \\(u_i\\) and the other data that the method needs, for example the history of a BDF method or the stages of a Runge-Kutta method. The output at a time point \\(t\\) is not usually at the end of a step. Thus there is also an interpolation function \\(h\\). If \\(t_i < t \le t_{i+1}\\), then \\(h\\) uses the solver state at the end of the step to calculate the solution at \\(t\\):

\\[
u(t) = h(s_{i+1}, t)
\\]

### Reverse-mode automatic differentiation

The diffsol code calculates the step function \\(\Phi\\) with:

1. a sequence of linear algebra operations on the solver state, and
2. control flow that uses the error estimate to select a step size that satisfies the tolerances. This includes the decision to accept or reject a step, and (for BDF) the selection of the order.

Reverse-mode automatic differentiation gives the gradient of (1). We ignore (2), because the forward solve already gives the step sizes, the orders and the accepted steps. We keep these as fixed values.

Reverse-mode automatic differentiation uses the chain rule from the output back to the input. Each step has a *cotangent* \\(\bar{s}\_{i+1} = \left(\frac{\partial G}{\partial s\_{i+1}}\right)^T\\), which is the gradient of the loss with respect to the solver state at the end of the step. The vector-Jacobian product (VJP) of the step moves this cotangent back to the start of the step, and adds the contribution of the step to the parameter gradient \\(\bar{p}\\):

\\[
\begin{aligned}
\bar{s}\_i &= \left(\frac{\partial \Phi}{\partial s\_i}\right)^T \bar{s}\_{i+1} \\\\
\bar{p} &\mathrel{+}= \left(\frac{\partial \Phi}{\partial p}\right)^T \bar{s}\_{i+1}
\end{aligned}
\\]

The VJP does not make the Jacobians \\(\frac{\partial \Phi}{\partial s_i}\\) or \\(\frac{\partial \Phi}{\partial p}\\). Each operation of the step has a VJP, and the step VJP applies these in the opposite order. Thus the cost of the step VJP is approximately the cost of one step, and it does not increase with the number of parameters. The VJP needs the solver state at each step, so the forward solve keeps a copy of the solver state after each step.

For explicit solvers this process is simple. We do not use an automatic differentiation library. For each solver, we know the sequence of operations of the step, so we write down the operations of its VJP directly. The VJP of an explicit step is exact to floating point error.

Implicit solvers must solve a nonlinear system of equations at each step, and this is the main problem for reverse-mode automatic differentiation. At each step (or at each stage of a Runge-Kutta method), the solver finds the value \\(z\\) that solves

\\[
F(z, s_i, p) = 0
\\]

The *Implicit Function Theorem* gives the derivatives of the solution \\(z\\) without differentiating the iterations of the nonlinear solver. This needs a linear solve with the transposed Jacobian of the ODE equations. During the forward solve, we usually have only a factorisation of an *approximate* Jacobian, because the Newton iteration does not update the Jacobian at each step. To calculate and factorise the exact Jacobian at each reverse step is expensive. Instead, we use an iterative method (a Richardson iteration) for the linear solve, and we use the approximate Jacobian as a preconditioner \\(P\\).

The iterative method stops when it satisfies its tolerance, `derivative_solve_tolerance` (see [`OdeSolverOptions`](https://docs.rs/diffsol/latest/diffsol/ode_solver/problem/struct.OdeSolverOptions.html)). The maximum number of iterations is `derivative_solve_max_iter`. Thus, for implicit solvers, the gradient of each step is accurate only to this tolerance. For explicit solvers, the gradient of each step is exact to floating point error. The forward sensitivities of diffsol differentiate the same steps in the same way, with the same linear solves and the same tolerance. Thus the forward and reverse-mode gradients of an implicit solver agree to this tolerance.

### Discrete gradient of the loss function

Here we use the second form of the cost function, a sum of functions \\(g_j\\) of the output at the time points \\(t_j\\):

\\[
G(p) = \sum_{j=1}^{n} g_j(u(t_j)), \qquad u(t_j) = h(s_{k(j)}, t_j)
\\]

where \\(k(j)\\) is the step that reaches \\(t_j\\). The VJPs of \\(\Phi\\) and \\(h\\) give \\(\frac{dG}{dp}\\) in three stages:

1. **Forward pass.** Solve the ODE from \\(t_0\\) to \\(t_n\\). Keep a copy of the solver state \\(s_i\\) after each step, and record the step \\(k(j)\\) that reaches each time point \\(t_j\\). Calculate the outputs \\(u(t_j)\\) and the loss gradients \\(\frac{\partial g_j}{\partial u}\\).
2. **Backward pass.** Start with \\(\bar{s}_N = 0\\) at the last step \\(N\\), and \\(\bar{p} = 0\\). For each step \\(i = N, N-1, \dots, 1\\):
   - for each time point \\(t_j\\) in the step (\\(k(j) = i\\)), add the VJP of the interpolation:
     \\(\bar{s}_i \mathrel{+}= \left(\frac{\partial h}{\partial s_i}\right)^T \left(\frac{\partial g_j}{\partial u}\right)^T\\)
   - use the VJP of the step to calculate \\(\bar{s}_{i-1}\\), and add the contribution of the step to \\(\bar{p}\\).
3. **Initial state.** The initial state \\(s_0\\) also depends on the parameters: \\(u_0\\) comes from the initial condition, and the solver calculates \\(\frac{du_0}{dt}\\) from \\(f(t_0, u_0, p)\\) (or, for a DAE, it solves for consistent initial conditions). Add the last contribution \\(\bar{p} \mathrel{+}= \left(\frac{\partial s_0}{\partial p}\right)^T \bar{s}_0\\).

At the end, \\(\bar{p} = \left(\frac{dG}{dp}\right)^T\\). If the model has an output function \\(m(t, u)\\), as in the [section above](#specifying-the-discrete-functions), the user supplies \\(\frac{\partial g_j}{\partial m}\\), and diffsol uses the chain rule through \\(m_u\\) and \\(m_p\\).

This is the gradient of the loss that the solver actually calculates, with fixed step sizes, orders and interpolation weights. It is not an approximation of the gradient of the exact solution, as the continuous adjoint is. Thus the gradient agrees with the loss that an optimiser sees. There are some limits:

- The mass matrix \\(M\\) must not depend on the parameters, because diffsol does not differentiate this term.
- The cost function must be the second form, a sum of functions at time points. The discrete gradients do not support the first form, an integral of the output (`integrate_out`).
- The solve must not stop at roots of a root function. A step that finds a root gives an error. Thus the discrete gradients do not support resets.

If any of these limitations affect you, please open an issue on [the repository](https://github.com/martinjrobins/diffsol).

The continuous adjoint method supports resets. The checkpointed forward solve applies each reset and splits the checkpoints at the reset. At each reset, the backward solve then updates \\(\lambda\\) and the gradient \\(\frac{dG}{dp}\\) with the adjoint of the reset. The adjoint of a reset does not support a mass matrix.
