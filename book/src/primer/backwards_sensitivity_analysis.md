# Backwards Sensitivity Analysis

Backwards sensitivity analysis, as the name suggests, starts from a given cost or loss function that you want to minimise, and derives the gradient of this function with respect to the parameters of the ODE systems using a lagrangian multiplier approach.

Diffsol supports two different classes of loss functions, the first being an integral of a model output function \\(g(t, u)\\) over time,

$$
G(p) = \int_0^{t_f} g(t, u) dt
$$

The second being a sum of \\(n\\) discrete functions \\(h_i(t, u)\\) at time points \\(t_i\\),

$$
G(p) = \int_0^{t_f} \sum_{i=1}^n h_i(t_i, u) \delta(t - t_i) dt
$$

Note that the \\(h_i\\) functions can be a combination of the (continuous) model output function \\(g\\) and a user-defined discrete function, such as the sum of squares difference between the model output and some observed data.

The derivation below is modified from that given in [(Rackauckas et. al. 2021)](https://arxiv.org/abs/2001.04385), and uses the first form of the cost function, but the same approach can be used for the second form.

## Lagrangian Multiplier Approach

We wish to minimise the cost function \\(G(p)\\) with respect to the parameters \\(p\\), and subject to the constraits of the ODE system of equations,

$$
M \frac{du}{dt} = f(t, u, p)
$$

where \\(M\\) is the mass matrix, \\(u\\) is the state vector, and \\(f\\) is the right-hand side function. We can write the lagrangian as

$$
L(u, \lambda, p) = G(p) + \int_0^{t_f} \lambda^T (M \frac{du}{dt} - f(t, u, p))
$$

where \\(\lambda\\) is the lagrangian multiplier. We already know we can generate a solution to the ODE system \\(u(t)\\) that will satisfy the constaint such that the last term in the lagrangian is zero, meaning that the gradient of the lagrangian is equal to the gradient of the cost function. Therefore, we can write the gradient of the cost function as

$$
\frac{dG}{dp} = \int_0^{t_f} (g_p + g_u u_p) dt - \int_0^{t_f} \lambda^T (M u \frac{du_p}{dt} - f_u u_p - f_p) dt
$$

where \\(g_p\\) and \\(g_u\\) are the partial derivatives of the output function \\(g\\) with respect to the parameters and state variables, respectively, and \\(u_p\\) is the partial derivative of the state vector with respect to the parameters, also known as the sensitivities. 

This equation can be simplified by using integration by parts, and requiring that the adjoint ODE system is satisfied, which is given by

$$
\begin{aligned}
M \frac{d \lambda}{dt} &= -f_u^T \lambda - g_u^T \\\\
\lambda(t_f) &= 0
\end{aligned}
$$

giving the gradient of the cost function as

$$
\frac{dG}{dp} = \lambda^T(0) M u_p(0) + \int_0^{t_f} (g_p + \lambda^T f_p) dt
$$


## Solving the Adjoint ODE System

Solving the adjoint ODE system is done in two stages. First, we solve the forward ODE system to get the state vector \\(u(t)\\). We require this solution to be valid the entire time interval \\([0, t_f]\\), so we use a checkpointing system to store the state vector at regular intervals in interpolate between them to get the state vector at any time point. The second stage is to solve the adjoint ODE system backwards in time, starting from the final time point \\(t_f\\) and using the interpolated state vector to supply \\(u(t)\\) as needed.

The gradient \\(\frac{dG}{dp}\\) can be calculated by performing a quadrature over the time interval \\([0, t_f]\\) to calculate the last term in the equation above. Special consideration needs to be taken for the second form of the cost function above, where the discrete functions are evaluated at specific time points. In this case, the solver will need to be stopped at each time point and the contribution \\(M^{-1} g_u^T\\) added to the state vector \\(\lambda\\), and the contribution \\(g_p\\) added to the gradient \\(\frac{dG}{dp}\\).

In the case that \\(M\\) is singular, but can be divided into a singular and zero blocks like so:

$$
M = \begin{bmatrix} M_{11} & 0 \\\\ 0 & 0 \end{bmatrix}
$$

where \\(M_{11}\\) is invertible. The corresponding block decomposition of the adjoint Jacobian and the partial derivative of the output function can be written as

$$
f_u^T = \begin{bmatrix} f_{dd} & f_{da} \\\\ f_{ad} & f_{aa} \end{bmatrix}
$$

and 

$$
g_u^T = \begin{bmatrix} g_{d} & g_{a} \end{bmatrix}.
$$

In this case, the contribution to \\(\lambda\\) can be calculated as

$$
-f_{da} f_{aa}^{-1} g_{a} + M_{11}^{-1} g_{d}
$$

## Specifying the discrete functions

Here we consider the second form of the cost function. If we have a model output function \\(m(t, u)\\), and a set of discrete functions \\(h_i(t, u)\\) which only depend on the model output (i.e. \\(h_i(t, u) = h_i(m(t, u))\\)), then the partial derivatives of the cost function \\(g\\) with respect to the parameters can be calculated as

$$
\begin{aligned}
g_u &= g_m m_u \\\\
g_p &= g_m m_p
\end{aligned}
$$

where \\(m_u\\) and \\(m_p\\) are the partial derivatives of the model output function with respect to the state variables and parameters, respectively. Therefore, a user only has to supply \\(g_m\\) at each of the time points \\(t_i\\) and Diffsol will be able to calculate the correct gradients as part of the backwards solution of the adjoint ODE system.