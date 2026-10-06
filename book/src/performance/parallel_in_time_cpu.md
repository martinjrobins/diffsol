# Parallel-in-Time (DEER)

Here we will illustrate the application of diffsol to a parallel-in-time method for solving a nonlinear ODE, the DEER algorithm introduced in [Lim et. al.](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3bfbd65743e60c685a3845bd61ce15f-Abstract-Conference.html). The example will use the forced the forced Duffing oscillator, a nonlinear second order ODE, and utilize the DEER algorithm to solve it in parallel across multiple time chunks using CPU threads and the `rayon` crate.

First we will define a few types, we will use the nalgebra backend and matrix/vector types, and a f64 scalar type.

```rust,ignore
{{#include ../../../examples/performance-cpu-parallel-in-time/src/main.rs:types}}
```

## The problem

The example used is the forced Duffing oscillator, a nonlinear second order ODE given by the equation

\\[
\ddot{x} + \delta \dot{x} + x + x^3 = \gamma \cos(\omega t),
\\]

which, writing \\(\mathbf{y} = (x, v)^T\\) with \\(v = \dot{x}\\), is the first order system

\\[
\begin{aligned}
\frac{dx}{dt} &= v, \\\\
\frac{dv}{dt} &= -\delta v - x - x^3 + \gamma \cos(\omega t),
\end{aligned}
\\]

or \\(\dot{\mathbf{y}} = \mathbf{f}(\mathbf{y}, t)\\), with \\(\delta = 0.5\\), \\(\gamma = 0.5\\), \\(\omega = 1\\) and \\(\mathbf{y}(0) = (1, 0)^T\\). The Jacobian of the right-hand side is

\\[
\frac{\partial \mathbf{f}}{\partial \mathbf{y}} = \begin{pmatrix} 0 & 1 \\\\ -1 - 3x^2 & -\delta \end{pmatrix}.
\\]

The parameters are the initial state, \\(\mathbf{p} = \mathbf{y}(t\_0)\\), so the right-hand side does not depend on them and the initial condition is the identity map:

\\[
\frac{\partial \mathbf{f}}{\partial \mathbf{p}} = \begin{pmatrix} 0 & 0 \\\\ 0 & 0 \end{pmatrix}, \quad \frac{\partial \mathbf{y}(t\_0)}{\partial \mathbf{p}} = \begin{pmatrix} 1 & 0 \\\\ 0 & 1 \end{pmatrix}.
\\]

```rust,ignore
{{#include ../../../examples/performance-cpu-parallel-in-time/src/main.rs:problem}}
```

## The DEER iteration

The states at the chunk boundaries \\(\mathbf{y}\_k \approx \mathbf{y}(t\_k)\\) satisfy the nonlinear recurrence

\\[
\mathbf{y}\_{k+1} = \boldsymbol{\phi}\_k(\mathbf{y}\_k), \quad k = 0, \dots, N - 1, \quad \mathbf{y}\_0 = \mathbf{y}(0).
\\]

Linearising about the current iterate \\(\mathbf{y}\_k^{(i)}\\) gives Newton's method for all boundary states at once,

\\[
\mathbf{y}\_{k+1}^{(i+1)} = \boldsymbol{\phi}\_k(\mathbf{y}\_k^{(i)}) + J\_k \left( \mathbf{y}\_k^{(i+1)} - \mathbf{y}\_k^{(i)} \right), \quad J\_k = \frac{\partial \boldsymbol{\phi}\_k}{\partial \mathbf{y}} \left( \mathbf{y}\_k^{(i)} \right),
\\]

which is a linear (affine) recurrence

\\[
\mathbf{y}\_{k+1}^{(i+1)} = J\_k \mathbf{y}\_k^{(i+1)} + \mathbf{b}\_k, \quad \mathbf{b}\_k = \boldsymbol{\phi}\_k(\mathbf{y}\_k^{(i)}) - J\_k \mathbf{y}\_k^{(i)}.
\\]

All of the \\(\boldsymbol{\phi}\_k(\mathbf{y}\_k^{(i)})\\) and \\(J\_k\\) can be computed in parallel. Each step of the recurrence is an affine map \\(\mathbf{y} \mapsto A \mathbf{y} + \mathbf{b}\\), and composing two such maps

\\[
(A\_2, \mathbf{b}\_2) \bullet (A\_1, \mathbf{b}\_1) = (A\_2 A\_1, A\_2 \mathbf{b}\_1 + \mathbf{b}\_2)
\\]

is associative with identity \\((I, \mathbf{0})\\). The prefixes

\\[
(P\_k, \mathbf{c}\_k) = (J\_k, \mathbf{b}\_k) \bullet (J\_{k-1}, \mathbf{b}\_{k-1}) \bullet \dots \bullet (J\_0, \mathbf{b}\_0)
\\]

can therefore be computed with a parallel scan, giving the new iterate

\\[
\mathbf{y}\_{k+1}^{(i+1)} = P\_k \mathbf{y}\_0 + \mathbf{c}\_k.
\\]

The iteration stops when

\\[
\max\_k \left\Vert \mathbf{y}\_k^{(i+1)} - \mathbf{y}\_k^{(i)} \right\Vert\_\infty < \text{tol}.
\\]

Newton's method still converges to the solution of the nonlinear recurrence if the \\(J\_k\\) are only approximate (here computed with a loose tolerance), but the convergence rate drops from quadratic to linear.

```rust,ignore
{{#include ../../../examples/performance-cpu-parallel-in-time/src/main.rs:deer}}
```

## Solving a single chunk

TODO

Split the time interval \\([0, T]\\) into \\(N\\) chunks with boundaries \\(t\_k = k \Delta t\\), \\(k = 0, \dots, N\\), where \\(\Delta t = T / N\\). Let \\(\boldsymbol{\phi}\_k(\mathbf{y})\\) be the flow map of chunk \\(k\\), the solution at \\(t\_{k+1}\\) of

\\[
\dot{\mathbf{y}} = \mathbf{f}(\mathbf{y}, t), \quad \mathbf{y}(t\_k) = \mathbf{y}.
\\]

Its Jacobian \\(J\_k = \partial \boldsymbol{\phi}\_k / \partial \mathbf{y}\\) is given by the forward sensitivities \\(S(t) = \partial \mathbf{y}(t) / \partial \mathbf{y}(t\_k)\\), which satisfy

\\[
\dot{S} = \frac{\partial \mathbf{f}}{\partial \mathbf{y}} S, \quad S(t\_k) = I, \quad J\_k = S(t\_{k+1}).
\\]

```rust,ignore
{{#include ../../../examples/performance-cpu-parallel-in-time/src/main.rs:chunk}}
```

## Initial guess

TODO: explain the coarse, loose-tolerance solve.

The initial guess \\(\mathbf{y}\_k^{(0)}\\), \\(k = 0, \dots, N\\), is a cheap sequential solve of \\(\dot{\mathbf{y}} = \mathbf{f}(\mathbf{y}, t)\\) with loose tolerances, evaluated at the chunk boundaries \\(t\_k\\).

```rust,ignore
{{#include ../../../examples/performance-cpu-parallel-in-time/src/main.rs:coarse}}
```

## Convergence

TODO

{{#include images/deer_error.html}}

## Thread Scaling

TODO

{{#include images/deer_scaling.html}}

TODO: discussion.
