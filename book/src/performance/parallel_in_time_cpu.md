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

The time interval \\([0, T]\\) is split into \\(N\\) chunks, and each chunk is integrated independently, in parallel, from a guess of the state at its start time:

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

Each step of the recurrence is an affine map \\(\mathbf{y} \mapsto A \mathbf{y} + \mathbf{b}\\), fully described by the pair \\((A, \mathbf{b})\\). Applying step 0 and then step 1 gives

\\[
\mathbf{y}\_2 = J\_1 (J\_0 \mathbf{y}\_0 + \mathbf{b}\_0) + \mathbf{b}\_1 = (J\_1 J\_0) \mathbf{y}\_0 + (J\_1 \mathbf{b}\_0 + \mathbf{b}\_1),
\\]

which is again an affine map. Writing this composition as an operator on pairs (read right to left, so the right-hand map is applied first),

\\[
(A\_2, \mathbf{b}\_2) \bullet (A\_1, \mathbf{b}\_1) = (A\_2 A\_1, A\_2 \mathbf{b}\_1 + \mathbf{b}\_2).
\\]

Like any composition of functions, this operator is associative, \\((f \bullet g) \bullet h = f \bullet (g \bullet h)\\), and it has the identity \\((I, \mathbf{0})\\), the map that leaves \\(\mathbf{y}\\) unchanged. Associativity means the steps can be grouped in any order, so different threads can combine different groups of steps at the same time. The prefix \\((P\_k, \mathbf{c}\_k)\\) is the single affine map equal to applying steps \\(0, 1, \dots, k\\) in order,

\\[
(P\_k, \mathbf{c}\_k) = (J\_k, \mathbf{b}\_k) \bullet (J\_{k-1}, \mathbf{b}\_{k-1}) \bullet \dots \bullet (J\_0, \mathbf{b}\_0),
\\]

which takes \\(\mathbf{y}\_0\\) straight to \\(\mathbf{y}\_{k+1}\\). A sequential loop would build each prefix from the previous one, but a parallel scan computes all \\(N\\) prefixes in \\(O(\log N)\\) rounds of combining (given enough threads). Each new boundary state then depends only on \\(\mathbf{y}\_0\\), so they can all be evaluated in parallel:

\\[
\mathbf{y}\_{k+1}^{(i+1)} = P\_k \mathbf{y}\_0 + \mathbf{c}\_k.
\\]

The iteration stops when

\\[
\max\_k \left\Vert \mathbf{y}\_k^{(i+1)} - \mathbf{y}\_k^{(i)} \right\Vert\_\infty < \text{tol}.
\\]

```rust,ignore
{{#include ../../../examples/performance-cpu-parallel-in-time/src/main.rs:deer}}
```

## Solving a single chunk

Let \\(\boldsymbol{\phi}\_k(\mathbf{y})\\) be the flow map of chunk \\(k\\), the solution at \\(t\_{k+1}\\) of

\\[
\dot{\mathbf{y}} = \mathbf{f}(\mathbf{y}, t), \quad \mathbf{y}(t\_k) = \mathbf{y}.
\\]

Its Jacobian \\(J\_k = \partial \boldsymbol{\phi}\_k / \partial \mathbf{y}\\) is given by the forward sensitivities \\(S(t) = \partial \mathbf{y}(t) / \partial \mathbf{y}(t\_k)\\), which satisfy

\\[
\dot{S} = \frac{\partial \mathbf{f}}{\partial \mathbf{y}} S, \quad S(t\_k) = I, \quad J\_k = S(t\_{k+1}).
\\]

Newton's method still converges to the solution of the nonlinear recurrence if the \\(J\_k\\) are only approximate, so we can use a much cheaper solve with loose tolerances and forward sensitivities to compute \\(J\_k\\).

```rust,ignore
{{#include ../../../examples/performance-cpu-parallel-in-time/src/main.rs:chunk}}
```

## Initial guess

The initial guess \\(\mathbf{y}\_k^{(0)}\\), \\(k = 0, \dots, N\\) also only needs to be approximate, so we use a cheap sequential solve of \\(\dot{\mathbf{y}} = \mathbf{f}(\mathbf{y}, t)\\) with loose tolerances, evaluated at the chunk boundaries \\(t\_k\\).

```rust,ignore
{{#include ../../../examples/performance-cpu-parallel-in-time/src/main.rs:coarse}}
```

## Convergence

With a reasonable initial guess the Newton iteration converges in a few iterations, and the final solution is accurate to the tolerance of the chunk solves.

{{#include images/deer_error.html}}

## Thread Scaling

We do not expect perfect linear scaling with the number of threads, since:

- the parallel scan requires \\(O(\log N)\\) rounds of combining
- the initial coarse solve is serial
- each solve chuck is an adaptive solve, so slower solves will hold up the faster ones

Here we see a maximum speedup of just over 5x with 25-30 threads and a total of 96 chunks. Some possible improvements that we could make to this example are:

- We are not getting much advantage out of the parallel scan, since its only 96 compositions of 2x2 matrices. A more expensive composition (more states or more chunks) could help here.
- The coarse solve is serial so is limiting the speedup, could investigate an even looser tolerance, or if the solve was more expensive (see first point), could try removing this rely on more newton iterations.
- We solve every chunk at each newton iteration, but the chunks that have already converged could be skipped in later iterations (e.g. the first chunk is always exact)
- Each chunk solve creates a new solver, but we could reuse the solvers between iterations and reset them to the new initial condition, which would save some setup time.
- Could reuse the first iteration's jacobians, since these only need to be approximate.

{{#include images/deer_scaling.html}}
