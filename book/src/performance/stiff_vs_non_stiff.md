# Stiff vs Non-Stiff

ODE problems are often categorised as being either stiff or non-stiff.
These categories are loosely defined, but some of the characteristics of stiff problems are:

- widely separate timescales or fast/slow modes, for example a problem that exhibits modes with decay rates of \\(1\\) and \\(10^6\\).
- a Jacobian \\(J(t, y) = \frac{\partial f}{\partial y}\\) that has eigenvalues with large negative real parts. This indicates rapidly decaying modes with short timescales.
- solving with explicit solvers leads to very small steps. The solution may look smooth and slowly varying, but tiny explicit steps are required.
- small perturbations to a slowly varying solution decay very rapidly.

Note that just because a solution rapidly changes, this does not mean that it is stiff. For example the problem

$$
\frac{dy}{dt}=\cos(1000t)
$$

requires very small steps to maintain accuracy, but this does not mean that the problem is stiff. On the other hand, a slowly changing solution can be stiff

$$
\frac{dy}{dt}=-1000(y-\cos t)-\sin t.
$$

After an initial transient (via the first term), the solution is largely governed by the second term and is simply \\(cos(t)\\), but explicit methods will still be constrained by the first term and take very small steps.

diffsol has a few solvers, both explicit and implicit, for solving ODE problems. Choosing a solver is balanced by a few characteristics of the problem, one of which is how stiff the problem is. Explicit methods such as Tsitouras 4(5) (Tsit45) are a good fit for non-stiff problems. They have inexpensive steps and can choose their step sizes primarily from the requested accuracy. Implicit methods such as the backward differentiation formula (BDF) do more work per step, including solving nonlinear systems, but can take much larger stable steps on stiff problems.

The [`performance-solver-comparison`](https://github.com/martinjrobins/diffsol/tree/main/examples/performance-solver-comparison) example solves two different problems, one non-stiff and the other stiff, using an explicit solver (Tsit45) and an implicit solver (BDF). It uses dense `nalgebra` vectors and matrices, with an LU linear solver for BDF:

```rust,ignore
{{#include ../../../examples/performance-solver-comparison/src/main.rs:types}}
```

## Non-stiff Logistic Growth

The logistic equation is smooth and has one natural timescale:

$$
\begin{align}
y'(t) &= 2y(t)\left(1 - \frac{y(t)}{10}\right), \\\\
\qquad y(0) &= 0.5, \\\\
\qquad 0 &\leq t \leq 5.
\end{align}
$$

Its exact solution is

$$
y(t) = \frac{10}{1 + 19e^{-2t}}.
$$

There is no fast transient forcing the solver to use a small stable step. Tsit45 can therefore solve this problem using fewer steps and fewer right-hand-side evaluations than BDF. BDF remains accurate, but its nonlinear solves are unnecessary overhead for this problem, and it may even end up taking many more steps than Tsit45 because it is, by necessity, conservative about changing its step size so as to reduce the number of times the Jacobian must be factorised.

The explicit definition supplies only the right-hand side; BDF needs the implicit definition, which additionally supplies the Jacobian-vector product.

{{#tabs}}
{{#tab name="Explicit"}}

```rust,ignore
{{#include ../../../examples/performance-solver-comparison/src/main.rs:logistic_explicit}}
```

{{#endtab}}
{{#tab name="Implicit"}}

```rust,ignore
{{#include ../../../examples/performance-solver-comparison/src/main.rs:logistic_implicit}}
```

{{#endtab}}
{{#endtabs}}

## Stiff Rapid Relaxation

The second problem has a fast transient and a slow long-term solution:

$$
\begin{align}
y'(t) &= -1000\bigl(y(t) - \cos t\bigr) - \sin t, \\\\
\qquad y(0) &= 0, \\\\
\qquad 0 &\leq t \leq 10.
\end{align}
$$

Its exact solution is

$$
y(t) = \cos t - e^{-1000t}.
$$

The \\(e^{-1000t}\\) component decays almost immediately, leaving the slowly varying \\(\cos t\\) solution. Despite that slow solution, an explicit method remains limited by the fast eigenvalue for stability and must take steps of roughly \\(0.002\\) or smaller. BDF is stable at much larger steps, which lets it follow the slow solution efficiently after the transient.

The explicit and implicit definitions follow the same pattern as logistic growth. The implicit version provides the constant Jacobian-vector product $Jv = -1000v$ required by BDF.

{{#tabs}}
{{#tab name="Explicit"}}

```rust,ignore
{{#include ../../../examples/performance-solver-comparison/src/main.rs:relaxation_explicit}}
```

{{#endtab}}
{{#tab name="Implicit"}}

```rust,ignore
{{#include ../../../examples/performance-solver-comparison/src/main.rs:relaxation_implicit}}
```

{{#endtab}}
{{#endtabs}}

## Running The Comparison

The example runs all four combinations and records the elapsed solve time, final value, time steps, rejected steps, and right-hand-side evaluations. The solve routines differ only in their solver construction: Tsit45 accepts an explicit problem, while BDF accepts an implicit problem and needs the linear-solver type.

{{#tabs}}
{{#tab name="Tsit45"}}

```rust,ignore
{{#include ../../../examples/performance-solver-comparison/src/main.rs:solve_tsit45}}
```

{{#endtab}}
{{#tab name="BDF"}}

```rust,ignore
{{#include ../../../examples/performance-solver-comparison/src/main.rs:solve_bdf}}
```

{{#endtab}}
{{#endtabs}}

## Results

The example writes the results of its latest run to this table:

<div class="table-wrapper">

{{#include stiff_vs_non_stiff_results.html}}

</div>

Wall-clock times depend on the machine and build profile, but they should always show similar trends, and the number of steps and RHS evals should be stable.
For logistic growth, the explicit solver needs far fewer steps than BDF, likely because the BDF solver is conservative when adapting its step size for a given problem. Each step is roughly the same wall time between the two solvers, while BDF needs to solve a nonlinear problem at each step (requires a factorisation of the Jacobian) due to the non-linear problem being rather simple it often takes less RHS evaluations to solve this problem, leading to an overall lower number of evaluations than Tsit45.

For rapid relaxation, Tsit45 needs significantly more stability-limited steps and RHS evaluations, leading to an order of magnitude difference in the wall-times between the two solvers.
The take home message is clear, use an explicit solver for smooth non-stiff systems, and prefer BDF when fast and slow timescales make the system stiff.
