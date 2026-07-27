# Events and Multistep Solvers

Repeated discontinuities can change which stiff solver is most efficient. A
multistep method such as BDF uses solution history to achieve high order. A
state jump invalidates that history, so BDF must restart at first order. A
one-step method such as ESDIRK34 has no multistep history to discard and can
continue with its fixed third-order formula after an event. This *may* mean
that BDF performs poorly in comparison with a one-step method, but this depends
on many factors, even with the restart BDF might perform significantly better between events,
so it is always worth trying both methods to see which is better.

The
[`performance-solver-events`](https://github.com/martinjrobins/diffsol/tree/main/examples/performance-solver-events)
example constructs a simple example that highlights the slowdown effect that can affect the BDF solver.
It compares BDF with ESDIRK34 using dense `nalgebra` vectors and matrices and an LU linear solver:

```rust,ignore
{{#include ../../../examples/performance-solver-events/src/main.rs:types}}
```

## A Stiff Slow Manifold

The model is a stiff system with a slowly varying solution.
We will perturb the system at regular events in such a way as to introduce discontinuities while maintaining a slowly varying solution
(previous versions of this problem showed that the BDF solver did better at fast transients after the event, reducing the effect we are trying to demonstrate).

$$
\begin{aligned}
y_1' &= -y_1, \\\\
y_2' &= 999y_1 - 1000y_2,
\end{aligned}
$$

with \\(y(0)=(1,1)^T\\). Its eigenvalues are \\(-1\\) and \\(-1000\\), giving a
stiffness ratio of \(1000\). The vector \((1,1)^T\) is the slow eigenvector.

At the event times

$$
t_k = 0.05k, \qquad k=1,\ldots,100,
$$

the example applies the jump

$$
y(t_k^+) = y(t_k^-) + \begin{pmatrix}1\\\\1\end{pmatrix}.
$$

Each jump stays on the slow eigenspace so we don't have any large fast transients.
This is done so the benchmark is dominated by the cost of BDF's history restart,
rather than the cost of repeatedly resolving the fast transients.

The model definition is:

```rust,ignore
{{#include ../../../examples/performance-solver-events/src/main.rs:model}}
```

## Restarting

After every event, the example updates both the state and its derivative. This
keeps the solver state consistent before BDF reconstructs its first-order
history. The loop stops exactly at each event, applies the jump, and then continues:

{{#tabs}}
{{#tab name="ESDIRK34"}}

```rust,ignore
{{#include ../../../examples/performance-solver-events/src/main.rs:run_esdirk34}}
```

{{#endtab}}
{{#tab name="BDF"}}

```rust,ignore
{{#include ../../../examples/performance-solver-events/src/main.rs:run_bdf}}
```

{{#endtab}}
{{#endtabs}}

## Results

The benchmark warms up both solvers and performs eleven timed runs of each.
The example writes its most recent comparison to this table:

<div class="table-wrapper">

{{#include events_and_multistep_solvers_results.html}}

</div>

Wall-clock times depend on the machine, but this problem should show
ESDIRK34 requiring substantially fewer steps than BDF at tight tolerances.
BDF must repeatedly climb from order one after each jump instead of preserving
high-order history over the full solve. However, it is not a general rule that
one-step methods always beat BDF for event-driven stiff systems: repeatedly
exciting a large fast mode can favour BDF instead, as we found for another version of
this same problem.
