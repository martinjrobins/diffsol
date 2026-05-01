# Example: Epidemic SIR with policy switching

The SIR model is a simple epidemic model that divides a population into susceptible, infected, and recovered compartments. In this example we will use it to illustrate a hybrid model where the transmission rate changes when a public-health policy switches on or off.

Let

\\[
\mathbf{y} = \begin{bmatrix} S \\\\ I \\\\ R \end{bmatrix}
\\]

where \\(S\\) is the susceptible population, \\(I\\) is the infected population, and \\(R\\) is the recovered population. For a population of size \\(P\\), recovery rate \\(\gamma\\), and transmission rate \\(\beta\\), the standard SIR equations are

\\[
\begin{align}
\frac{dS}{dt} &= -\beta \frac{SI}{P}, \\\\
\frac{dI}{dt} &= \beta \frac{SI}{P} - \gamma I, \\\\
\frac{dR}{dt} &= \gamma I.
\end{align}
\\]

To model policy switching, we will make \\(\beta\\) depend on a discrete model index \\(N\\). The open-policy mode has a larger transmission rate, and the lockdown mode has a smaller transmission rate:

\\[
\beta_N =
\begin{cases}
\beta_\text{open}, & N = 0, \\\\
\beta_\text{lockdown}, & N = 1.
\end{cases}
\\]

The policy uses hysteresis so that lockdown is not switched on and off at the same threshold. Starting in the open-policy mode, lockdown starts when the infected population rises above a high threshold \\(I_\text{high}\\). Once lockdown is active, it remains active until the infected population falls below a lower threshold \\(I_\text{low}\\):

\\[
N^+ =
\begin{cases}
1, & N^- = 0 \text{ and } I = I_\text{high}, \\\\
0, & N^- = 1 \text{ and } I = I_\text{low}.
\end{cases}
\\]

This can be encoded with a single mode-dependent stopping condition,

\\[
\mathbf{s}(\mathbf{y}) = I - \mathbf{h},
\\]

where

\\[
h =
\begin{bmatrix}
I_\text{low} \\\\
I_\text{high}
\end{bmatrix}
\\]

In DiffSL we write the two transmission rates and two thresholds as vectors indexed by `N`. We implement the changes in model index \\(N\\) by setting the model index to whichever root index is true for \\(\mathbf{s}\\). As for the dosing model, we use a [`Solution`](https://docs.rs/diffsol/latest/diffsol/ode_solver/solution/struct.Solution.html) struct and [`solve_soln`](https://docs.rs/diffsol/latest/diffsol/ode_solver/method/trait.OdeSolverMethod.html#method.solve_soln) methods to stop through the different segments of the hybrid model, after each segment using the root index found (`root_idx`) to set the value for \\(N\\).

```rust,ignore
{{#include ../../../examples/epidemic-sir-policy-switching/src/main.rs}}
```

{{#include images/sir-policy-switching.html}}
