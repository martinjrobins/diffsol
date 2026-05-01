# Example: Dosing Protocols

In [compartmental models of drug delivery](compartmental_models_of_drug_delivery.md) we introduced a simple two-compartment model of drug delivery. The model consists of a central compartment and a peripheral compartment. Transitions between the two compartments and the elimination of the drug from the central compartment are governed by the following equations:

\\[
\frac{dq_c}{dt} = \text{Dose}(t) - \frac{q_c}{V_c} CL - Q_{p1} \left ( \frac{q_c}{V_c} - \frac{q_{p1}}{V_{p1}} \right ),
\\]

\\[
\frac{dq_{p1}}{dt} =  Q_{p1} \left ( \frac{q_c}{V_c} - \frac{q_{p1}}{V_{p1}} \right ).
\\]

We want to use a hybrid model with stopping conditions and a reset condition to model a three dose protocol where a separate dose level is administered at specific times. To do this, we can encode the stopping conditions as follows:

\\[
\mathbf{s}(\mathbf{y}) = \begin{bmatrix}
1 \\\\
t - t_2 \\\\
t - t_3
\end{bmatrix}
\\]

where \\(t_1, t_2, t_3\\) are the times at which doses are administered. Note that we start the model at \\(t_1\\). The first stopping condition is never satisfied because we never want to transition to \\(N=0\\) (i.e. this is the starting value), but the other stopping conditions will be satisfied at the appropriate times. The reset condition can then be defined as follows:

\\[
\mathbf{r}(\mathbf{y}) = \begin{bmatrix}
q_c + \mathbf{d_N} \\\\
q_{p1}
\end{bmatrix}
\\]

where \\(\mathbf{d}\\) is a vector of length 3, and \\(\mathbf{d}_N\\) is the \\(N\\)th dose level. This means that when the stopping condition is satisfied, the amount of drug in the central compartment is increased by the appropriate dose level, while the amount of drug in the peripheral compartment remains unchanged.

Finally we give the initial conditions for the model based on the first dose level as follows:

\\[
\mathbf{y}(0) = \begin{bmatrix}\mathbf{d}_1 \\\\
0
\end{bmatrix}
\\]

To implement this in diffsol we will use the diffsl language to write down the ODE equations, stop conditions and reset functions. Then we will use the [`Solution`](https://docs.rs/diffsol/latest/diffsol/ode_solver/solution/struct.Solution.html) and [`solve_soln`](https://docs.rs/diffsol/latest/diffsol/ode_solver/method/trait.OdeSolverMethod.html#method.solve_soln) methods to stop through the `N` segments of the hybrid model. 

`Solution` is a struct for iteratively building up a solution from multiple segments, and `solve_soln` is the method that (a) runs a solve for each segment, and (b) appends the solve result to the `Solution` arg. After each segment we move out the internal state of the solver. This drops the solver, which contains a reference to the problem, so that we can then mutate the problem by setting the new model index based on the index of the root found in the previous call to `solve_soln`. Then we can apply our reset condition to the state, and create another solver with this state in order to continue the iteration.

Note that we could have mutated other aspects of the problem/solver at each stage if we wished, setting parameters, changing solver methods or tolerances, based on how we wanted to implement our hybrid model. In this case our only rule is that `N` is to be set to the index of the previous found root, *before* the reset is applied.

```rust,ignore
{{#include ../../../examples/compartmental-models-drug-delivery-hybrid/src/main.rs}}
```

{{#include images/drug-delivery-hybrid.html}}