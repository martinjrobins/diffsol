# Example: Fitting a spring-mass model to data

In this example we'll fit a damped spring-mass system to some synthetic data (using the model to generate the data). The system consists of a mass \\(m\\) attached to a spring with spring constant \\(k\\), and a damping force proportional to the velocity of the mass with damping coefficient \\(c\\).

\\[
\begin{align*}
\frac{dx}{dt} &= v \\\\
\frac{dv}{dt} &= -\frac{k}{m} x - \frac{c}{m} v
\end{align*}
\\]

where \\(v = \frac{dx}{dt}\\) is the velocity of the mass.

We'll use the argmin crate to perform the optimisation. To hold the synthetic data and the model, we'll create a `struct Problem` like so

```rust,ignore
{{#include ../../../examples/mass-spring-fitting-adjoint/src/main_llvm.rs:problem}}
```

To use argmin we need to specify traits giving the loss function and its gradient. In this case we'll define a loss function equal to the sum of squares error between the model output and the synthetic data.

$$
\text{loss} = \sum_i (y_i(p) - \hat{y}_i)^2
$$

where \\(y_i(p)\\) is the model output as a function of the parameters \\(p\\), and \\(\hat{y}_i\\) is the observed data at time index \\(i\\).

```rust,ignore
{{#include ../../../examples/mass-spring-fitting-adjoint/src/main_llvm.rs:cost_function}}
```

The gradient of this cost function with respect to the model outputs \\(y_i\\) is

$$
\frac{\partial \text{loss}}{\partial y_i} = 2 (y_i(p) - \hat{y}_i)
$$

Diffsol gives two methods to calculate the gradient of the loss with respect to the parameters \\(p\\). Both use reverse mode, so the cost does not increase with the number of parameters.

## Method 1: continuous adjoint

The continuous adjoint method ("optimise-then-discretise") first derives the adjoint ODE equations, and then solves them backwards in time with an ODE solver. First we solve the forwards problem, which gives a checkpointing struct. We then use the forward solution to calculate \\(\frac{\partial \text{loss}}{\partial y_i}\\) at each time point, and give this to the adjoint backwards pass. The backwards pass calculates the gradient of the loss with respect to the parameters.

```rust,ignore
{{#include ../../../examples/mass-spring-fitting-adjoint/src/main_llvm.rs:gradient_continuous}}
```

The `dloss_dy` helper gives the matrix of \\(\frac{\partial \text{loss}}{\partial y_i}\\), with one column for each time point.

## Method 2: reverse-mode AD of the solver steps

The second method ("discretise-then-optimise") uses the [`StepVjp`](https://docs.rs/diffsol/latest/diffsol/ode_solver/step_ad/trait.StepVjp.html) trait. It does not solve an adjoint ODE. Instead, it pulls the loss gradient back through each step that the forward solver took, in reverse order. The step sizes, the orders and the accept/reject decisions of the forward solve are kept fixed. Thus the result is the gradient of the *numerical* solution, not an approximation of the gradient of the exact solution.

The code has the same shape as the continuous method. `solve_dense_vjp_fwd` solves the forwards problem and keeps the state after each step in a tape. We calculate \\(\frac{\partial \text{loss}}{\partial y_i}\\) as before. Then `solve_dense_vjp_bkwd` uses the tape to calculate the gradient with respect to the parameters.

```rust,ignore
{{#include ../../../examples/mass-spring-fitting-adjoint/src/main_llvm.rs:gradient_discrete}}
```

The `Gradient` trait for argmin selects one of the two methods:

```rust,ignore
{{#include ../../../examples/mass-spring-fitting-adjoint/src/main_llvm.rs:gradient}}
```

## Comparing the two methods

In our main function we make the model and generate some synthetic data. Then we compare the two methods in two ways:

1. We calculate the gradient at the initial parameters 1000 times with each method, and measure the mean time for one gradient.
2. We use argmin's L-BFGS solver to fit the model to the data with each method, and measure the time for the full fit.

```rust,ignore
{{#include ../../../examples/mass-spring-fitting-adjoint/src/main_llvm.rs:main}}
```

Run the example in release mode, because timings of a debug build are not useful:

```sh
cargo run --release -p mass-spring-fitting-adjoint --features diffsl-llvm17
```

The output is:

```
Gradient at [0.9, 0.09000000000000001]:
  ContinuousAdjoint: [-169.961431321948, -106.83268194215616], 2.031135ms per gradient
  DiscreteAdjoint: [-169.96068853944487, -106.80930261280977], 218.84µs per gradient
  Largest relative difference: 2.19e-4
  Speed-up of discrete over continuous: 9.28x

Fit with ContinuousAdjoint:
  time:           104.769459ms
  iterations:     13
  cost evals:     27
  gradient evals: 40
  best cost:      7.179238912566991e-25
  best params:    [0.9999999999999731, 0.10000000000000907]
  termination:    Solver converged

Fit with DiscreteAdjoint:
  time:           24.460875ms
  iterations:     12
  cost evals:     29
  gradient evals: 41
  best cost:      1.6127427301319806e-20
  best params:    [1.000000000003015, 0.09999999999754586]
  termination:    Solver converged

True parameters: [1.0, 0.1]
```

The times depend on your machine, but the ratio between the two methods is a better guide. For this problem, one gradient with reverse-mode AD of the steps is approximately 9 times faster than one gradient with the continuous adjoint. The continuous adjoint must solve a second ODE backwards in time, with its own adaptive steps and error control. Reverse-mode AD of the steps does not take new steps: it goes back through the steps of the forward solve that it kept in the tape. The full fit is approximately 4 times faster. The speed-up is smaller because the fit also calculates the loss, which costs the same for the two methods.

The two gradients agree to approximately \\(2 \times 10^{-4}\\) relative error. This difference comes from (a) the continuous adjoint method: it approximates the gradient of the exact ODE solution to the solver tolerances and (b) the discrete gradient method: we use an implicit solver so the linear solve for each VJP is only solved to the tolerances (an explicit solver will give an exact gradient up to floating point precision). The two fits converge to the true parameters, and they use a similar number of iterations.
