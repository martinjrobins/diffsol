# Fitting a Predator-Prey Model to data

In this example we will again use the [Lotka-Volterra equations](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations), which are described in more detail in the [Population Dynamics - Predator-Prey Model](./population_dynamics.md) example.

The equations are

\\[
\frac{dx}{dt} = a x - b x y \\\\
\frac{dy}{dt} = c x y - d y
\\]

We also have initial conditions for the populations at time \\(t = 0\\).

\\[
x(0) = x0 \\\\
y(0) = y0
\\]

This model has six parameters, \\(a, b, c, d, x0, y0\\). For the purposes of this example, we'll fit two of these parameters \\(b, d)\\) to some synthetic data. We'll use the model itself to generate the synthetic data, so we'll know the true values of the parameters to verify the fitting process.

We'll use the argmin crate to perform the optimisation. This is a popular rust crate that contains a number of optimisation algorithms. It does have some limitations, such as the lack of support for constraints, so it may not be suitable for many real ODE fitting problems as the solver can easily fail to converge if the parameter vector moves into a difficult region (e.g. the Lotka-Volterra model only makes sense for positive values of the parameters). However, it will be sufficient for this example to demonstrate the sensitivity analysis capabilities of `diffsol`.

First of all we will need to implement some of the argmin traits to specify the optimisation problem. We'll create a `struct Problem` and implement the `CostFunction` and `Gradient` traits for it. The `Problem` struct will hold our synthetic data (held by the `ys_data` and `ts_data` fields) and the `OdeSolverProblem` that we'll use to solve the ODEs. We'll also create some type aliases for the nalgebra and diffsol types we'll be using.

Note that the `problem` field of the `Problem` struct is wrapped in a `RefCell` so that we can mutate it in the `cost` and `gradient` methods. Setting the parameters of the ODE solver problem is a mutable operation (i.e. you are changing the equations), so we need to use `RefCell` and interior mutability to do this.

```rust,ignore
{{#include ../../../examples/predator-prey-fitting-forward/src/main_llvm.rs::24}}
```

The argmin `CostFunction` trait requires an implementation of the `cost` method, which will calculate the sum-of-squares difference between the synthetic data and the model output. Since the argmin crate does not support constraints, we'll return a large value if the ODE solver fails to converge.

```rust,ignore
{{#include ../../../examples/predator-prey-fitting-forward/src/main_llvm.rs:26:45}}
```

The argmin `Gradient` trait requires an implementation of the `gradient` method, which will calculate the gradient of the cost function with respect to the parameters. Our sum-of-squares cost function can be written as

\\[
\text{loss} = \sum_i (y_i(p) - \hat{y}_i)^2
\\]

where \\(y_i(p)\\) is the model output as a function of the parameters \\(p\\), and \\(\hat{y}_i\\) is the observed data at time index \\(i\\). Threrefore, the gradient of this cost function with respect to the parameters is

\\[
\frac{\partial \text{loss}}{\partial p} = 2 \sum_i (y_i(p) - \hat{y}_i) \cdot \frac{\partial y_i}{\partial p}
\\]

where \\(\frac{\partial y_i}{\partial p}\\) is the sensitivity of the model output with respect to the parameters. We can calculate this sensitivity using the `solve_dense_sensitivities` method of the ODE solver. The gradient of the cost function is then the sum of the dot product of the residuals and the sensitivities for each time point. Again, if the ODE solver fails to converge, we'll return a large value for the gradient.

```rust,ignore
{{#include ../../../examples/predator-prey-fitting-forward/src/main_llvm.rs:47:70}}
```

With these implementation out of the way, we can now perform the fitting problem. We'll generate some synthetic data using the Lotka-Volterra equations with some true parameters, and then fit the model to this data. We'll use the `LBFGS` solver from the argmin crate, which is a quasi-Newton method that uses the Broyden-Fletcher-Goldfarb-Shanno (BFGS) update formula. We'll also use the `SlogLogger` observer to log the progress of the optimisation.

We'll initialise the optimizer a short distance away from the true parameter values, and then check the final optimised parameter values against the true values.

```rust,ignore
{{#include ../../../examples/predator-prey-fitting-forward/src/main_llvm.rs:72::}}
```

```
Feb 03 13:16:44.604 INFO L-BFGS
Feb 03 13:16:44.842 INFO iter: 0, cost: 21.574177406963013, best_cost: 21.574177406963013, cost_count: 6, gradient_count: 7, gamma: 1, time: 0.238573908
Feb 03 13:16:44.920 INFO iter: 1, cost: 0.6811721224055488, best_cost: 0.6811721224055488, cost_count: 8, gradient_count: 10, time: 0.077082344, gamma: 0.00036901099013336356
Feb 03 13:16:44.969 INFO iter: 2, cost: 0.6478536174002669, best_cost: 0.6478536174002669, cost_count: 9, gradient_count: 12, time: 0.049218286, gamma: 0.00017983731521908368
Feb 03 13:16:45.018 INFO iter: 3, cost: 0.5515637814971768, best_cost: 0.5515637814971768, cost_count: 10, gradient_count: 14, time: 0.049264513, gamma: 0.00013404417466199433
Feb 03 13:16:45.069 INFO iter: 4, cost: 0.2889819270908579, best_cost: 0.2889819270908579, cost_count: 11, gradient_count: 16, time: 0.050718659, gamma: 0.00019004425867568796
Feb 03 13:16:45.120 INFO iter: 5, cost: 0.06441855702549167, best_cost: 0.06441855702549167, cost_count: 12, gradient_count: 18, gamma: 0.0005522578375180803, time: 0.05102388
Feb 03 13:16:45.172 INFO iter: 6, cost: 0.001969603423448309, best_cost: 0.001969603423448309, cost_count: 13, gradient_count: 20, time: 0.051874014, gamma: 0.002084311472606979
Feb 03 13:16:45.224 INFO iter: 7, cost: 0.00018682781933202676, best_cost: 0.00018682781933202676, cost_count: 14, gradient_count: 22, gamma: 0.00020342834067043386, time: 0.051705468
Feb 03 13:16:45.276 INFO iter: 8, cost: 0.0000004145187755175781, best_cost: 0.0000004145187755175781, cost_count: 15, gradient_count: 24, time: 0.052326582, gamma: 0.00013653160933392906
Feb 03 13:16:45.328 INFO iter: 9, cost: 0.00000019683422285908518, best_cost: 0.00000019683422285908518, cost_count: 16, gradient_count: 26, time: 0.05250708, gamma: 0.0002897867622209216
Feb 03 13:16:45.381 INFO iter: 10, cost: 0.00000018573267264008743, best_cost: 0.00000018573267264008743, cost_count: 17, gradient_count: 28, time: 0.052503364, gamma: 0.0006940640307853263
Feb 03 13:16:47.047 INFO iter: 11, cost: 0.0000001857326722089365, best_cost: 0.0000001857326722089365, cost_count: 74, gradient_count: 86, gamma: 0.00012163138683631344, time: 1.665703414
Feb 03 13:16:47.531 INFO iter: 12, cost: 0.00000018573267291060315, best_cost: 0.0000001857326722089365, cost_count: 90, gradient_count: 103, gamma: 0.000001225820261527594, time: 0.484634742
Feb 03 13:16:49.078 INFO iter: 13, cost: 0.000000185732672314337, best_cost: 0.0000001857326722089365, cost_count: 143, gradient_count: 157, time: 1.5466635069999999, gamma: 0.00000005014738728081247
Feb 03 13:16:49.562 INFO iter: 14, cost: 0.00000018573267355654775, best_cost: 0.0000001857326722089365, cost_count: 159, gradient_count: 174, gamma: 0.00000386732692846693, time: 0.483722743
Feb 03 13:16:51.051 INFO iter: 15, cost: 0.0000001857326728225398, best_cost: 0.0000001857326722089365, cost_count: 210, gradient_count: 226, time: 1.489475966, gamma: -0.00000029042697044185633
Feb 03 13:16:51.880 INFO iter: 16, cost: 0.00000018573267387423262, best_cost: 0.0000001857326722089365, cost_count: 238, gradient_count: 255, time: 0.828862625, gamma: 0.0000014262616055754604
Feb 03 13:16:53.202 INFO iter: 17, cost: 0.00000018573267376814762, best_cost: 0.0000001857326722089365, cost_count: 283, gradient_count: 301, gamma: 0.00000006448722374243354, time: 1.32160584
OptimizationResult:
    Solver:        L-BFGS
    param (best):  [1.3333663799297248, 1.0000008551827637]
    cost (best):   0.0000001857326722089365
    iters (best):  11
    iters (total): 18
    termination:   Solver converged
    time:          8.62392681s

Best parameter vector: [1.3333663799297248, 1.0000008551827637]
True parameter vector: [1.3333333333333333, 1.0]
```

So, we've successfully fitted the Lotka-Volterra model to some synthetic data and recovered the original true parameters. This is a simple example and could easily be improved. For example, you will note from the output that the argmin crate is calling both the cost and gradient functions, and this is often done using the exact same parameter vector. Ideally we'd like to cache the results of the `solve_dense_sensitivities` method and reuse them in both the `cost` and `gradient` functions.
