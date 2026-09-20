# CPU Parallelism

In this section we will demonstrate how to use the popular `rayon` crate to solve many ODEs in parallel on the CPU. The example we will use is a simple population dynamics model, where we will solve the same ODE with different values of the growth parameter, in parallel. To
generate the different growth parameters we will use the `ChaCha` crate to generate random values according to a log-normal distribution.

First of all, lets define a few types and constants that we will use. We will use the `nalgebra` dense matrix type for the linear algebra backend, and then we define:

- the number of ODEs we want to solve in parallel (`N_SAMPLES`),
- the number of time points we want to solve for (`N_TIMES`),
- the final time point we want to solve to (`T_FINAL`),
- the carrying capacity parameter for the population dynamics model (`K`),
- an initial value for the population (`Y0`), and
- the random seed for the random number generator (`SEED`).

```rust,ignore
{{#include ../../../examples/performance-cpu-ensemble/src/main.rs:types}}
```

Now we will define an function `ensemble` that will draw the random growth rates and solve the ensemble of ODEs in parallel. Once the ensemble has been solved, we will reduce the results (also in parallel) to the 5%, 50% and 95% quantiles at each evaluation time. The function will return a tuple of the evaluation times and the quantiles.

Both the solve and reduce stages will run on the rayon thread pool. A single random number generator cannot by shared across threads, but the `ChaCha` crate offers many independent streams per seed, so each sample takes the stream matching its index.

Similarly, the diffsol `Problem` cannot be shared across threads as it needs to be mutated in order to set the growth rate for each sample. It also uses a `RefCell` to store operator statistics, so in Rust terms it is `Send` but not `Sync` (please raise an issue on the repo if `Sync` is required for your work). So that the `Problem` is not shared between worker threads, we will use `map_init` to give each thread its own problem, which will be reused across every sample that this particular thread works on.

```rust,ignore
{{#include ../../../examples/performance-cpu-ensemble/src/main.rs:ensemble}}
```

Once we have called `ensemble` and obtained the results, we can plot the quantiles using the `plotters` crate. The following code will create a Plotly plot with the plot of the 5%, 50% and 95% quantiles of the population dynamics model, which is shown below:

```rust,ignore
{{#include ../../../examples/performance-cpu-ensemble/src/main.rs:plot}}
```

{{#include images/parallel_ensemble_bands.html}}

## Thread Scaling

Now we can examine how effective the parallelism is by varying the number of threads used in the rayon thread pool. The following code will run the ensemble with different numbers of threads, and record the time taken for each run. We will repeat each run a few times and then take the median to reduce benchmark noise.

```rust,ignore
{{#include ../../../examples/performance-cpu-ensemble/src/main.rs:scaling}}
```

Again we will use plotly to plot the results and compare against a reference line indicating perfect linear scaling.

{{#include images/parallel_ensemble_scaling.html}}

While we expect that the actual ODE solve will be linear in the number of threads, there are a few other factors that will reduce the scaling below the linear line, these include:

- problem setup: each thread worker needs to build its own `OdeSolverProblem` in `map_init`.
- reducing the solutions: This requires `N_TIMES` sorts which are `O(N log N)` each.
- memory bandwidth: here we need to allocate, write then read `N_SAMPLES` solution trajectories.
- scheduling overhead: Rayon uses a thread pool and schedules work across it, this creates more work at higher thread counts.
- simple ODE: The logistic growth ODE is trivial to solve, so the linear portion is relatively cheap, increasing the weighting of the other factors above.
