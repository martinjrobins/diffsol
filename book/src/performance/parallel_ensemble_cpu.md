# CPU Parallelism

In this section we demonstrate how to use the popular `rayon` crate to solve many ODEs in parallel on the CPU. The example is a simple population dynamics model: we solve the same ODE with different values of the growth parameter in parallel. To generate the growth parameters, we use `rand_chacha::ChaCha12Rng` to generate random values according to a log-normal distribution.

First, let's define the types and constants that we use. We use the `nalgebra` dense matrix type for the linear algebra backend, and define:

- the number of ODEs we want to solve in parallel (`N_SAMPLES`),
- the number of time points we want to solve for (`N_TIMES`),
- the final time point we want to solve to (`T_FINAL`),
- the carrying capacity parameter for the population dynamics model (`K`),
- an initial value for the population (`Y0`), and
- the random seed for the random number generator (`SEED`).

```rust,ignore
{{#include ../../../examples/performance-cpu-ensemble/src/main.rs:types}}
```

Now we define an `ensemble` function that draws the random growth rates and solves the ensemble of ODEs in parallel. Once the ensemble has been solved, we reduce the results, also in parallel, to the 5%, 50%, and 95% quantiles at each evaluation time. The function returns the quantile bands; the evaluation times are passed in by the caller.

Both the solve and reduce stages run on the Rayon thread pool. A single random number generator cannot be shared across threads, but `ChaCha12Rng` offers independent streams from one seed, so each sample uses the stream matching its index.

Similarly, the diffsol `Problem` cannot be shared across threads as it needs to be mutated in order to set the growth rate for each sample. It also uses a `RefCell` to store operator statistics, so in Rust terms it is `Send` but not `Sync` (please raise an issue on the repo if `Sync` is required for your work). So that the `Problem` is not shared between worker threads, we will use `map_init` to give each thread its own problem, which will be reused across every sample that this particular thread works on.

```rust,ignore
{{#include ../../../examples/performance-cpu-ensemble/src/main.rs:ensemble}}
```

Once we have called `ensemble` and obtained the results, we can plot the quantiles using Plotly. The following code creates a plot of the 5%, 50%, and 95% quantiles of the population dynamics model, shown below:

```rust,ignore
{{#include ../../../examples/performance-cpu-ensemble/src/main.rs:plot}}
```

{{#include images/parallel_ensemble_bands.html}}

## Thread Scaling

Now we can examine how effective the parallelism is by varying the number of threads used in the rayon thread pool. The following code will run the ensemble with different numbers of threads, and record the time taken for each run. We will repeat each run a few times and then take the median to reduce benchmark noise.

```rust,ignore
{{#include ../../../examples/performance-cpu-ensemble/src/main.rs:scaling}}
```

We use Plotly to plot the results and compare them against a reference line indicating ideal linear scaling.

{{#include images/parallel_ensemble_scaling.html}}

Ideal speed-up would be linear with the number of threads, but several factors reduce the measured scaling below the ideal line:

- problem setup: each thread worker needs to build its own `OdeSolverProblem` in `map_init`.
- reducing the solutions: This requires `N_TIMES` sorts which are `O(N log N)` each.
- memory bandwidth: here we need to allocate, write then read `N_SAMPLES` solution trajectories.
- scheduling overhead: Rayon uses a thread pool and schedules work across it, this creates more work at higher thread counts.
- simple ODE: The logistic growth ODE is trivial to solve, so the linear portion is relatively cheap, increasing the weighting of the other factors above.
