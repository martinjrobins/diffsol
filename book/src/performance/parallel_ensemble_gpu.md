# GPU Parallelism

The model used in this example represents `nbuses` electrical generators, each tied to all the others through a network of transmission lines. Each generator bus is described by the swing equation, a second-order ODE describing the dynamics of the generator's rotor angle and frequency. The implementation writes this as a system of first-order ODEs.

We introduce uncertainty into the model by adding an uncertain additional demand at bus 0, drawn from a normal distribution. The goal is to solve many instances of the model and obtain the maximum speed deviation for each ensemble member.

We will solve this ensemble of models in parallel, both with the GPU and CPU, using diffsol's batching features and using the `cuda-oxide` and `nalgebra` backends.

The swing equation is a non-linear ODE, and is given by:

$$
\begin{align}
\frac{d\delta_i}{dt} &= \omega_i \\\\
M \frac{d\omega_i}{dt} &= -D\omega_i - B\sum_{j=0}^{n-1} \sin(\delta_i-\delta_j) - d\ \mathbf{1}_{i=0}
\end{align}
$$

where:

- \\(\delta_i\\): rotor angle at bus \\(i\\) in the reference frame, radians
- \\(\omega_i\\): speed deviation at bus \\(i\\) from the reference frame, radians per second
- \\(B\\): pairwise line susceptance (how strongly each transmission line transfers power between two buses)
- \\(d\\): uncertain additional demand at bus \\(0\\)
- \\(\mathbf{1}_{i=0}\\): equals \(1\) at bus \\(0\\), otherwise \\(0\\)

First, we write the model using diffsol's `OdeEquations` trait. We cannot use the easier builder and closures API because it is not supported for GPU models.

We will define the generator inertia `M`, damping `D` and per-line susceptance `B`:

```rust,ignore
{{#include ../../../examples/performance-gpu-ensemble/src/swing.rs:constants}}
```

Then we define the swing equations over a batched ensemble using a single `SwingEqn` struct with one demand value per batch lane. We hold the demand values in a vector `demand` on the struct. The network has `nbuses` generators, each tied to every other. State `i < nbuses` is the rotor angle of bus `i`, and state `nbuses + g` is the speed deviation of bus `g`, so each state vector holds `2 * nbuses` states.

We define the equations using the `for_each_elem` method, which is called for each element `i` of the state vector in each batch lane. Since the rotor angles are in the first half of the state vector and the speed deviations are in the second, we can use the index `i` to determine which equation to evaluate.

We will not show much of the boilerplate code here, but the full code is available in the `examples/performance-gpu-ensemble` directory of the diffsol repository.

```rust,ignore
{{#include ../../../examples/performance-gpu-ensemble/src/swing.rs:equations}}
```

Once we have defined our equations, we can create an `OdeSolverProblem` using the builder. To set up the batches on the GPU, we create a vector of `N_SAMPLES` demand values drawn from a normal distribution with `DEMAND_MEAN = 0.3` and `DEMAND_SD = 0.08`, then pass them to the builder through the `demands` argument.

```rust,ignore
{{#include ../../../examples/performance-gpu-ensemble/src/swing.rs:problem}}
```

We can then solve the ensemble of ODEs on the GPU using the standard diffsol solve API. Under the hood, diffsol solves `nbatch` ODEs in lockstep on the GPU using the `cuda-oxide` backend.

```rust,ignore
{{#include ../../../examples/performance-gpu-ensemble/src/ensemble.rs:solve_gpu}}
```

We want to display the distribution of maximum speed deviation across the ensemble of ODEs, so we use the lower-level `step` API to advance the equations in time. At each time step, `reduce_elem` computes one maximum speed deviation per batch lane, avoiding the need to hold the entire trajectory in memory.

```rust,ignore
{{#include ../../../examples/performance-gpu-ensemble/src/ensemble.rs:reduce}}
```

We can then plot the histogram of the maximum speed deviation across the ensemble, which is shown below:

{{#include images/gpu_ensemble_frequency.html}}

## GPU versus CPU

It is useful to compare the GPU batch solves against a CPU parallel implementation using the `rayon` crate. The CPU version uses `map_init` to create a `SwingEqn` problem for each worker, then solves one ODE per sample in parallel using the thread pool.

```rust,ignore
{{#include ../../../examples/performance-gpu-ensemble/src/ensemble.rs:solve_cpu}}
```

To show the relative performance, we loop over different ensemble sizes and, for each size, time the GPU and CPU solves. We take the median of 15 runs to reduce noise. The results are plotted below.

Below an ensemble size of 100, fixed costs such as kernel launches, allocation, and setup dominate, so GPU execution time is approximately flat for this workload. Between 100 and 500 samples, computation becomes a larger part of the runtime. At 1000 samples, the timing is approximately linear. In this case, the GPU appears to reach its throughput limit before all of its theoretical resident threads are occupied, GPU profiling would be required to confirm the limiting factor here. CPU scaling becomes linear at smaller ensemble sizes because the CPU has fewer workers available for parallel work. Once both CPU and GPU are in the linear regime, the GPU is about twice as fast as the CPU. The A40's relatively low FP64 peak rate, approximately 1/64 of its FP32 rate, is one factor that limits its advantage, others include memory access, kernel structure, and launch overhead.

{{#include images/gpu_ensemble_scaling.html}}
