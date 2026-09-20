# GPU Parallelism

```rust,ignore
{{#include ../../../examples/performance-gpu-ensemble/src/swing.rs:constants}}
```

```rust,ignore
{{#include ../../../examples/performance-gpu-ensemble/src/swing.rs:equations}}
```

```rust,ignore
{{#include ../../../examples/performance-gpu-ensemble/src/swing.rs:problem}}
```

```rust,ignore
{{#include ../../../examples/performance-gpu-ensemble/src/ensemble.rs:types}}
```

```rust,ignore
{{#include ../../../examples/performance-gpu-ensemble/src/ensemble.rs:sample}}
```

```rust,ignore
{{#include ../../../examples/performance-gpu-ensemble/src/ensemble.rs:solve_gpu}}
```

```rust,ignore
{{#include ../../../examples/performance-gpu-ensemble/src/ensemble.rs:reduce}}
```

```rust,ignore
{{#include ../../../examples/performance-gpu-ensemble/src/ensemble.rs:plot_histogram}}
```

{{#include images/gpu_ensemble_frequency.html}}

## GPU versus CPU

```rust,ignore
{{#include ../../../examples/performance-gpu-ensemble/src/ensemble.rs:solve_cpu}}
```

```rust,ignore
{{#include ../../../examples/performance-gpu-ensemble/src/ensemble.rs:plot_scaling}}
```

{{#include images/gpu_ensemble_scaling.html}}
