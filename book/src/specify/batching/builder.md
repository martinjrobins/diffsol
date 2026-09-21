# Builder

Each vector, matrix or operation in diffsol has its own context, which is used to configure the number of batches (as well as other backend-specific information). You can create a new context with a set number of batches using the `Context::with_batch` method.

```rust,ignore
{{#include ../../../../examples/batching/src/main.rs:context}}
```

This context can be passed to the `OdeBuilder` in order to create a new `OdeSolverProblem` with the specified number of batches. When setting the parameters, you can pass in a vector of values of length `nstates * nbatch`, which will be split into `nbatch` separate state vectors of length `nstates`.

Note that you can ignore batching when specifying your rhs, init and other equation closures. Each closure will be called once for each batch lane and will handle the specific broadcasting rules used by diffsol. The input arguments for each closure are simply `T` slices that correspond to a single batch lane, and the return values are mutable `T` slices of the same length.

```rust,ignore
{{#include ../../../../examples/batching/src/main.rs:builder}}
```
