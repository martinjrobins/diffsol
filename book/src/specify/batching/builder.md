# Builder

Each vector, matrix or operation in diffsol has its own context, which is used to configure the number of batches (as well as other backend-specific information). You can create a new context with a set number of batches using the `with_nbatch` method of the context type (e.g. [`NalgebraContext::with_nbatch`](https://docs.rs/diffsol/latest/diffsol/struct.NalgebraContext.html#method.with_nbatch)).

```rust,ignore
{{#include ../../../../examples/batching/src/main.rs:context}}
```

Give this context to the [`OdeBuilder::context`](https://docs.rs/diffsol/latest/diffsol/ode_solver/builder/struct.OdeBuilder.html#method.context) method to create a new `OdeSolverProblem` with the specified number of batches. When you set the parameters, give a vector of values of length `nparams * nbatch`. Diffsol splits this vector into `nbatch` blocks of length `nparams`, one block for each batch lane.

Note that you can ignore batching when specifying your rhs, init and other equation closures. Each closure will be called once for each batch lane and will handle the specific broadcasting rules used by diffsol. The input arguments for each closure are `&[T]` slices that correspond to a single batch lane. Each closure does not return a value. It writes its result into a mutable `&mut [T]` output slice argument for the same batch lane.

```rust,ignore
{{#include ../../../../examples/batching/src/main.rs:builder}}
```
