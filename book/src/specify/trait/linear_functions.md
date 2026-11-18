# Linear functions

Naturally, we can also implement the `LinearOp` trait if we want to include a mass matrix in our model. A common use case for implementing this trait is to represent the mass matrix with a custom struct. The `gemv_inplace` method must calculate \\(y = M x + \beta y\\). In this example, the mass matrix is the \\(1 \times 1\\) identity matrix:

```rust,ignore
{{#include ../../../../examples/custom-ode-equations/src/my_mass.rs}}
```
