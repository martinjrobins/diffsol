# Linear functions

Naturally, we can also implement the `LinearOp` trait if we want to include a mass matrix in our model. A common use case for implementing this trait is to store the mass matrix in a custom struct, like so:

```rust,ignore
{{#include ../../../../examples/custom-ode-equations/src/my_mass.rs}}
```
