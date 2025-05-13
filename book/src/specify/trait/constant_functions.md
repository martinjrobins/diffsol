# Constant functions

Now we've implemented the rhs function, but how about the initial condition? We can implement the `ConstantOp` trait to specify the initial condition. Since this is quite similar to the `NonLinearOp` trait, we will do it all in one go.

```rust,ignore
{{#include ../../../../examples/custom-ode-equations/src/my_init.rs}}
```