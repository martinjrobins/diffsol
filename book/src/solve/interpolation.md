# Interpolation

Often you will want to get the solution at a specific time \\(t_o\\), which is probably different to the internal timesteps chosen by the solver. To do this, you can use the `step` method to first step the solver forward in time until you are just beyond \\(t_o\\), and then use the `interpolate` method to get the solution at \\(t_o\\). 

```rust,ignore
{{#include ../../../examples/intro-logistic-closures/src/solve_interpolate.rs}}
```