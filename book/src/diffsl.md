# DiffSL

Thus far we have used Rust code to specify the problem we want to solve. This is fine if you are using DiffSol from Rust, but what if you want to use DiffSol from a higher-level language like Python or R?
For this usecase we have designed a Domain Specific Language (DSL) called DiffSL that can be used to specify the problem. DiffSL is not a general purpose language but is tightly constrained to 
the specification of a system of ordinary differential equations. It features a relativly simple syntax that consists of writing a series of tensors (dense or sparse) that represent the equations of the system.
For more detail on the syntax of DiffSL see the [DiffSL book](https://martinjrobins.github.io/diffsl/). This section will focus on how to use DiffSL to specify a problem in DiffSol.


## DiffSL Context

The main struct that is used to specify a problem in DiffSL is the [`DiffSlContext`](https://docs.rs/diffsol/latest/diffsol/ode_solver/diffsl/struct.DiffSlContext.html) struct. Creating this struct
Just-In-Time (JIT) compiles your DiffSL code into a form that can be executed efficiently by DiffSol. 

```rust
# fn main() {
use diffsol::DiffSlContext;
type M = nalgebra::DMatrix<f64>;
        
let context = DiffSlContext::<M>::new("
    in = [r, k]
    r { 1.0 }
    k { 1.0 }
    u { 0.1 }
    F { r * u * (1.0 - u / k) }
    out { u }
").unwrap();
# }
```


```rust
# fn main() {
# use diffsol::DiffSlContext;
use diffsol::{OdeBuilder, Bdf, OdeSolverMethod};
# type M = nalgebra::DMatrix<f64>;

        
# let context = DiffSlContext::<M>::new("
#     in = [r, k]
#     r { 1.0 }
#     k { 1.0 }
#     u { 0.1 }
#     F { r * u * (1.0 - u / k) }
#     out { u }
# ").unwrap();
let problem = OdeBuilder::new()
.rtol(1e-6)
.p([1.0, 10.0])
.build_diffsl(&context).unwrap();
let mut solver = Bdf::default();
let t = 0.4;
let _soln = solver.solve(&problem, t).unwrap();
# }
```
