# DiffSL

Thus far we have used Rust code to specify the problem we want to solve. This is fine if you are using DiffSol from Rust, but what if you want to use DiffSol from a higher-level language like Python or R?
For this usecase we have designed a Domain Specific Language (DSL) called DiffSL that can be used to specify the problem. DiffSL is not a general purpose language but is tightly constrained to
the specification of a system of ordinary differential equations. It features a relatively simple syntax that consists of writing a series of tensors (dense or sparse) that represent the equations of the system.
For more detail on the syntax of DiffSL see the [DiffSL book](https://martinjrobins.github.io/diffsl/). This section will focus on how to use DiffSL to specify a problem in DiffSol.

## DiffSL Context

The main struct that is used to specify a problem in DiffSL is the [`DiffSl`](https://docs.rs/diffsol/latest/diffsol/ode_solver/diffsl/struct.DiffSl.html) struct. Creating this struct
Just-In-Time (JIT) compiles your DiffSL code into a form that can be executed efficiently by DiffSol.

```rust,ignore
{{#include ../../../examples/intro-logistic-diffsl/src/main.rs::17}}
```

The `CG` parameter specifies the backend that you want to use to compile the DiffSL code. The `CraneliftModule` backend is behind the `diffsl-cranelift` feature flag. If you want to use the faster LLVM backend you can use the `LlvmModule` backend, which is behind one of the `diffsl-llvm*` feature flags, depending on the version of LLVM you have installed.

Once you have created the `DiffSl` struct you can use it to create a problem using the `build_from_eqn` method on the [`OdeBuilder`](https://docs.rs/diffsol/latest/diffsol/ode_solver/builder/struct.OdeBuilder.html) struct.

```rust,ignore
{{#include ../../../examples/intro-logistic-diffsl/src/main.rs:18:}}
```
