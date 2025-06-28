# DiffSL

Many ODE libraries allow you to specify your ODE system using the host language, for example an ODE library written in C might allow you to write a C function for your RHS equations. However, this has limitations if you want to wrap this library in a higher level language like Python or R, how then do you provide the RHS functions?

For this usecase we have designed a Domain Specific Language (DSL) called DiffSL that can be used to specify the problem. DiffSL is not a general purpose language but is tightly constrained to
the specification of a system of ordinary differential equations. It features a relatively simple syntax that consists of writing a series of tensors (dense or sparse) that represent the equations of the system.

## DiffSL syntax

For more detail on the syntax of DiffSL see the [DiffSL book](https://martinjrobins.github.io/diffsl/). This section will focus on how to use DiffSL to specify a problem in Diffsol.

## Creating a DiffSL problem

To create a DiffSL problem you simply need to use the `build_from_eqn` method on the [`OdeBuilder`](https://docs.rs/diffsol/latest/diffsol/ode_solver/builder/struct.OdeBuilder.html) struct, passing in a `str` containing the DiffSL code. The DiffSL code is then parsed and compiled into native machine code using either the LLVM or Cranelift backends. The `CG` type parameter specifies the backend that you want to use to compile the DiffSL code. The `CraneliftJitModule` backend is behind the `diffsl-cranelift` feature flag. The faster [`LlvmModule`](https://docs.rs/diffsol/latest/diffsol/struct.LlvmModule.html) backend is behind one of the `diffsl-llvm*` feature flags (currently `diffsl-llvm15`, `diffsl-llvm16`, `diffsl-llvm17` or `diffsl-llvm18`), depending on the version of LLVM you have installed.

For example, here is an example of specifying a simple logistic equation using DiffSL:

```rust,ignore
{{#include ../../../../examples/intro-logistic-diffsl/src/main.rs}}
```
