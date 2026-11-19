# Introduction

diffsol is a Rust library that solves ordinary differential equations (ODEs) and semi-explicit differential algebraic equations (DAEs) of the form

\\[
M \frac{dy}{dt} = f(t, y, p), \quad y(t_0) = y_0(t_0, p),
\\]

where \\(M\\) is an optional mass matrix (it can be singular), \\(y\\) is the state vector, \\(t\\) is the time and \\(p\\) is a vector of parameters.

You can give the equations to diffsol in one of these forms:

- Rust closures, which you give to the [`OdeBuilder`](https://docs.rs/diffsol/latest/diffsol/ode_solver/builder/struct.OdeBuilder.html).
- The [DiffSL](https://martinjrobins.github.io/diffsl/) domain-specific language. diffsol compiles DiffSL equations at runtime, and it calculates the Jacobians with automatic differentiation.
- Your own types that implement the diffsol traits.

The [README](https://github.com/martinjrobins/diffsol#readme) lists the features of diffsol, and tells you how to install it.

## How to use this book

- [Modelling with diffsol](./primer/modelling_with_diffsol.md) is a primer on ODE modelling. Each section introduces a type of model (first-order and higher-order ODEs, events, DAEs, PDEs and sensitivity analysis), and gives examples that you can run.
- [Specifying ODE problems](./specify/specifying_the_problem.md) shows the three ways to give your equations to diffsol: the builder with closures, DiffSL, and the diffsol traits.
- [Creating a solver](./solver/creating_a_solver.md) and [Solving the problem](./solve/solving_the_problem.md) show how to make a solver and how to get the solution, the sensitivities and the gradients.
- [Performance](./performance/performance.md) gives advice about how to make your solves fast.
- [Using diffsol from other languages](./use/use_from_other_languages.md) shows how to use diffsol from Python, C and WebAssembly.
- [Benchmarks](./benchmarks/benchmarks.md) compares diffsol with other ODE solvers.

The [API documentation on docs.rs](https://docs.rs/diffsol) gives the full API.

The code for all the examples is in the [`examples`](https://github.com/martinjrobins/diffsol/tree/main/examples) folder of the diffsol repository.
