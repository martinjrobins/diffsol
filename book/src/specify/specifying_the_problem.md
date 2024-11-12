# DiffSol APIs for specifying problems

Most of the DiffSol user-facing API revolves around specifying the problem you want to solve, thus a large part of this book will be dedicated to explaining how to specify a problem. All the examples presented in [the primer](../primer/modelling_with_odes.md) used the DiffSL DSL to specify the problem, but DiffSol also provides a pure Rust API for specifying problems. This API is sometimes more verbose than the DSL, but is more flexible, more performant, and easier to use if you have a model already written in Rust or another language that you can easily port or call from Rust.

## ODE equations

The class of ODE equations that DiffSol can solve are of the form

\\[M(t) \frac{dy}{dt} = f(t, y, p),\\]
\\[y(t_0) = y_0(t_0, p),\\]
\\[z(t) = g(t, y, p),\\]

where:
- \\(f(t, y, p)\\) is the right-hand side of the ODE, 
- \\(y\\) is the state vector, 
- \\(p\\) are the parameters, 
- \\(t\\) is the time.
- \\(y_0(t_0, p)\\) is the initial state vector at time \\(t_0\\). 
- \\(M(t)\\) is the mass matrix (this is optional, and is implicitly the identity matrix if not specified),
- \\(g(t, y, p)\\) is an output function that can be used to calculate additional outputs from the state vector (this is optional, and is implicitly \\(g(t, y, p) = y\\) if not specified).

The user can also optionally specify a root function \\(r(t, y, p)\\) that can be used to find the time at which a root occurs.

## DiffSol problem APIs

DiffSol has three main APIs for specifying problems:
- The [`OdeBuilder`](https://docs.rs/diffsol/latest/diffsol/ode_solver/builder/struct.OdeBuilder.html) struct, where the user can specify the functions above using Rust closures.
  This is the easiest API to use from Rust, and is the recommended API for most users.
- The [`OdeEquations`](https://docs.rs/diffsol/latest/diffsol/ode_solver/equations/trait.OdeEquations.html) trait 
  where the user can implement the functions above on their own structs.
  This API is more flexible than the `OdeBuilder` API, but is more complex to use. It is useful if you have custom data structures and code that you want to use to evaluate
  your functions that does not fit within the `OdeBuilder` API.
- The [`DiffSlContext`](https://docs.rs/diffsol/latest/diffsol/ode_solver/diffsl/struct.DiffSlContext.html) struct, where the user can specify the functions above using the [DiffSL](https://martinjrobins.github.io/diffsl/)
  Domain Specific Language (DSL). This API is behind a feature flag (`diffsl` if you want to use the slower cranelift backend, `diffsl-llvm*` if you want to use the faster LLVM backend), but has the best API if you want to use DiffSL from a higher-level language like Python or R while still having similar performance.



