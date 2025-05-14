# Diffsol APIs for specifying problems

Most of the Diffsol user-facing API revolves around specifying the problem you want to solve, thus a large part of this book will be dedicated to explaining how to specify a problem. All the examples presented in [the primer](../primer/modelling_with_odes.md) used the DiffSL DSL to specify the problem, but Diffsol also provides a pure Rust API for specifying problems. This API is sometimes more verbose than the DSL, but is more flexible, more performant, and easier to use if you have a model already written in Rust or another language that you can easily port or call from Rust.

## ODE equations

The class of ODE equations that Diffsol can solve are of the form

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

## Diffsol problem APIs

Specifying a problem in Diffsol is done via the [`OdeBuilder`](https://docs.rs/diffsol/latest/diffsol/ode_solver/builder/struct.OdeBuilder.html) struct, using the `OdeBuilder::new` method to create a new builder, and then chaining methods to set the equations to be solved, initial time, initial step size, relative & absolute tolerances, and parameters, or leaving them at their default values. Then, call a `build` method to create a new problem.

Users can specify the equations to be solved via three main APIs, ranging from the simplest to the most complex (but also the most flexible):

- The [`DiffSl`](https://docs.rs/diffsol/latest/diffsol/ode_solver/diffsl/struct.DiffSl.html) struct allows users to specify the equations above using the [DiffSL](https://martinjrobins.github.io/diffsl/)
  Domain Specific Language (DSL). This API is behind a feature flag (`diffsl-cranelift` if you want to use the slower cranelift backend, `diffsl-llvm*` if you want to use the faster LLVM backend). This is the easiest API to use as it can use automatic differentiation to calculate the neccessary gradients, and is the best API if you want to use DiffSL from a higher-level language like Python or R while still having similar performance to Rust.
- The [`OdeBuilder`](https://docs.rs/diffsol/latest/diffsol/ode_solver/builder/struct.OdeBuilder.html) struct also has methods to set the equations using rust closures (see e.g. [OdeBuilder::rhs_implicit](https://docs.rs/diffsol/latest/diffsol/ode_solver/builder/struct.OdeBuilder.html#method.rhs_implicit)). This API is convenient if you want to stick to pure rust code without using DiffSL and the JIT compiler, but requires you to calculate the gradients of the equations yourself.
- Implementing the [`OdeEquations`](https://docs.rs/diffsol/latest/diffsol/ode_solver/equations/trait.OdeEquations.html) trait allows users to implement the equations on their own structs. This API is the most flexible as it allows users to use custom data structures and code that might not fit within the `OdeBuilder` API. However, it is the most verbose API and requires users to be more familiar with the various Diffsol traits.
