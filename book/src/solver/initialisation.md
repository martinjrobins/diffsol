# Initialisation

Each solver has an internal state that holds information like the current state vector, the gradient of the state vector, the current time, and the current step size. When you create a solver using the `bdf` or `sdirk` methods on the `OdeSolverProblem` struct, the solver will be initialised with an initial state based on the initial conditions of the problem as well as satisfying any algebraic constraints. An initial time step will also be chosen based on your provided equations.

Each solver's state struct implements the [`OdeSolverState`](https://docs.rs/diffsol/latest/diffsol/ode_solver/state/trait.OdeSolverState.html) trait, and if you wish to manually create and setup a state, you can use the methods on this trait to do so.

For example, say that you wish to bypass the initialisation of the state as you already have the algebraic constraints and so don't need to solve for them. You can use the `new_without_initialise` method on the `OdeSolverState` trait to create a new state without initialising it. You can then use the `as_mut` method to get a mutable reference to the state and set the values manually.

```rust,ignore
{{#include ../../../examples/intro-logistic-closures/src/create_solvers_uninit.rs}}
```

Note that each state struct has a [`as_ref`](https://docs.rs/diffsol/latest/diffsol/ode_solver/state/trait.OdeSolverState.html#tymethod.as_ref) and [`as_mut`](https://docs.rs/diffsol/latest/diffsol/ode_solver/state/trait.OdeSolverState.html#tymethod.as_mut) methods that return a [`StateRef`](https://docs.rs/diffsol/latest/diffsol/ode_solver/state/struct.StateRef.html) or [`StateRefMut`](https://docs.rs/diffsol/latest/diffsol/ode_solver/state/struct.StateRefMut.html) struct respectively. These structs provide a solver-independent way to access the state values so you can use the same code with different solvers.

