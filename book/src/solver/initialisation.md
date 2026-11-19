# Initialisation

Each solver has an internal state that holds information like the current state vector, the gradient of the state vector, the current time, and the current step size. When you create a solver using the `bdf`, `tr_bdf2` or `esdirk34` methods on the `OdeSolverProblem` struct, the solver will be initialised with an initial state based on the initial conditions of the problem as well as satisfying any algebraic constraints. The `rk_state_and_consistent` and `bdf_state` methods also create a consistent state. The `rk_state` method does not solve for the algebraic constraints, so the state is not consistent if the problem has a mass matrix. The `tsit45` method also does not make the state consistent, but this is not necessary because the explicit Runge-Kutta solver does not support a mass matrix. An initial time step will also be chosen based on your provided equations.

Each solver's state struct implements the [`OdeSolverState`](https://docs.rs/diffsol/latest/diffsol/ode_solver/state/trait.OdeSolverState.html) trait, and if you wish to manually create and setup a state, you can use the methods on this trait to do so.

For example, say that you wish to bypass the initialisation of the state as you already have the algebraic constraints and so don't need to solve for them. You can use the `new_without_initialise` method on the `OdeSolverState` trait to create a new state without initialising it. You can then use the `as_mut` method to get a mutable reference to the state and set the values manually.

```rust,ignore
{{#include ../../../examples/intro-logistic-closures/src/create_solvers_uninit.rs}}
```

Note that each state struct has a [`as_ref`](https://docs.rs/diffsol/latest/diffsol/ode_solver/state/trait.OdeSolverState.html#tymethod.as_ref) and [`as_mut`](https://docs.rs/diffsol/latest/diffsol/ode_solver/state/trait.OdeSolverState.html#tymethod.as_mut) methods that return a [`StateRef`](https://docs.rs/diffsol/latest/diffsol/ode_solver/state/struct.StateRef.html) or [`StateRefMut`](https://docs.rs/diffsol/latest/diffsol/ode_solver/state/struct.StateRefMut.html) struct respectively. These structs provide a solver-independent way to access the state values so you can use the same code with different solvers.

