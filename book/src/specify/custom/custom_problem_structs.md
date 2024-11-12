# Custom Problem Structs

While the [`OdeBuilder`](https://docs.rs/diffsol/latest/diffsol/ode_solver/builder/struct.OdeBuilder.html) struct is a convenient way to specify the problem, it may not be suitable in all cases. 
Often users will want to provide their own struct that can hold custom data structures and methods for evaluating the right-hand side of the ODE, the jacobian, and other functions.

## Traits

To use a custom struct to specify a problem, the primary goal is to implement the [`OdeEquations`](https://docs.rs/diffsol/latest/diffsol/ode_solver/equations/trait.OdeEquations.html) trait.
This trait has a number of associated traits that need to be implemented in order to specify each function, depending on if they are:
- Non-linear functions. In this case the [`NonLinearOp`](https://docs.rs/diffsol/latest/diffsol/op/trait.NonLinearOp.html) trait needs to be implemented.
- Linear functions. In this case the [`LinearOp`](https://docs.rs/diffsol/latest/diffsol/op/trait.LinearOp.html) trait needs to be implemented.
- Constant functions. In this case the [`ConstantOp`](https://docs.rs/diffsol/latest/diffsol/op/trait.ConstantOp.html) trait needs to be implemented.

Additionally, each function needs to implement the base operation trait [`Op`](https://docs.rs/diffsol/latest/diffsol/op/trait.Op.html).

## OdeSolverEquations struct

The [`OdeSolverEquations`](https://docs.rs/diffsol/latest/diffsol/ode_solver/equations/struct.OdeSolverEquations.html) struct is a convenience struct that already implements the `OdeEquations` trait, and can be used as a base struct for custom problem structs.
It is not neccessary to use this struct, but it can be useful to reduce boilerplate code. The example below will use this struct, but if it does not fit your use case, you can implement the `OdeEquations` trait directly.
