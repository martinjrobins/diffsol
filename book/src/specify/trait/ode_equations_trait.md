# OdeEquations trait

While the [`OdeBuilder`](https://docs.rs/diffsol/latest/diffsol/ode_solver/builder/struct.OdeBuilder.html) struct is a convenient way to specify the problem, it may not be suitable in all cases. 
Often users will want to provide their own structs that can hold custom data structures and methods for evaluating the right-hand side of the ODE, the jacobian, and other functions.

## Traits

To create your own structs for the ode system, the final goal is to implement the [`OdeEquations`](https://docs.rs/diffsol/latest/diffsol/ode_solver/equations/trait.OdeEquations.html) trait.
When you have done this, you can use the `build_from_eqn` method on the `OdeBuilder` struct to create the problem.

For each function in your system of equations, you will need to implement the appropriate trait for each function.
- Non-linear functions (rhs, out, root). In this case the [`NonLinearOp`](https://docs.rs/diffsol/latest/diffsol/op/trait.NonLinearOp.html) trait needs to be implemented.
- Linear functions (mass). In this case the [`LinearOp`](https://docs.rs/diffsol/latest/diffsol/op/trait.LinearOp.html) trait needs to be implemented.
- Constant functions (init). In this case the [`ConstantOp`](https://docs.rs/diffsol/latest/diffsol/op/trait.ConstantOp.html) trait needs to be implemented.

Additionally, each function needs to implement the base operation trait [`Op`](https://docs.rs/diffsol/latest/diffsol/op/trait.Op.html).

Once you have implemented the appropriate traits for your custom struct, you can use the [`OdeBuilder`](https://docs.rs/diffsol/latest/diffsol/ode_solver/builder/struct.OdeBuilder.html) struct to specify the problem.


