# Parameters

The fundamental trait that diffsol uses to specify a mathematical operation is [`Op`](https://docs.rs/diffsol/latest/diffsol/op/trait.Op.html).
This trait defines the number of parameters in your ODE model via [`Op::nparams`](https://docs.rs/diffsol/latest/diffsol/op/trait.Op.html#tymethod.nparams).

## What is a "parameter"

In short, a parameter is
*any symbol in your ODE model where you care about how the solution (or a loss function)
varies in response to a change in that parameter*.

Now, you might have any number of parameters in your model that don't fit the description above.
For example, say you were modelling the [bouncing ball example](../../primer/bouncing_ball.html)
using a custom struct, and you wanted to parameterise the gravitational constant by storing
the constant on a field called `gravity`:

```rust,ignore
struct MyBouncingBallEqn {
  pub gravity: f64
}
```

In this case you probably don't care how your solution changes with respect to `gravity`,
since you are reasonably confident that the gravitational constant will be constant
in all the universes that you care about. So in this case your `Op::nparams` will return
0.

The notion of a parameter in diffsol is closely linked to diffsol's forward and adjoint sensitivity
features. Put another way, a parameter in diffsol is any *parameter of your model that you want to
calculate a gradient with respect to*. If you don't want that parameter to be include in either
the forward or adjoint sensitivity calculation, then don't include it in `nparams`.

## Varying parameters

The rust borrow checker governs any change to your model. Any change in a parameter of your model, whether its a parameter you have reported to diffsol or otherwise, requires a mutation of your `OdeEquations` struct. Each solver that you create takes a reference to
[`OdeSolverProblem`](https://docs.rs/diffsol/latest/diffsol/ode_solver/problem/struct.OdeSolverProblem.html), and therefore to your custom `OdeEquations` struct, so you won't
be able to mutate it until you drop all the solvers with that reference. Given an instance of `OdeSolverProblem`, you can get a mutable reference to you custom `OdeEquations` struct using
[`OdeSolverProblem::eqn_mut`](https://docs.rs/diffsol/latest/diffsol/ode_solver/problem/struct.OdeSolverProblem.html#method.eqn_mut).
