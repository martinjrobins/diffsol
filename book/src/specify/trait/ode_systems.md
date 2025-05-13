# ODE systems

So far we've focused on using custom structs to specify individual equations, now we need to put these together to specify the entire system of equations. 

## Implementing the OdeEquations trait

To specify the entire system of equations, you need to implement the `Op`, [`OdeEquations`](https://docs.rs/diffsol/latest/diffsol/ode_solver/equations/trait.OdeEquations.html) 
and [`OdeEquationsRef`](https://docs.rs/diffsol/latest/diffsol/ode_solver/equations/trait.OdeEquationsRef.html) traits for your struct.

## Getting all your traits in order

The `OdeEquations` trait requires methods that return objects corresponding to the right-hand side function, mass matrix, root function, initial condition, and output functions.
Therefore, you need to already have structs that implement the `NonLinearOp`, `LinearOp`, and `ConstantOp` traits for these functions. For the purposes of this example, we will assume that
you have already implemented these traits for your structs. 

Often, the structs that implement these traits will have to use data defined in the struct that implements the `OdeEquations` trait. For example, they might wish to have a reference to the same parameter vector `p`. Therefore, you will often need to define lifetimes for these structs to ensure that they can access the data they need. 

Note that these struct will need to be lightweight and should not contain a significant amount of data. The data should be stored in the struct that implements the `OdeEquations` trait. This is because these structs will be created and destroyed many times during the course of the simulation (e.g. every time the right-hand side function is called).

## My custom ode equations

We'll define a central struct `MyEquations` that will hold the data for the entire system of equations. In this example, `MyEquations` will own a parameter vector `p` that is used by all the equations.

```rust,ignore
{{#include ../../../../examples/custom-ode-equations/src/my_equations.rs}}
```

As with individual functions, we also need to implement the `Op` trait for this struct. 

```rust,ignore
{{#include ../../../../examples/custom-ode-equations/src/my_equations_impl_op.rs}}
```

## Implementing the OdeEquations traits

`OdeEquations` and `OdeEquationsRef` are a pair of traits that together define the system of equations using the individual function structs that we defined earlier. The `OdeEquationsRef` trait is designed to allow individual function structs to be able to reference the data in the main `OdeEquations` struct, allowing them to share data.


```rust,ignore
{{#include ../../../../examples/custom-ode-equations/src/my_equations_impl_ode_equations.rs}}
```

## Creating the problem

Now that we have our custom `OdeEquations` struct, we can use it in an `OdeBuilder` to create the problem.

```rust,ignore
{{#include ../../../../examples/custom-ode-equations/src/build.rs}}
```