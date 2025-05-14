# Non-linear functions

To illustrate how to implement a custom problem struct, we will take the familar logistic equation:

\\[\frac{dy}{dt} = r y (1 - y/K),\\]

Our goal is to implement a custom struct that can evaluate the rhs function \\(f(y, p, t)\\) and the jacobian multiplied by a vector \\(f'(y, p, t, v)\\).

To start with, lets define a few linear algebra types that we will use in our function. We need four types:
- `T` is the scalar type (e.g. `f64`)
- `V` is the vector type (e.g. `NalgebraVec<T>`)
- `M` is the matrix type (e.g. `NalgebraMat<T>`)
- `C` is the context type for the rest (e.g. `NalgebraContext`)

```rust,ignore
{{#include ../../../../examples/custom-ode-equations/src/common.rs}}
```

Next, we'll define a struct that we'll use to calculate our RHS equations \\(f(y, p, t)\\). We'll pretend that this struct has a reference to a parameter vector \\(p\\) that we'll use to calculate the rhs function. This makes sense since we'll have multiple functions that make up our systems of equations, and they will probably share some parameters. 

```rust,ignore
{{#include ../../../../examples/custom-ode-equations/src/my_rhs.rs}}
```

Now we will implement the base [`Op`](https://docs.rs/diffsol/latest/diffsol/op/trait.Op.html) trait for our struct. The `Op` trait specifies the types of the vectors and matrices that will be used, as well as the number of states and outputs in the rhs function.

```rust,ignore
{{#include ../../../../examples/custom-ode-equations/src/my_rhs_impl_op.rs}}
```

Next we implement the [`NonLinearOp`](https://docs.rs/diffsol/latest/diffsol/op/nonlinear_op/trait.NonLinearOp.html) and [`NonLinearOpJacobian`](https://docs.rs/diffsol/latest/diffsol/op/nonlinear_op/trait.NonLinearOpJacobian.html) trait for our struct. This trait specifies the functions that will be used to evaluate the rhs function and the jacobian multiplied by a vector.

```rust,ignore
{{#include ../../../../examples/custom-ode-equations/src/my_rhs_impl_nonlinear.rs}}
```

There we go, all done! This demonstrates how to implement a custom struct to specify a rhs function.

