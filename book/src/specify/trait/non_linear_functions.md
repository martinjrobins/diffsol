# Non-linear functions

To illustrate how to implement a custom problem struct, we will take the familar logistic equation:

\\[\frac{dy}{dt} = r y (1 - y/K),\\]

Our goal is to implement a custom struct that can evaluate the rhs function \\(f(y, p, t)\\) and the jacobian multiplied by a vector \\(f'(y, p, t, v)\\).
First we define an empty struct. For a more complex problem, this struct could hold data structures neccessary to compute the rhs.

```rust
# fn main() {
type T = f64;
type V = nalgebra::DVector<T>;

struct MyProblem;
# }
```

Now we will implement the base `Op` trait for our struct. The `Op` trait specifies the types of the vectors and matrices that will be used, as well as the number of states and outputs in the rhs function.

```rust
# fn main() {
use diffsol::Op;

type T = f64;
type V = nalgebra::DVector<T>;
type M = nalgebra::DMatrix<T>;

# struct MyProblem;
# 
# impl MyProblem {
#     fn new() -> Self {
#         MyProblem {}
#     }
# }
# 
impl Op for MyProblem {
    type T = T;
    type V = V;
    type M = M;
    fn nstates(&self) -> usize {
        1
    }
    fn nout(&self) -> usize {
        1
    }
    fn nparams(&self) -> usize {
        0
    }
}
# }
```


Next we implement the `NonLinearOp` and `NonLinearOpJacobian` trait for our struct. This trait specifies the functions that will be used to evaluate the rhs function and the jacobian multiplied by a vector.

```rust
# fn main() {
use diffsol::{
  NonLinearOp, NonLinearOpJacobian
};
# use diffsol::Op;

# type T = f64;
# type V = nalgebra::DVector<T>;
# type M = nalgebra::DMatrix<T>;
#
# struct MyProblem;
# 
# impl MyProblem {
#     fn new() -> Self {
#         MyProblem { }
#     }
# }
# 
# impl Op for MyProblem {
#     type T = T;
#     type V = V;
#     type M = M;
#     fn nstates(&self) -> usize {
#         1
#     }
#     fn nout(&self) -> usize {
#         1
#     }
#     fn nparams(&self) -> usize {
#         0
#     }
# }

impl<'a> NonLinearOp for MyProblem {
    fn call_inplace(&self, x: &V, _t: T, y: &mut V) {
        y[0] = x[0] * (1.0 - x[0]);
    }
}
impl<'a> NonLinearOpJacobian for MyProblem {
    fn jac_mul_inplace(&self, x: &V, _t: T, v: &V, y: &mut V) {
        y[0] = v[0] * (1.0 - 2.0 * x[0]);
    }
}
# }
```

There we go, all done! This demonstrates how to implement a custom struct to specify a rhs function.

