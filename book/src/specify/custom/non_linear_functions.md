# Non-linear functions

To illustrate how to implement a custom problem struct, we will take the familar logistic equation:

\\[\frac{dy}{dt} = r y (1 - y/K),\\]

Our goal is to implement a custom struct that can evaluate the rhs function \\(f(y, p, t)\\) and the jacobian multiplied by a vector \\(f'(y, p, t, v)\\).
First we define a struct that, for this simple example, only holds the parameters of interest. For a more complex problem, this struct could hold data structures neccessary to compute the rhs.

```rust
# fn main() {
use std::rc::Rc;
type T = f64;
type V = nalgebra::DVector<T>;

struct MyProblem {
    p: Rc<V>,
}
# }
```

We use an `Rc` to hold the parameters because these parameters will need to be shared between the different functions that we will implement.

Now we will implement the base `Op` trait for our struct. This trait specifies the types of the vectors and matrices that will be used, as well as the number of states and outputs in the rhs function.

```rust
# fn main() {
# use std::rc::Rc;
use diffsol::Op;

type T = f64;
type V = nalgebra::DVector<T>;
type M = nalgebra::DMatrix<T>;

# struct MyProblem {
#     p: Rc<V>,
# }
# 
# impl MyProblem {
#     fn new(p: Rc<V>) -> Self {
#         MyProblem { p }
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
}
# }
```

Next we implement the `NonLinearOp` and `NonLinearOpJacobian` trait for our struct. This trait specifies the functions that will be used to evaluate the rhs function and the jacobian multiplied by a vector.

```rust
# fn main() {
# use std::rc::Rc;
use diffsol::{
  NonLinearOp, NonLinearOpJacobian, OdeSolverEquations, OdeSolverProblem, 
  Op, UnitCallable, ConstantClosure
};

# type T = f64;
# type V = nalgebra::DVector<T>;
# type M = nalgebra::DMatrix<T>;
#
# struct MyProblem {
#     p: Rc<V>,
# }
# 
# impl MyProblem {
#     fn new(p: Rc<V>) -> Self {
#         MyProblem { p }
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
# }
# 
impl NonLinearOp for MyProblem {
    fn call_inplace(&self, x: &V, _t: T, y: &mut V) {
        y[0] = self.p[0] * x[0] * (1.0 - x[0] / self.p[1]);
    }
}
impl NonLinearOpJacobian for MyProblem {
    fn jac_mul_inplace(&self, x: &V, _t: T, v: &V, y: &mut V) {
        y[0] = self.p[0] * v[0] * (1.0 - 2.0 * x[0] / self.p[1]);
    }
}
# }
```

There we go, all done! This demonstrates how to implement a custom struct to specify a rhs function. But this is a fair bit of boilerplate code, do we really need to do all this for **every** function we want to implement?

Thankfully, the answer is no. If we didn't need to use our own struct for this particular function, we can alternativly use
the [`Closure`](https://docs.rs/diffsol/latest/diffsol/op/closure/struct.Closure.html) struct to implement the `NonLinearOp` trait for us.

```rust
# fn main() {
# use std::rc::Rc;
# type T = f64;
# type V = nalgebra::DVector<T>;
# type M = nalgebra::DMatrix<T>;
#
use diffsol::Closure;

let rhs_fn = |x: &V, p: &V, _t: T, y: &mut V| {
    y[0] = p[0] * x[0] * (1.0 - x[0] / p[1]);
};
let jac_fn = |x: &V, p: &V, _t: T, v: &V, y: &mut V| {
    y[0] = p[0] * v[0] * (1.0 - 2.0 * x[0] / p[1]);
};
let p = Rc::new(V::from_vec(vec![1.0, 10.0]));
let rhs = Rc::new(Closure::<M, _, _>::new(rhs_fn, jac_fn, 1, 1, p.clone()));
# }
```

