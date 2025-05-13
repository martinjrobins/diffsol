# Constant functions

Now we've implemented the rhs function, but how about the initial condition? We can implement the `ConstantOp` trait to specify the initial condition. Since this is quite similar to the `NonLinearOp` trait, we will do it all in one go.

```rust
# fn main() {
use diffsol::{Op, ConstantOp};

# type T = f64;
# type V = nalgebra::DVector<T>;
# type M = nalgebra::DMatrix<T>;
#
struct MyInit {}

impl Op for MyInit {
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

impl ConstantOp for MyInit {
    fn call_inplace(&self, _t: T, y: &mut V) {
        y[0] = 0.1;
    }
}
# }
```