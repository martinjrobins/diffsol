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
}

impl ConstantOp for MyInit {
    fn call_inplace(&self, _t: T, y: &mut V) {
        y[0] = 0.1;
    }
}
# }
```

Again, we can use the [`ConstantClosure`](https://docs.rs/diffsol/latest/diffsol/op/constant_closure/struct.ConstantClosure.html) struct to implement the `ConstantOp` trait for us if it's not neccessary to use our own struct.

```rust
# fn main() {
# use std::rc::Rc;
use diffsol::ConstantClosure;

# type T = f64;
# type V = nalgebra::DVector<T>;
# type M = nalgebra::DMatrix<T>;
#
let p = Rc::new(V::from_vec(vec![1.0, 10.0]));
let init_fn = |_p: &V, _t: T| V::from_element(1, 0.1);
let init = Rc::new(ConstantClosure::<M, _>::new(init_fn, p.clone()));
# }
```
