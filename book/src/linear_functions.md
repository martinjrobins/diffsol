# Linear functions

Naturally, we can also implement the `LinearOp` trait if we want to include a mass matrix in our model. A common use case for implementing this trait is to store the mass matrix in a custom struct, like so:

```rust
# fn main() {
use diffsol::{Op, LinearOp};

# type T = f64;
# type V = nalgebra::DVector<T>;
# type M = nalgebra::DMatrix<T>;
#
struct MyMass {
  mass: M,
}

impl MyMass {
  fn new() -> Self {
      let mass = M::from_element(1, 1, 1.0);
      Self { mass }
  }
}

impl Op for MyMass {
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

impl LinearOp for MyMass {
  fn gemv_inplace(&self, x: &V, _t: T, beta: T, y: &mut V) {
      y.gemv(1.0, &self.mass, x, beta)
  }
}
# }
```

Alternatively, we can use the [`LinearClosure`](https://docs.rs/diffsol/latest/diffsol/op/linear_closure/struct.LinearClosure.html) struct to implement the `LinearOp` trait for us.

```rust
# fn main() {
# use std::rc::Rc;
use diffsol::LinearClosure;

# type T = f64;
# type V = nalgebra::DVector<T>;
# type M = nalgebra::DMatrix<T>;
#
# let p = Rc::new(V::from_vec(vec![1.0, 10.0]));
let mass_fn = |v: &V, _p: &V, _t: T, beta: T, y: &mut V| {
    y[0] = v[0] + beta * y[0];
};
let mass = Rc::new(LinearClosure::<M, _>::new(mass_fn, 1, 1, p.clone()));
# }
```

