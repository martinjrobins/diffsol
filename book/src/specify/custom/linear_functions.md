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
  fn nparams(&self) -> usize {
      0
  }
}

impl LinearOp for MyMass {
  fn gemv_inplace(&self, x: &V, _t: T, beta: T, y: &mut V) {
      y.gemv(1.0, &self.mass, x, beta)
  }
}
# }
```


