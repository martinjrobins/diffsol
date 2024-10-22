# Sparse problems

Lets consider a large system of equations that have a jacobian matrix that is sparse. For simplicity we will start with the logistic equation from the ["Specifying the Problem"](./specifying_the_problem.md) section,
but we will duplicate this equation 10 times to create a system of 10 equations. This system will have a jacobian matrix that is a diagonal matrix with 10 diagonal elements, and all other elements are zero.

Since this system is sparse, we choose a sparse matrix type to represent the jacobian matrix. We will use the `diffsol::SparseColMat<T>` type, which is a thin wrapper around `faer::sparse::SparseColMat<T>`, a sparse compressed sparse column matrix type.

```rust
# fn main() {
use diffsol::OdeBuilder;
type M = diffsol::SparseColMat<f64>;
type V = faer::Col<f64>;

let problem = OdeBuilder::new()
    .t0(0.0)
    .rtol(1e-6)
    .atol([1e-6])
    .p(vec![1.0, 10.0])
    .build_ode::<M, _, _, _>(
       |x, p, _t, y| {
         for i in 0..10 {
           y[i] = p[0] * x[i] * (1.0 - x[i] / p[1]);
         }
       },
       |x, p, _t, v , y| {
         for i in 0..10 {
           y[i] = p[0] * v[i] * (1.0 - 2.0 * x[i] / p[1]);
         }
       },
       |_p, _t| V::from_fn(10, |_| 0.1),
    ).unwrap();
# }
```

Note that we have not specified the jacobian itself, but instead we have specified the jacobian multiplied by a vector function \\(f'(y, p, t, v)\\). 
DiffSol will use this function to generate a jacobian matrix, and since we have specified a sparse matrix type, DiffSol will attempt to 
guess the sparsity pattern of the jacobian matrix and use this to efficiently generate the jacobian matrix.

To illustrate this, we can calculate the jacobian matrix from the `rhs` function contained in the `problem` object:

```rust
# use diffsol::OdeBuilder;
use diffsol::{OdeEquations, NonLinearOp, NonLinearOpJacobian, Matrix, ConstantOp};

# type M = diffsol::SparseColMat<f64>;
# type V = faer::Col<f64>;
#
# fn main() {
#  let problem = OdeBuilder::new()
#    .t0(0.0)
#    .rtol(1e-6)
#    .atol([1e-6])
#    .p(vec![1.0, 10.0])
#    .build_ode::<M, _, _, _>(
#    |x, p, _t, y| {
#      for i in 0..10 {
#        y[i] = p[0] * x[i] * (1.0 - x[i] / p[1]);
#      }
#    },
#    |x, p, _t, v , y| {
#      for i in 0..10 {
#        y[i] = p[0] * v[i] * (1.0 - 2.0 * x[i] / p[1]);
#      }
#    },
#    |_p, _t| V::from_fn(10, |_| 0.1),
#    ).unwrap();
let t0 = problem.t0;
let y0 = problem.eqn.init().call(t0);
let jacobian = problem.eqn.rhs().jacobian(&y0, t0);
for (i, j, v) in jacobian.triplet_iter() {
    println!("({}, {}) = {}", i, j, v);
}
# }
```

which will print the jacobian matrix in triplet format:

```
(0, 0) = 0.98
(1, 1) = 0.98
(2, 2) = 0.98
(3, 3) = 0.98
(4, 4) = 0.98
(5, 5) = 0.98
(6, 6) = 0.98
(7, 7) = 0.98
(8, 8) = 0.98
(9, 9) = 0.98
```

DiffSol attempts to guess the sparsity pattern of your jacobian matrix by calling the \\(f'(y, p, t, v)\\) function repeatedly with different one-hot vectors \\(v\\) 
with a `NaN` value at each index. The output of this function (i.e. which elements are `0` and which are `NaN`) is then used to determine the sparsity pattern of the jacobian matrix.
Due to the fact that for IEEE 754 floating point numbers, `NaN` is propagated through most operations, this method is able to detect which output elements are dependent on which input elements.

However, this method is not foolproof, and it may fail to detect the correct sparsity pattern in some cases, particularly if values of `v` are used in control-flow statements. 
If DiffSol does not detect the correct sparsity pattern, you can manually specify the jacobian. To do this, you need
to implement the [`diffsol::NonLinearOp`](https://docs.rs/diffsol/latest/diffsol/op/trait.NonLinearOp.html) trait for the rhs function. 
This is described in more detail in the ["Custom Problem Structs"](./custom_problem_structs.md) section, but is illustrated below. 

```rust
# fn main() {
use std::rc::Rc;
use faer::sparse::{SparseColMat, SymbolicSparseColMatRef};
use diffsol::{NonLinearOp, NonLinearOpJacobian, OdeSolverEquations, OdeSolverProblem, Op, UnitCallable, ConstantClosure, OdeBuilder};

type T = f64;
type V = faer::Col<T>;
type M = diffsol::SparseColMat<T>;

struct MyProblem {
  jacobian: SparseColMat<usize, T>,
  p: Rc<V>,
}

impl MyProblem {
  fn new(p: Rc<V>) -> Self {
    let mut triplets = Vec::new();
    for i in 0..10 {
      triplets.push((i, i, 1.0));
    }
    let jacobian = SparseColMat::try_new_from_triplets(10, 10, triplets.as_slice()).unwrap();
    MyProblem { p, jacobian }
  }
}

impl Op for MyProblem {
  type T = T;
  type V = V;
  type M = M;
  fn nstates(&self) -> usize {
    10
  }
  fn nout(&self) -> usize {
    10
  }
  fn sparsity(&self) -> Option<SymbolicSparseColMatRef<usize>> {
    Some(self.jacobian.symbolic())
  }
}
  
impl NonLinearOp for MyProblem {
  fn call_inplace(&self, x: &V, _t: T, y: &mut V) {
    for i in 0..10 {
      y[i] = self.p[0] * x[i] * (1.0 - x[i] / self.p[1]);
    }
  }
 
}
impl NonLinearOpJacobian for MyProblem {
  fn jac_mul_inplace(&self, x: &V, _t: T, v: &V, y: &mut V) {
    for i in 0..10 {
      y[i] = self.p[0] * v[i] * (1.0 - 2.0 * x[i] / self.p[1]);
    }
  }
  fn jacobian_inplace(&self, x: &Self::V, _t: Self::T, y: &mut Self::M) {
    for i in 0..10 {
      let row = y.faer().row_indices()[i];
      y.faer_mut().values_mut()[i] = self.p[0] * (1.0 - 2.0 * x[row] / self.p[1]);
    }
  }

}

let p = [1.0, 10.0];
let p = Rc::new(V::from_fn(p.len(), |i| p[i]));
let rhs = Rc::new(MyProblem::new(p.clone()));

// use the provided constant closure to define the initial condition
let init_fn = |_p: &V, _t: T| V::from_fn(10, |_| 0.1);
let init = Rc::new(ConstantClosure::new(init_fn, p.clone()));

// we don't have a mass matrix, root or output functions, so we can set to None
// we still need to give a placeholder type for these, so we use the diffsol::UnitCallable type
let mass: Option<Rc<UnitCallable<M>>> = None;
let root: Option<Rc<UnitCallable<M>>> = None;
let out: Option<Rc<UnitCallable<M>>> = None;

let p = Rc::new(V::zeros(0));
let eqn = OdeSolverEquations::new(rhs, mass, root, init, out, p.clone());
let _problem = OdeBuilder::new().build_from_eqn(eqn).unwrap();
# }
```

