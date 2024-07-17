# Sparse problems

Lets consider a large system of equations that have a jacobian matrix that is sparse. For simplicity we will start with the logistic equation from the ["Specifying the Problem"](./specifying_the_problem.md) section,
but we will duplicate this equation 10 times to create a system of 10 equations. This system will have a jacobian matrix that is a diagonal matrix with 10 diagonal elements, and all other elements are zero.

Since this system is sparse, we choose a sparse matrix type to represent the jacobian matrix. We will use the `diffsol::SparseColMat<T>` type, which is a thin wrapper around `faer::sparse::SparseColMat<T>`, a sparse compressed sparse column matrix type.

```rust
# fn main() {
use diffsol::OdeBuilder;
use nalgebra::DVector;
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
       |_p, _t| V::from_element(10, 0.1),
    ).unwrap();
# }
```

Note that we have not specified the jacobian itself, but instead we have specified the jacobian multiplied by a vector function \\(f'(y, p, t, v)\\). 
DiffSol will use this function to generate a jacobian matrix, and since we have specified a sparse matrix type, DiffSol will attempt to 
guess the sparsity pattern of the jacobian matrix and use this to efficiently generate the jacobian matrix.

To illustrate this, we can calculate the jacobian matrix from the `rhs` function contained in the `problem` object:

```rust
# use crate::OdeBuilder;
# type M = crate::SparseColMat<f64>;
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
#    |_p, _t| V::from_element(100, 0.1)
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
If DiffSol does not detect the correct sparsity pattern, you can manually specify the sparsity pattern. To do this, you need
to implement the [`diffsol::NonLinearOp`](https://docs.rs/diffsol/latest/diffsol/op/trait.NonLinearOp.html) trait for the rhs function. 
This is described in more detail in the ["Custom Problem Structs"](./custom_problem_structs.md) section, but is illustrated below. 
Note that here we use DiffSol's [`JacobianColoring`](https://docs.rs/diffsol/latest/diffsol/jacobian_coloring/struct.JacobianColoring.html) struct 
to calculate the jacobian matrix. Alternativly, we could have simply stored a `SparseColMat` in the `MyProblem` struct and calculated the jacobian matrix directly.

```rust
#fn main() {
use std::rc::Rc;
use faer::sparse::{SymbolicSparseColMat, SymbolicSparseColMatRef};
use crate::{NonLinearOp, OdeSolverEquations, OdeSolverProblem, Op, UnitCallable, ConstantClosure, JacobianColoring};

type T = f64;
type V = faer::Col<T>;
type M = crate::SparseColMat<T>;

struct MyProblem {
  sparsity: SymbolicSparseColMat<usize>,
  coloring: Option<JacobianColoring<M>>,
  p: V,
}
impl MyProblem {
  fn new(p: V) -> Self {
    let col_ptrs = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let row_indices = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    let sparsity = SymbolicSparseColMat::new_checked(10, 10, col_ptrs, None, row_indices);
    let non_zeros = vec![(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)];
    let mut ret = MyProblem { sparsity, p, coloring: None };
    let coloring = JacobianColoring::new_from_non_zeros(&ret, non_zeros);
    ret.coloring = Some(coloring);
    ret
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
  Some(self.sparsity.as_ref())
 }
}
  
impl NonLinearOp for MyProblem {
  fn call_inplace(&self, x: &V, _t: T, y: &mut V) {
    for i in 0..10 {
      y[i] = self.p[0] * x[i] * (1.0 - x[i] / self.p[1]);
    }
  }
 fn jac_mul_inplace(&self, x: &V, _t: T, v: &V, y: &mut V) {
    for i in 0..10 {
      y[i] = self.p[0] * v[i] * (1.0 - 2.0 * x[i] / self.p[1]);
    }
  }
  fn jacobian_inplace(&self, x: &V, t: T, y: &mut M) {
     self.coloring.as_ref().unwrap().jacobian_inplace(self, x, t, y);
  } 
}

let rhs = Rc::new(MyProblem::new(V::from_vec(vec![1.0, 10.0])));

// use the provided constant closure to define the initial condition
let init_fn = |_p: &V, _t: T| V::from_element(10, 0.1);
let init = Rc::new(ConstantClosure::new(init_fn, Rc::new(V::from_vec(vec![]))));

// we don't have a mass matrix, root or output functions, so we can set to None
// we still need to give a placeholder type for these, so we use the diffsol::UnitCallable type
let mass: Option<Rc<UnitCallable<M>>> = None;
let root: Option<Rc<UnitCallable<M>>> = None;
let out: Option<Rc<UnitCallable<M>>> = None;

let p = Rc::new(V::from_vec(vec![]));
let eqn = OdeSolverEquations::new(rhs, mass, root, init, out, p);
let rtol = 1e-6;
let atol = V::from_element(10, 1e-6);
let t0 = 0.0;
let h0 = 1.0;
let _problem = OdeSolverProblem::new(eqn, rtol, atol, t0, h0, false, false).unwrap();
# }
```

