# Sparse problems

Lets consider a large system of equations that have a jacobian matrix that is sparse. For simplicity we will start with the logistic equation from the ["Specifying the Problem"](./specifying_the_problem.md) section,
but we will duplicate this equation 10 times to create a system of 10 equations. This system will have a jacobian matrix that is a diagonal matrix with 10 diagonal elements, and all other elements are zero.

Since this system is sparse, we choose a sparse matrix type to represent the jacobian matrix. We will use the `diffsol::SparseColMat<T>` type, which is a thin wrapper around `faer::sparse::SparseColMat<T>`, a sparse compressed sparse column matrix type.

```rust
# fn main() {
use diffsol::OdeBuilder;
type M = diffsol::SparseColMat<f64>;
type V = faer::Col<f64>;

let problem = OdeBuilder::<M>::new()
    .t0(0.0)
    .rtol(1e-6)
    .atol([1e-6])
    .p(vec![1.0, 10.0])
    .rhs_implicit(
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
    )
    .init(
        |_p, _t| V::from_fn(10, |_| 0.1),
    )
    .build()
    .unwrap();
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
# let problem = OdeBuilder::<M>::new()
#     .t0(0.0)
#     .rtol(1e-6)
#     .atol([1e-6])
#     .p(vec![1.0, 10.0])
#     .rhs_implicit(
#         |x, p, _t, y| {
#             for i in 0..10 {
#                 y[i] = p[0] * x[i] * (1.0 - x[i] / p[1]);
#             }
#         },
#         |x, p, _t, v , y| {
#             for i in 0..10 {
#                 y[i] = p[0] * v[i] * (1.0 - 2.0 * x[i] / p[1]);
#             }
#         },
#     )
#     .init(
#         |_p, _t| V::from_fn(10, |_| 0.1),
#     )
#     .build()
#     .unwrap();
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
with a `NaN` value at each hot index. The output of this function (i.e. which elements are `0` and which are `NaN`) is then used to determine the sparsity pattern of the jacobian matrix.
Due to the fact that for IEEE 754 floating point numbers, `NaN` is propagated through most operations, this method is able to detect which output elements are dependent on which input elements.

However, this method is not foolproof, and it may fail to detect the correct sparsity pattern in some cases, particularly if values of `v` are used in control-flow statements. 
If DiffSol does not detect the correct sparsity pattern, you can manually specify the jacobian. To do this, you need to use a custom struct that implements the `OdeEquations` trait,
This is described in more detail in the ["Custom Problem Structs"](./custom_problem_structs.md) section.