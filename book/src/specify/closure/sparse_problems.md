# Sparse problems

Lets consider a large system of equations that have a jacobian matrix that is sparse. For simplicity we will start with the logistic equation from the ["Specifying the Problem"](./specifying_the_problem.md) section,
but we will duplicate this equation 10 times to create a system of 10 equations. This system will have a jacobian matrix that is a diagonal matrix with 10 diagonal elements, and all other elements are zero.

Since this system is sparse, we choose a sparse matrix type to represent the jacobian matrix. We will use the `diffsol::FaerSparseMat<T>` type, which is a thin wrapper around `faer::sparse::FaerSparseMat<T>`, a sparse compressed sparse column matrix type.

```rust,ignore
{{#include ../../../../examples/intro-logistic-closures/src/problem_sparse.rs}}
```

Note that we have not specified the jacobian itself, but instead we have specified the jacobian multiplied by a vector function \\(f'(y, p, t, v)\\). 
Diffsol will use this function to generate a jacobian matrix, and since we have specified a sparse matrix type, Diffsol will attempt to 
guess the sparsity pattern of the jacobian matrix and use this to efficiently generate the jacobian matrix.

To illustrate this, we can calculate the jacobian matrix from the `rhs` function contained in the `problem` object:

```rust,ignore
{{#include ../../../../examples/intro-logistic-closures/src/print_jacobian.rs}}
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

Diffsol attempts to guess the sparsity pattern of your jacobian matrix by calling the \\(f'(y, p, t, v)\\) function repeatedly with different one-hot vectors \\(v\\) 
with a `NaN` value at each hot index. The output of this function (i.e. which elements are `0` and which are `NaN`) is then used to determine the sparsity pattern of the jacobian matrix.
Due to the fact that for IEEE 754 floating point numbers, `NaN` is propagated through most operations, this method is able to detect which output elements are dependent on which input elements.

However, this method is not foolproof, and it may fail to detect the correct sparsity pattern in some cases, particularly if values of `v` are used in control-flow statements. 
If Diffsol does not detect the correct sparsity pattern, you can manually specify the jacobian. To do this, you need to use a custom struct that implements the `OdeEquations` trait,
This is described in more detail in the ["OdeEquations trait"](../trait/ode_equations_trait.md) section.