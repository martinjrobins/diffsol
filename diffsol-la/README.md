# diffsol-la

Linear algebra foundation for [diffsol](https://github.com/martinjrobins/diffsol) —
the vector, matrix, and linear-solver abstractions used by the ODE/DAE solvers,
together with concrete backends.

Most users should depend on the [`diffsol`](https://crates.io/crates/diffsol) crate
directly. Use `diffsol-la` when you want to build on the linear algebra layer itself.

## What's in this crate

- **Vectors**: `Vector`, `VectorView`, `VectorViewMut`, `VectorIndex`, `VectorHost`.
- **Matrices**: `Matrix`, `DenseMatrix`, `MatrixView`, `MatrixViewMut`, and sparsity handling.
- **Linear operators and solvers**: the `LinearOp` trait describing a linear
  operator `A`, and the `LinearSolver` trait for solving `Ax = b`.
- **Backends**: `NalgebraLU`, `FaerLU`, `FaerSparseLU`, `KLU` (suitesparse), and `CudaLU`.
- **Support types**: `Context`, `Scalar`, `Scale`, and the `LaError` error type.

## Features

- `nalgebra`: nalgebra-backed containers and solvers (enabled by default).
- `faer`: faer-backed containers and solvers (enabled by default).
- `cuda`: in-built CUDA containers and solvers (disabled by default, experimental).
- `suitesparse`: the KLU sparse direct solver from SuiteSparse (disabled by default).

## Links

- Main crate: <https://crates.io/crates/diffsol>
- Documentation: <https://docs.rs/diffsol-la>
- Book: <https://martinjrobins.github.io/diffsol/>

## License

Licensed under the MIT license. See the [repository](https://github.com/martinjrobins/diffsol) for details.
