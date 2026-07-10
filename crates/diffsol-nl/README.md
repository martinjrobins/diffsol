# diffsol-nl

Non-linear solver foundation for [diffsol](https://github.com/martinjrobins/diffsol) —
the non-linear operator and solver traits used by the ODE/DAE solvers,
together with a Newton implementation.

Most users should depend on the [`diffsol`](https://crates.io/crates/diffsol) crate
directly. Use `diffsol-nl` when you want to build on the non-linear solver layer itself.

## Traits

- **Operators**: the `NonLinearOp` trait describing a function `F(x)`, and
  `NonLinearOpJacobian` describing its Jacobian `J(x)`. These are deliberately
  time-unaware; the richer, time-aware operator traits live in the `diffsol` crate.
- **Solvers**: the `NonLinearSolver` trait for solving `F(x) = 0`.
- **Support types**: `Convergence`, the `LineSearch` trait with `NoLineSearch` and
  `BacktrackingLineSearch` implementations, and the `NlError` error type.

## Implementations

- `NewtonNonlinearSolver`: a Newton solver built on top of a
  [`diffsol-la`](https://crates.io/crates/diffsol-la) `LinearSolver`.

Each linear-algebra backend is selected via a feature flag, forwarded to `diffsol-la`:

- `nalgebra`: nalgebra-backed containers and solvers (enabled by default).
- `faer`: faer-backed containers and solvers (enabled by default).
- `cuda`: in-built CUDA containers and solvers (disabled by default).
- `suitesparse`: the KLU sparse direct solver from SuiteSparse (disabled by default).

## Links

- Main crate: <https://crates.io/crates/diffsol>
- Documentation: <https://docs.rs/diffsol-nl>
- Book: <https://martinjrobins.github.io/diffsol/>

## License

Licensed under the MIT license. See the [repository](https://github.com/martinjrobins/diffsol) for details.
