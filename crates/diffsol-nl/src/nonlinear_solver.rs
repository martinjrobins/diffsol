use diffsol_la::Matrix;

use crate::{
    convergence::Convergence,
    error::NlError,
    nonlinear_op::{NonLinearOp, NonLinearOpAdjoint, NonLinearOpJacobian},
};

pub struct NonLinearSolveSolution<V> {
    pub x0: V,
    pub x: V,
}

impl<V> NonLinearSolveSolution<V> {
    pub fn new(x0: V, x: V) -> Self {
        Self { x0, x }
    }
}

/// A solver for the nonlinear problem `F(x) = 0`.
pub trait NonLinearSolver<M: Matrix>: Default {
    /// Set the problem to be solved, any previous problem is discarded.
    fn set_problem<C: NonLinearOpJacobian<V = M::V, T = M::T, M = M, C = M::C>>(&mut self, op: &C);

    fn is_jacobian_set(&self) -> bool;

    /// Reset the approximation of the Jacobian matrix.
    fn reset_jacobian<C: NonLinearOpJacobian<V = M::V, T = M::T, M = M, C = M::C>>(
        &mut self,
        op: &C,
        x: &M::V,
    );

    /// Clear the approximation of the Jacobian matrix.
    fn clear_jacobian(&mut self);

    /// Solve the problem `F(x) = 0` and return the solution `x`.
    fn solve<C: NonLinearOp<V = M::V, T = M::T, M = M>>(
        &mut self,
        op: &C,
        x: &M::V,
        error_y: &M::V,
        convergence: &mut Convergence<'_, M::V>,
    ) -> Result<M::V, NlError> {
        let mut x = x.clone();
        self.solve_in_place(op, &mut x, error_y, convergence)?;
        Ok(x)
    }

    /// Solve the problem `F(x) = 0` in place.
    fn solve_in_place<C: NonLinearOp<V = M::V, T = M::T, M = M>>(
        &mut self,
        op: &C,
        x: &mut C::V,
        error_y: &C::V,
        convergence: &mut Convergence<'_, M::V>,
    ) -> Result<(), NlError>;

    /// Solve the linearised problem `J * x = b`, where `J` was calculated using [Self::reset_jacobian].
    /// The input `b` is provided in `x`, and the solution is returned in `x`.
    fn solve_linearised_in_place(&self, x: &mut M::V) -> Result<(), NlError>;

    /// Solve `J(x_lin) * x = b` with the operator's own Jacobian at `x_lin`, applied matrix-free.
    ///
    /// This is [Self::solve_in_place] for the linear problem `F(x) = J(x_lin) x - b`: the
    /// Jacobian set by [Self::reset_jacobian], which may be stale, is the Newton matrix, so each
    /// iteration is a step of preconditioned Richardson iteration. `x` holds the initial guess
    /// and returns the solution; `error_y` and `convergence` control the iteration as for
    /// [Self::solve_in_place]. If the stored Jacobian is too far from `J(x_lin)` the iteration
    /// fails to converge, and the caller can [Self::reset_jacobian] and retry.
    #[allow(clippy::too_many_arguments)]
    fn solve_jacobian_in_place<C: NonLinearOpJacobian<V = M::V, T = M::T, M = M, C = M::C>>(
        &mut self,
        op: &C,
        x_lin: &M::V,
        b: &M::V,
        x: &mut M::V,
        error_y: &M::V,
        convergence: &mut Convergence<'_, M::V>,
    ) -> Result<(), NlError>;

    /// As [Self::solve_jacobian_in_place], for the transposed problem `J(x_lin)^T * x = b`,
    /// preconditioned by the transpose of the stored Jacobian.
    #[allow(clippy::too_many_arguments)]
    fn solve_jacobian_transpose_in_place<
        C: NonLinearOpAdjoint<V = M::V, T = M::T, M = M, C = M::C>,
    >(
        &mut self,
        op: &C,
        x_lin: &M::V,
        b: &M::V,
        x: &mut M::V,
        error_y: &M::V,
        convergence: &mut Convergence<'_, M::V>,
    ) -> Result<(), NlError>;
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::{
        line_search::NoLineSearch, newton::NewtonNonlinearSolver, nonlinear_op::NonLinearOp,
    };
    use diffsol_la::{
        scale, DenseMatrix, IndexType, MatrixCommon, NalgebraLU, NalgebraMat, Vector,
    };
    use num_traits::{FromPrimitive, One, Zero};

    /// A simple nonlinear operator `F(x) = J * x * x - 8` used for testing.
    struct SquareOp<M: DenseMatrix> {
        jac: M,
        eights: M::V,
        ctx: M::C,
    }

    impl<M: DenseMatrix> SquareOp<M> {
        fn new() -> Self {
            let jac = M::from_diagonal(&M::V::from_vec(
                vec![M::T::from_f64(2.0).unwrap(), M::T::from_f64(2.0).unwrap()],
                Default::default(),
            ));
            let ctx = jac.context().clone();
            let eights = M::V::from_vec(
                vec![M::T::from_f64(8.0).unwrap(), M::T::from_f64(8.0).unwrap()],
                ctx.clone(),
            );
            Self { jac, eights, ctx }
        }
    }

    impl<M: DenseMatrix> NonLinearOp for SquareOp<M> {
        type T = M::T;
        type V = M::V;
        type M = M;
        type C = M::C;

        fn nstates(&self) -> IndexType {
            2
        }
        fn nout(&self) -> IndexType {
            2
        }
        fn context(&self) -> &Self::C {
            &self.ctx
        }
        fn call_inplace(&self, x: &Self::V, y: &mut Self::V) {
            // y = J * x * x - 8
            self.jac.gemv(M::T::one(), x, M::T::zero(), y);
            y.component_mul_assign(x);
            y.axpy(-M::T::one(), &self.eights, M::T::one());
        }
    }

    impl<M: DenseMatrix> NonLinearOpJacobian for SquareOp<M> {
        fn jac_mul_inplace(&self, x: &Self::V, v: &Self::V, y: &mut Self::V) {
            // J = 2 * J * x * dx
            self.jac
                .gemv(M::T::from_f64(2.0).unwrap(), x, M::T::zero(), y);
            y.component_mul_assign(v);
        }
    }

    /// `F(x) = A x` for a non-symmetric 2x2 `A`, so that transposed solves differ.
    struct LinearOp2<M: DenseMatrix> {
        a: M,
        ctx: M::C,
    }

    impl<M: DenseMatrix> LinearOp2<M> {
        /// `A = [[3, 1], [-1, 2]] + eps * [[1, 2], [3, 4]]`
        fn new(eps: f64) -> Self {
            let f = |x: f64| M::T::from_f64(x).unwrap();
            let a = M::from_vec(
                2,
                2,
                vec![
                    f(3.0 + eps),
                    f(-1.0 + 3.0 * eps),
                    f(1.0 + 2.0 * eps),
                    f(2.0 + 4.0 * eps),
                ],
                Default::default(),
            );
            let ctx = a.context().clone();
            Self { a, ctx }
        }
    }

    impl<M: DenseMatrix> NonLinearOp for LinearOp2<M> {
        type T = M::T;
        type V = M::V;
        type M = M;
        type C = M::C;
        fn nstates(&self) -> IndexType {
            2
        }
        fn nout(&self) -> IndexType {
            2
        }
        fn context(&self) -> &Self::C {
            &self.ctx
        }
        fn call_inplace(&self, x: &Self::V, y: &mut Self::V) {
            self.a.gemv(M::T::one(), x, M::T::zero(), y);
        }
    }

    impl<M: DenseMatrix> NonLinearOpJacobian for LinearOp2<M> {
        fn jac_mul_inplace(&self, _x: &Self::V, v: &Self::V, y: &mut Self::V) {
            self.a.gemv(M::T::one(), v, M::T::zero(), y);
        }
    }

    impl<M: DenseMatrix> NonLinearOpAdjoint for LinearOp2<M> {
        fn jac_transpose_mul_inplace(&self, _x: &Self::V, v: &Self::V, y: &mut Self::V) {
            // y = -A^T v
            let (a, v) = (|i, j| self.a.get_index(i, j), v.clone_as_vec());
            let out = (0..2).map(|i| -(a(0, i) * v[0] + a(1, i) * v[1])).collect();
            y.copy_from(&M::V::from_vec(out, self.ctx.clone()));
        }
    }

    /// `solve_jacobian_in_place` and its transpose with the Jacobian stored for `A + eps E`:
    /// exact and close preconditioners give the solution of `A x = b` (or `A^T x = b`), a far
    /// one fails to converge.
    #[test]
    fn test_solve_jacobian() {
        type M = NalgebraMat<f64>;
        type V = <M as MatrixCommon>::V;
        let op = LinearOp2::<M>::new(0.0);
        let x_lin = V::zeros(2, Default::default());
        let atol = V::from_vec(vec![1e-10, 1e-10], Default::default());
        let mut convergence = Convergence::new(1e-10, &atol);
        convergence.set_max_iter(20);
        // A (1, 2) = (5, 3) and A^T (1, 2) = (1, 5)
        let expected = V::from_vec(vec![1.0, 2.0], Default::default());
        for (eps, converges) in [(0.0, true), (0.05, true), (1.5, false)] {
            let mut s = NewtonNonlinearSolver::new(NalgebraLU::default(), NoLineSearch);
            s.set_problem(&op);
            s.reset_jacobian(&LinearOp2::<M>::new(eps), &x_lin);
            for transpose in [false, true] {
                let b = if transpose {
                    vec![1.0, 5.0]
                } else {
                    vec![5.0, 3.0]
                };
                let b = V::from_vec(b, Default::default());
                let mut x = V::zeros(2, Default::default());
                convergence.reset_eta();
                let result = if transpose {
                    s.solve_jacobian_transpose_in_place(
                        &op,
                        &x_lin,
                        &b,
                        &mut x,
                        &expected,
                        &mut convergence,
                    )
                } else {
                    s.solve_jacobian_in_place(&op, &x_lin, &b, &mut x, &expected, &mut convergence)
                };
                assert_eq!(
                    result.is_ok(),
                    converges,
                    "eps = {eps}, transpose = {transpose}"
                );
                if converges {
                    x.assert_eq_st(&expected, 1e-8);
                }
            }
        }
    }

    #[test]
    fn test_newton_cpu_square() {
        type M = NalgebraMat<f64>;
        let op = SquareOp::<M>::new();
        let ctx = *op.context();
        let rtol = 1e-6;
        let atol = <M as MatrixCommon>::V::from_vec(vec![1e-6, 1e-6], ctx);
        let x0 = <M as MatrixCommon>::V::from_vec(vec![2.1, 2.1], ctx);
        let expected = <M as MatrixCommon>::V::from_vec(vec![2.0, 2.0], ctx);

        let mut s = NewtonNonlinearSolver::new(NalgebraLU::default(), NoLineSearch);
        s.set_problem(&op);
        let mut convergence = Convergence::new(rtol, &atol);
        s.reset_jacobian(&op, &x0);
        let x = s.solve(&op, &x0, &x0, &mut convergence).unwrap();
        let tol = x.clone() * scale(rtol) + &atol;
        x.assert_eq(&expected, &tol);
    }
}
