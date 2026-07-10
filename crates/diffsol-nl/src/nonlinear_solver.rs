use diffsol_la::Matrix;

use crate::{
    convergence::Convergence,
    error::NlError,
    nonlinear_op::{NonLinearOp, NonLinearOpJacobian},
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
