use crate::{error::DiffsolError, Matrix, NonLinearOp, NonLinearOpJacobian};
use convergence::Convergence;

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
    fn set_problem<C: NonLinearOpJacobian<V = M::V, T = M::T, M = M>>(&mut self, op: &C);

    /// Reset the approximation of the Jacobian matrix.
    fn reset_jacobian<C: NonLinearOpJacobian<V = M::V, T = M::T, M = M>>(
        &mut self,
        op: &C,
        x: &M::V,
        t: M::T,
    );

    // Solve the problem `F(x, t) = 0` for fixed t, and return the solution `x`.
    fn solve<C: NonLinearOpJacobian<V = M::V, T = M::T, M = M>>(
        &mut self,
        op: &C,
        x: &M::V,
        t: M::T,
        error_y: &M::V,
        convergence: &mut Convergence<'_, M::V>,
    ) -> Result<M::V, DiffsolError> {
        let mut x = x.clone();
        self.solve_in_place(op, &mut x, t, error_y, convergence)?;
        Ok(x)
    }

    /// Solve the problem `F(x) = 0` in place.
    fn solve_in_place<C: NonLinearOp<V = M::V, T = M::T, M = M>>(
        &mut self,
        op: &C,
        x: &mut C::V,
        t: C::T,
        error_y: &C::V,
        convergence: &mut Convergence<'_, M::V>,
    ) -> Result<(), DiffsolError>;

    /// Solve the linearised problem `J * x = b`, where `J` was calculated using [Self::reset_jacobian].
    /// The input `b` is provided in `x`, and the solution is returned in `x`.
    fn solve_linearised_in_place(&self, x: &mut M::V) -> Result<(), DiffsolError>;
}

pub mod convergence;
pub mod newton;
pub mod root;

//tests
#[cfg(test)]
pub mod tests {
    use self::newton::NewtonNonlinearSolver;
    use crate::{
        linear_solver::nalgebra::lu::LU,
        matrix::MatrixCommon,
        op::{closure::Closure, ParameterisedOp},
        scale, DenseMatrix, Vector,
    };

    use super::*;
    use num_traits::{One, Zero};

    #[allow(clippy::type_complexity)]
    pub fn get_square_problem<M>() -> (
        Closure<
            M,
            impl Fn(&M::V, &M::V, M::T, &mut M::V),
            impl Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
        >,
        M::T,
        M::V,
        Vec<NonLinearSolveSolution<M::V>>,
    )
    where
        M: DenseMatrix + 'static,
    {
        let jac1 = M::from_diagonal(&M::V::from_vec(vec![2.0.into(), 2.0.into()]));
        let jac2 = jac1.clone();
        let p = M::V::zeros(0);
        let op = Closure::new(
            // 0 = J * x * x - 8
            move |x: &<M as MatrixCommon>::V, _p: &<M as MatrixCommon>::V, _t, y| {
                jac1.gemv(M::T::one(), x, M::T::zero(), y); // y = J * x
                y.component_mul_assign(x);
                y.add_scalar_mut(M::T::from(-8.0));
            },
            // J = 2 * J * x * dx
            move |x: &<M as MatrixCommon>::V, _p: &<M as MatrixCommon>::V, _t, v, y| {
                jac2.gemv(M::T::from(2.0), x, M::T::zero(), y); // y = 2 * J * x
                y.component_mul_assign(v);
            },
            2,
            2,
            p.len(),
        );
        let rtol = M::T::from(1e-6);
        let atol = M::V::from_vec(vec![1e-6.into(), 1e-6.into()]);
        let solns = vec![NonLinearSolveSolution::new(
            M::V::from_vec(vec![2.1.into(), 2.1.into()]),
            M::V::from_vec(vec![2.0.into(), 2.0.into()]),
        )];
        (op, rtol, atol, solns)
    }

    pub fn test_nonlinear_solver<C>(
        mut solver: impl NonLinearSolver<C::M>,
        op: C,
        rtol: C::T,
        atol: &C::V,
        solns: Vec<NonLinearSolveSolution<C::V>>,
    ) where
        C: NonLinearOpJacobian,
    {
        solver.set_problem(&op);
        let mut convergence = Convergence::new(rtol, atol);
        let t = C::T::zero();
        solver.reset_jacobian(&op, &solns[0].x0, t);
        for soln in solns {
            let x = solver
                .solve(&op, &soln.x0, t, &soln.x0, &mut convergence)
                .unwrap();
            let tol = x.clone() * scale(rtol) + atol;
            x.assert_eq(&soln.x, &tol);
        }
    }

    type MCpu = nalgebra::DMatrix<f64>;

    #[test]
    fn test_newton_cpu_square() {
        let lu = LU::default();
        let (op, rtol, atol, soln) = get_square_problem::<MCpu>();
        let p = nalgebra::DVector::zeros(0);
        let op = ParameterisedOp::new(&op, &p);
        let s = NewtonNonlinearSolver::new(lu);
        test_nonlinear_solver(s, op, rtol, &atol, soln);
    }
}
