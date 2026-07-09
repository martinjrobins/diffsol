use crate::{
    error::DiffsolError, Matrix, NonLinearOp as TimeAwareNonLinearOp,
    NonLinearOpJacobian as TimeAwareNonLinearOpJacobian,
};
use diffsol_nl::NonLinearSolver as NlNonLinearSolver;
use num_traits::Zero;

/// Re-exports of the generic, time-unaware non-linear solver machinery from the
/// [`diffsol_nl`] crate. These keep existing `crate::nonlinear_solver::<module>`
/// paths resolving.
pub use diffsol_nl::{convergence, line_search, newton};

pub use diffsol_nl::{
    BacktrackingLineSearch, Convergence, ConvergenceStatus, LineSearch, NewtonNonlinearSolver,
    NoLineSearch, NonLinearSolveSolution,
};

pub mod root;

/// A borrowing adapter that presents a time-aware [TimeAwareNonLinearOp] (evaluated
/// at a fixed time `t`) as a time-unaware [`diffsol_nl::NonLinearOp`].
///
/// This is the bridge that allows the time-aware, operator-based [NonLinearSolver]
/// trait in `diffsol` to be implemented on top of the time-unaware
/// [`diffsol_nl::NonLinearSolver`] implementations.
pub struct NonLinearisedRef<'a, C: TimeAwareNonLinearOp> {
    op: &'a C,
    t: C::T,
}

impl<'a, C: TimeAwareNonLinearOp> NonLinearisedRef<'a, C> {
    /// Create an adapter used only to query the sparsity pattern (no time set).
    pub fn sparsity_only(op: &'a C) -> Self {
        Self {
            op,
            t: C::T::zero(),
        }
    }

    /// Create an adapter that evaluates the operator at time `t`.
    pub fn at(op: &'a C, t: C::T) -> Self {
        Self { op, t }
    }
}

impl<C: TimeAwareNonLinearOp> diffsol_nl::NonLinearOp for NonLinearisedRef<'_, C> {
    type T = C::T;
    type V = C::V;
    type M = C::M;
    type C = C::C;

    fn nstates(&self) -> usize {
        self.op.nstates()
    }

    fn nout(&self) -> usize {
        self.op.nout()
    }

    fn context(&self) -> &Self::C {
        self.op.context()
    }

    fn call_inplace(&self, x: &Self::V, y: &mut Self::V) {
        self.op.call_inplace(x, self.t, y);
    }
}

impl<C: TimeAwareNonLinearOpJacobian> diffsol_nl::NonLinearOpJacobian for NonLinearisedRef<'_, C> {
    fn jac_mul_inplace(&self, x: &Self::V, v: &Self::V, y: &mut Self::V) {
        self.op.jac_mul_inplace(x, self.t, v, y);
    }

    fn jacobian_inplace(&self, x: &Self::V, y: &mut Self::M) {
        self.op.jacobian_inplace(x, self.t, y);
    }

    fn jacobian_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.op.jacobian_sparsity()
    }
}

/// A solver for the (time-aware) nonlinear problem `F(x, t) = 0` for fixed `t`.
pub trait NonLinearSolver<M: Matrix>: Default {
    /// Set the problem to be solved, any previous problem is discarded.
    fn set_problem<C: TimeAwareNonLinearOpJacobian<V = M::V, T = M::T, M = M, C = M::C>>(
        &mut self,
        op: &C,
    );

    fn is_jacobian_set(&self) -> bool;

    /// Reset the approximation of the Jacobian matrix.
    fn reset_jacobian<C: TimeAwareNonLinearOpJacobian<V = M::V, T = M::T, M = M, C = M::C>>(
        &mut self,
        op: &C,
        x: &M::V,
        t: M::T,
    );

    /// Clear the approximation of the Jacobian matrix.
    fn clear_jacobian(&mut self);

    // Solve the problem `F(x, t) = 0` for fixed t, and return the solution `x`.
    fn solve<C: TimeAwareNonLinearOp<V = M::V, T = M::T, M = M>>(
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

    /// Solve the problem `F(x, t) = 0` in place.
    fn solve_in_place<C: TimeAwareNonLinearOp<V = M::V, T = M::T, M = M>>(
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

/// Any [`diffsol_nl::NonLinearSolver`] implementation automatically implements the
/// time-aware, operator-based [NonLinearSolver] trait via the [NonLinearisedRef] bridge.
impl<M: Matrix, S: NlNonLinearSolver<M>> NonLinearSolver<M> for S {
    fn set_problem<C: TimeAwareNonLinearOpJacobian<V = M::V, T = M::T, M = M, C = M::C>>(
        &mut self,
        op: &C,
    ) {
        NlNonLinearSolver::set_problem(self, &NonLinearisedRef::sparsity_only(op));
    }

    fn is_jacobian_set(&self) -> bool {
        NlNonLinearSolver::is_jacobian_set(self)
    }

    fn reset_jacobian<C: TimeAwareNonLinearOpJacobian<V = M::V, T = M::T, M = M, C = M::C>>(
        &mut self,
        op: &C,
        x: &M::V,
        t: M::T,
    ) {
        NlNonLinearSolver::reset_jacobian(self, &NonLinearisedRef::at(op, t), x);
    }

    fn clear_jacobian(&mut self) {
        NlNonLinearSolver::clear_jacobian(self);
    }

    fn solve_in_place<C: TimeAwareNonLinearOp<V = M::V, T = M::T, M = M>>(
        &mut self,
        op: &C,
        x: &mut C::V,
        t: C::T,
        error_y: &C::V,
        convergence: &mut Convergence<'_, M::V>,
    ) -> Result<(), DiffsolError> {
        NlNonLinearSolver::solve_in_place(
            self,
            &NonLinearisedRef::at(op, t),
            x,
            error_y,
            convergence,
        )
        .map_err(Into::into)
    }

    fn solve_linearised_in_place(&self, x: &mut M::V) -> Result<(), DiffsolError> {
        NlNonLinearSolver::solve_linearised_in_place(self, x).map_err(Into::into)
    }
}

//tests
#[cfg(test)]
pub mod tests {
    use crate::{
        linear_solver::nalgebra::lu::LU,
        matrix::{dense_nalgebra_serial::NalgebraMat, MatrixCommon},
        op::{closure::Closure, ParameterisedOp},
        scale, BacktrackingLineSearch, DenseMatrix, NalgebraVec, NoLineSearch, Op, Vector,
    };

    use super::*;
    use num_traits::{FromPrimitive, One, Zero};

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
        let jac1 = M::from_diagonal(&M::V::from_vec(
            vec![M::T::from_f64(2.0).unwrap(), M::T::from_f64(2.0).unwrap()],
            Default::default(),
        ));
        let jac2 = jac1.clone();
        let p = M::V::zeros(0, jac1.context().clone());
        let eights = M::V::from_vec(
            vec![M::T::from_f64(8.0).unwrap(), M::T::from_f64(8.0).unwrap()],
            jac1.context().clone(),
        );
        let op = Closure::new(
            // 0 = J * x * x - 8
            move |x: &<M as MatrixCommon>::V, _p: &<M as MatrixCommon>::V, _t, y| {
                jac1.gemv(M::T::one(), x, M::T::zero(), y); // y = J * x
                y.component_mul_assign(x);
                y.axpy(-M::T::one(), &eights, M::T::one());
            },
            // J = 2 * J * x * dx
            move |x: &<M as MatrixCommon>::V, _p: &<M as MatrixCommon>::V, _t, v, y| {
                jac2.gemv(M::T::from_f64(2.0).unwrap(), x, M::T::zero(), y); // y = 2 * J * x
                y.component_mul_assign(v);
            },
            2,
            2,
            p.len(),
            p.context().clone(),
        );
        let rtol = M::T::from_f64(1e-6).unwrap();
        let atol = M::V::from_vec(
            vec![M::T::from_f64(1e-6).unwrap(), M::T::from_f64(1e-6).unwrap()],
            p.context().clone(),
        );
        let solns = vec![NonLinearSolveSolution::new(
            M::V::from_vec(
                vec![M::T::from_f64(2.1).unwrap(), M::T::from_f64(2.1).unwrap()],
                p.context().clone(),
            ),
            M::V::from_vec(
                vec![M::T::from_f64(2.0).unwrap(), M::T::from_f64(2.0).unwrap()],
                p.context().clone(),
            ),
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
        C: TimeAwareNonLinearOpJacobian,
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

    type MCpu = NalgebraMat<f64>;

    #[test]
    fn test_newton_cpu_square() {
        let lu = LU::default();
        let (op, rtol, atol, soln) = get_square_problem::<MCpu>();
        let p = NalgebraVec::zeros(0, *op.context());
        let op = ParameterisedOp::new(&op, &p);
        let nls = NoLineSearch;
        let s = NewtonNonlinearSolver::new(lu, nls);
        test_nonlinear_solver(s, op, rtol, &atol, soln);
    }

    #[test]
    fn test_newton_cpu_square_backtrack() {
        let lu = LU::default();
        let (op, rtol, atol, soln) = get_square_problem::<MCpu>();
        let p = NalgebraVec::zeros(0, *op.context());
        let op = ParameterisedOp::new(&op, &p);
        let ls = BacktrackingLineSearch::default();
        let s = NewtonNonlinearSolver::new(lu, ls);
        test_nonlinear_solver(s, op, rtol, &atol, soln);
    }
}
