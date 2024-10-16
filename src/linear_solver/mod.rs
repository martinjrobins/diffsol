
use std::rc::Rc;

use crate::{error::DiffsolError, Matrix, NonLinearOpJacobian};

#[cfg(feature = "nalgebra")]
pub mod nalgebra;

#[cfg(feature = "faer")]
pub mod faer;

#[cfg(feature = "sundials")]
pub mod sundials;

#[cfg(feature = "suitesparse")]
pub mod suitesparse;

pub use faer::lu::LU as FaerLU;
pub use nalgebra::lu::LU as NalgebraLU;

/// A solver for the linear problem `Ax = b`, where `A` is a linear operator that is obtained by taking the linearisation of a nonlinear operator `C`
pub trait LinearSolver<M: Matrix>: Default {
    // sets the point at which the linearisation of the operator is evaluated
    // the operator is assumed to have the same sparsity as that given to [Self::set_problem]
    fn set_linearisation<C: NonLinearOpJacobian<V=M::V, T=M::T, M=M>>(&mut self, op: &C, x: &M::V, t: M::T);

    /// Set the problem to be solved, any previous problem is discarded.
    /// Any internal state of the solver is reset.
    fn set_problem<C:  NonLinearOpJacobian<V=M::V, T=M::T, M=M>>(&mut self, op: &C, rtol: M::T, atol: Rc<M::V>);

    /// Solve the problem `Ax = b` and return the solution `x`.
    /// panics if [Self::set_linearisation] has not been called previously
    fn solve(&self, b: &M::V) -> Result<M::V, DiffsolError> {
        let mut b = b.clone();
        self.solve_in_place(&mut b)?;
        Ok(b)
    }

    fn solve_in_place(&self, b: &mut M::V) -> Result<(), DiffsolError>;
}

pub struct LinearSolveSolution<V> {
    pub x: V,
    pub b: V,
}

impl<V> LinearSolveSolution<V> {
    pub fn new(b: V, x: V) -> Self {
        Self { x, b }
    }
}

#[cfg(test)]
pub mod tests {
    use std::rc::Rc;

    use crate::{
        linear_solver::{FaerLU, NalgebraLU},
        op::closure::Closure,
        scalar::scale,
        vector::VectorRef,
        LinearSolver, Matrix, NonLinearOpJacobian, Vector,
    };
    use num_traits::{One, Zero};

    use super::LinearSolveSolution;

    #[allow(clippy::type_complexity)]
    pub fn linear_problem<M: Matrix + 'static>() -> (
        impl NonLinearOpJacobian<M = M, V = M::V, T = M::T>,
        M::T,
        Rc<M::V>,
        Vec<LinearSolveSolution<M::V>>,
    ) {
        let diagonal = M::V::from_vec(vec![2.0.into(), 2.0.into()]);
        let jac1 = M::from_diagonal(&diagonal);
        let jac2 = M::from_diagonal(&diagonal);
        let p = Rc::new(M::V::zeros(0));
        let mut op = Closure::new(
            // f = J * x
            move |x, _p, _t, y| jac1.gemv(M::T::one(), x, M::T::zero(), y),
            move |_x, _p, _t, v, y| jac2.gemv(M::T::one(), v, M::T::zero(), y),
            2,
            2,
            p,
        );
        op.calculate_sparsity(&M::V::from_element(2, M::T::one()), M::T::zero());
        let rtol = M::T::from(1e-6);
        let atol = Rc::new(M::V::from_vec(vec![1e-6.into(), 1e-6.into()]));
        let solns = vec![LinearSolveSolution::new(
            M::V::from_vec(vec![2.0.into(), 4.0.into()]),
            M::V::from_vec(vec![1.0.into(), 2.0.into()]),
        )];
        (op, rtol, atol, solns)
    }

    pub fn test_linear_solver<C>(
        mut solver: impl LinearSolver<C::M>,
        op: C,
        rtol: C::T,
        atol: Rc<C::V>,
        solns: Vec<LinearSolveSolution<C::V>>,
    ) where
        C: NonLinearOpJacobian,
        for<'a> &'a C::V: VectorRef<C::V>,
    {
        solver.set_problem(&op, rtol, atol.clone());
        let x = C::V::zeros(op.nout());
        let t = C::T::zero();
        solver.set_linearisation(&op, &x, t);
        for soln in solns {
            let x = solver.solve(&soln.b).unwrap();
            let tol = { &soln.x * scale(rtol) + atol.as_ref() };
            x.assert_eq(&soln.x, &tol);
        }
    }

    type MCpuNalgebra = nalgebra::DMatrix<f64>;
    type MCpuFaer = faer::Mat<f64>;

    #[test]
    fn test_lu_nalgebra() {
        let (op, rtol, atol, solns) = linear_problem::<MCpuNalgebra>();
        let s = NalgebraLU::default();
        test_linear_solver(s, op, rtol, atol, solns);
    }
    #[test]
    fn test_lu_faer() {
        let (op, rtol, atol, solns) = linear_problem::<MCpuFaer>();
        let s = FaerLU::default();
        test_linear_solver(s, op, rtol, atol, solns);
    }
}
