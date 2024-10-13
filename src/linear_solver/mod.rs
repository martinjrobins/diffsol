use crate::{error::DiffsolError, op::Op, solver::SolverProblem, NonLinearOpJacobian};

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
pub trait LinearSolver<C: Op>: Default {
    type SelfNewOp<C2: NonLinearOpJacobian<T = C::T, V = C::V, M = C::M>>: LinearSolver<C2>;

    /// Set the problem to be solved, any previous problem is discarded.
    /// Any internal state of the solver is reset.
    fn set_problem(&mut self, problem: &SolverProblem<C>);

    /// Clear the current problem, any internal state of the solver is reset.
    fn clear_problem(&mut self);

    // sets the point at which the linearisation of the operator is evaluated
    fn set_linearisation(&mut self, x: &C::V, t: C::T);

    /// Solve the problem `Ax = b` and return the solution `x`.
    /// panics if [Self::set_linearisation] has not been called previously
    fn solve(&self, b: &C::V) -> Result<C::V, DiffsolError> {
        let mut b = b.clone();
        self.solve_in_place(&mut b)?;
        Ok(b)
    }

    fn solve_in_place(&self, b: &mut C::V) -> Result<(), DiffsolError>;
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
        NonLinearOpJacobian,
        scalar::scale,
        vector::VectorRef,
        LinearSolver, Matrix, SolverProblem, Vector,
    };
    use num_traits::{One, Zero};

    use super::LinearSolveSolution;

    pub fn linear_problem<M: Matrix + 'static>() -> (
        SolverProblem<impl NonLinearOpJacobian<M = M, V = M::V, T = M::T>>,
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
        let op = Rc::new(op);
        let rtol = M::T::from(1e-6);
        let atol = Rc::new(M::V::from_vec(vec![1e-6.into(), 1e-6.into()]));
        let problem = SolverProblem::new(op, atol, rtol);
        let solns = vec![LinearSolveSolution::new(
            M::V::from_vec(vec![2.0.into(), 4.0.into()]),
            M::V::from_vec(vec![1.0.into(), 2.0.into()]),
        )];
        (problem, solns)
    }

    pub fn test_linear_solver<C>(
        mut solver: impl LinearSolver<C>,
        problem: SolverProblem<C>,
        solns: Vec<LinearSolveSolution<C::V>>,
    ) where
        C: NonLinearOpJacobian,
        for<'a> &'a C::V: VectorRef<C::V>,
    {
        solver.set_problem(&problem);
        let x = C::V::zeros(problem.f.nout());
        let t = C::T::zero();
        solver.set_linearisation(&x, t);
        for soln in solns {
            let x = solver.solve(&soln.b).unwrap();
            let tol = { &soln.x * scale(problem.rtol) + problem.atol.as_ref() };
            x.assert_eq(&soln.x, &tol);
        }
    }

    type MCpuNalgebra = nalgebra::DMatrix<f64>;
    type MCpuFaer = faer::Mat<f64>;

    #[test]
    fn test_lu_nalgebra() {
        let (p, solns) = linear_problem::<MCpuNalgebra>();
        let s = NalgebraLU::default();
        test_linear_solver(s, p, solns);
    }
    #[test]
    fn test_lu_faer() {
        let (p, solns) = linear_problem::<MCpuFaer>();
        let s = FaerLU::default();
        test_linear_solver(s, p, solns);
    }
}
