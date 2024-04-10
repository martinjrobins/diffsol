use crate::{op::Op, solver::SolverProblem};
use anyhow::Result;

pub mod gmres;
#[cfg(feature = "nalgebra")]
pub mod nalgebra;

#[cfg(feature = "faer")]
pub mod faer;

#[cfg(feature = "sundials")]
pub mod sundials;

pub use faer::lu::LU as FaerLU;
pub use nalgebra::lu::LU as NalgebraLU;

/// A solver for the linear problem `Ax = b`.
/// The solver is parameterised by the type `C` which is the type of the linear operator `A` (see the [Op] trait for more details).
pub trait LinearSolver<C: Op> {
    /// Set the problem to be solved, any previous problem is discarded.
    /// Any internal state of the solver is reset.
    fn set_problem(&mut self, problem: SolverProblem<C>);

    /// Get a reference to the current problem, if any.
    fn problem(&self) -> Option<&SolverProblem<C>>;

    /// Get a mutable reference to the current problem, if any.
    fn problem_mut(&mut self) -> Option<&mut SolverProblem<C>>;

    /// Take the current problem, if any, and return it.
    fn take_problem(&mut self) -> Option<SolverProblem<C>>;

    fn reset(&mut self) {
        if let Some(problem) = self.take_problem() {
            self.set_problem(problem);
        }
    }

    /// Solve the problem `Ax = b` and return the solution `x`.
    fn solve(&self, b: &C::V) -> Result<C::V> {
        let mut b = b.clone();
        self.solve_in_place(&mut b)?;
        Ok(b)
    }

    fn solve_in_place(&self, b: &mut C::V) -> Result<()>;
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
        linear_solver::FaerLU,
        linear_solver::NalgebraLU,
        op::{linear_closure::LinearClosure, LinearOp},
        scalar::scale,
        vector::VectorRef,
        DenseMatrix, LinearSolver, SolverProblem, Vector,
    };
    use num_traits::{One, Zero};

    use super::LinearSolveSolution;

    fn linear_problem<M: DenseMatrix + 'static>() -> (
        SolverProblem<impl LinearOp<M = M, V = M::V, T = M::T>>,
        Vec<LinearSolveSolution<M::V>>,
    ) {
        let diagonal = M::V::from_vec(vec![2.0.into(), 2.0.into()]);
        let jac = M::from_diagonal(&diagonal);
        let p = Rc::new(M::V::zeros(0));
        let op = Rc::new(LinearClosure::new(
            // f = J * x
            move |x, _p, _t, y| jac.gemv(M::T::one(), x, M::T::zero(), y),
            2,
            2,
            p,
        ));
        let t = M::T::zero();
        let rtol = M::T::from(1e-6);
        let atol = Rc::new(M::V::from_vec(vec![1e-6.into(), 1e-6.into()]));
        let problem = SolverProblem::new(op, t, atol, rtol);
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
        C: LinearOp,
        for<'a> &'a C::V: VectorRef<C::V>,
    {
        solver.set_problem(problem);
        for soln in solns {
            let x = solver.solve(&soln.b).unwrap();
            let tol = {
                let problem = solver.problem().unwrap();
                &soln.x * scale(problem.rtol) + problem.atol.as_ref()
            };
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
