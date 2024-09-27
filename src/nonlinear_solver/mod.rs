use crate::{error::DiffsolError, op::Op, solver::SolverProblem, NonLinearOp};
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
pub trait NonLinearSolver<C: Op> {
    /// Get the problem to be solved.
    fn problem(&self) -> &SolverProblem<C>;

    fn convergence(&self) -> &Convergence<C::V>;

    fn convergence_mut(&mut self) -> &mut Convergence<C::V>;

    /// Set the problem to be solved, any previous problem is discarded.
    fn set_problem(&mut self, problem: &SolverProblem<C>);

    /// Reset the approximation of the Jacobian matrix.
    fn reset_jacobian(&mut self, x: &C::V, t: C::T);

    // Solve the problem `F(x, t) = 0` for fixed t, and return the solution `x`.
    fn solve(&mut self, x: &C::V, t: C::T, error_y: &C::V) -> Result<C::V, DiffsolError> {
        let mut x = x.clone();
        self.solve_in_place(&mut x, t, error_y)?;
        Ok(x)
    }

    /// Solve the problem `F(x) = 0` in place.
    fn solve_in_place(&mut self, x: &mut C::V, t: C::T, error_y: &C::V)
        -> Result<(), DiffsolError>;
    
    /// Solve the problem `G(x) = 0` in place, where the jacobian of `G` is assumed to be the same as `F`.
    fn solve_other_in_place(&mut self, g: impl NonLinearOp<M=C::M, V=C::V, T=C::T>, x: &mut C::V, t: C::T, error_y: &C::V)
        -> Result<(), DiffsolError>;

    /// Solve the linearised problem `J * x = b`, where `J` was calculated using [Self::reset_jacobian].
    /// The input `b` is provided in `x`, and the solution is returned in `x`.
    fn solve_linearised_in_place(&self, x: &mut C::V) -> Result<(), DiffsolError>;
}

pub mod convergence;
pub mod newton;
pub mod root;

//tests
#[cfg(test)]
pub mod tests {
    use std::rc::Rc;

    use self::newton::NewtonNonlinearSolver;
    use crate::{
        linear_solver::nalgebra::lu::LU,
        matrix::MatrixCommon,
        op::{closure::Closure, NonLinearOp},
        scale, DenseMatrix, Vector,
    };

    use super::*;
    use num_traits::{One, Zero};

    pub fn get_square_problem<M>() -> (
        SolverProblem<impl NonLinearOp<M = M, V = M::V, T = M::T>>,
        Vec<NonLinearSolveSolution<M::V>>,
    )
    where
        M: DenseMatrix + 'static,
    {
        let jac1 = M::from_diagonal(&M::V::from_vec(vec![2.0.into(), 2.0.into()]));
        let jac2 = jac1.clone();
        let p = Rc::new(M::V::zeros(0));
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
            p,
        );
        let rtol = M::T::from(1e-6);
        let atol = M::V::from_vec(vec![1e-6.into(), 1e-6.into()]);
        let problem = SolverProblem::new(Rc::new(op), Rc::new(atol), rtol);
        let solns = vec![NonLinearSolveSolution::new(
            M::V::from_vec(vec![2.1.into(), 2.1.into()]),
            M::V::from_vec(vec![2.0.into(), 2.0.into()]),
        )];
        (problem, solns)
    }

    pub fn test_nonlinear_solver<C>(
        mut solver: impl NonLinearSolver<C>,
        problem: SolverProblem<C>,
        solns: Vec<NonLinearSolveSolution<C::V>>,
    ) where
        C: NonLinearOp,
    {
        solver.set_problem(&problem);
        let t = C::T::zero();
        for soln in solns {
            let x = solver.solve(&soln.x0, t, &soln.x0).unwrap();
            let tol = x.clone() * scale(problem.rtol) + problem.atol.as_ref();
            x.assert_eq(&soln.x, &tol);
        }
    }

    type MCpu = nalgebra::DMatrix<f64>;

    #[test]
    fn test_newton_cpu_square() {
        let lu = LU::default();
        let (prob, soln) = get_square_problem::<MCpu>();
        let s = NewtonNonlinearSolver::new(lu);
        test_nonlinear_solver(s, prob, soln);
    }
}
