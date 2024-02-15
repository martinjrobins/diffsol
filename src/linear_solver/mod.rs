pub mod lu;
pub mod gmres;

#[cfg(test)]
pub mod tests {
    use std::rc::Rc;

    use crate::{callable::{linear_closure::LinearClosure, LinearOp}, solver::LinearSolveSolution, vector::VectorRef, Matrix, Solver, SolverProblem, Vector, LU};
    use num_traits::{One, Zero};

    fn linear_problem<M: Matrix + 'static>() -> (SolverProblem<impl LinearOp<M = M, V = M::V, T = M::T>>, Vec<LinearSolveSolution<M::V>>) {
        let diagonal = M::V::from_vec(vec![2.0.into(), 2.0.into()]);
        let jac = M::from_diagonal(&diagonal);
        let op = Rc::new(LinearClosure::new(
            // f = J * x
            move | x, _p, _t, y | jac.gemv(M::T::one(), x, M::T::zero(), y),
            2, 2, 0
        ));
        let p = M::V::zeros(0);
        let t = M::T::zero();
        let problem = SolverProblem::new(op, p, t);
        let solns = vec![
            LinearSolveSolution::new(M::V::from_vec(vec![2.0.into(), 4.0.into()]), M::V::from_vec(vec![1.0.into(), 2.0.into()]))
        ];
        (problem, solns)
    }

    pub fn test_linear_solver<C>(mut solver: impl Solver<C>, problem: SolverProblem<C>, solns: Vec<LinearSolveSolution<C::V>>) 
    where
        C: LinearOp,
        for <'a> &'a C::V: VectorRef<C::V>,
    {
        let problem = Rc::new(problem);
        solver.set_problem(problem.clone());
        let problem = problem.as_ref();
        for soln in solns {
            let x = solver.solve(&soln.b).unwrap();
            let tol = &soln.x * problem.rtol + &problem.atol;
            x.assert_eq(&soln.x, tol[0]);
        }
    }

    type MCpu = nalgebra::DMatrix<f64>;
    
    #[test]
    fn test_lu() {
        let (p, solns) = linear_problem::<MCpu>();
        let s = LU::default();
        test_linear_solver(s, p, solns);
    }
}