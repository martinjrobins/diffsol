pub mod lu;
pub mod gmres;

#[cfg(test)]
pub mod tests {
    use std::rc::Rc;

    use crate::{callable::linear_closure::LinearClosure, Matrix, Solver, SolverProblem, Vector, LU};
    use num_traits::{One, Zero};

    // f = J * x
    fn square<M: Matrix>(x: &M::V, _p: &M::V, y: &mut M::V, jac: &M) {
        jac.gemv(M::T::one(), x, M::T::zero(), y); // y = J * x
    }


    pub fn test_linear_solver<M: Matrix + 'static, S: Solver<LinearClosure<M, M>>>(mut solver: S) {
        let diagonal = M::V::from_vec(vec![2.0.into(), 2.0.into()]);
        let data = M::from_diagonal(&diagonal);
        let op = Rc::new(LinearClosure::<M, M>::new(
            square,
            data, 
            2,
        ));
        let p = M::V::zeros(0);
        let problem = Rc::new(SolverProblem::new(op, p));
        solver.set_problem(problem);
        let b = M::V::from_vec(vec![2.0.into(), 4.0.into()]);
        let x = solver.solve(&b).unwrap();
        let expect = M::V::from_vec(vec![(1.0).into(), 2.0.into()]);
        x.assert_eq(&expect, 1e-6.into());
    }
    
    #[test]
    fn test_lu() {
        type T = f64;
        type M = nalgebra::DMatrix<T>;
        type S = LU<T>;
        test_linear_solver::<M, S>(S::default());
    }
}