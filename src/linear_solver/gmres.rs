use anyhow::Result;

use crate::{Scalar, Vector, Solver, SolverStatistics, callable::Callable};


pub struct GMRES<T: Scalar, V: Vector<T>, C: Callable<T, V>> {
    _phantom: std::marker::PhantomData<(T, V, C)>,
}

impl<T: Scalar, V: Vector<T>, C: Callable<T, V>> GMRES<T, V, C> {
    // ...
}


impl <'a, T: Scalar, V: Vector<T>, C: Callable<T, V>> Solver<'a, T, V, C> for GMRES<T, V, C> {

    fn options(&self) -> Option<&crate::solver::SolverOptions<T>> {
        todo!()
    }

    fn set_options(&mut self, options: crate::solver::SolverOptions<T>) {
        todo!()
    }

    fn problem(&self) -> Option<&crate::solver::SolverProblem<'a, T, V, C>> {
        todo!()
    }

    fn set_problem(&mut self, state: &V, problem: crate::solver::SolverProblem<'a, T, V, C>) {
        todo!()
    }

    fn get_statistics(&self) -> &SolverStatistics {
        todo!()
    }

    fn solve(&mut self, state: V) -> Result<V> {
        todo!()
    }
} 