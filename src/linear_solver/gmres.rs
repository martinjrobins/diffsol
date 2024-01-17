use anyhow::Result;

use crate::{Scalar, Vector, Solver, SolverStatistics, callable::Callable};


pub struct GMRES<T: Scalar, V: Vector<T>, C: Callable<T, V>> {
    // ...
}

impl<T: Scalar, V: Vector<T>, C: Callable<T, V>> GMRES<T, V, C> {
    // ...
}

impl<'a, T: Scalar, V: Vector<T>, C: Callable<T, V>> Solver<'a, T, V, C> for GMRES<T, V, C> {
    fn set_callable(&mut self, c: &'a C, p: &'a V) {
        todo!()
    }

    fn get_statistics(&self) -> &SolverStatistics {
        todo!()
    }

    fn is_callable_set(&self) -> bool {
        todo!()
    }

    fn clear_callable(&mut self) {
        todo!()
    }

    fn solve(&self, b: &V) -> Result<V> {
        todo!()
    }
}