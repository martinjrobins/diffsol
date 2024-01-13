use anyhow::Result;

use crate::{Scalar, Vector, Solver, SolverStatistics};


pub struct GMRES<T: Scalar, V: Vector<T>> {
    // ...
}

impl<T: Scalar, V: Vector<T>> GMRES<T, V> {
    // ...
}

impl<'a, T: Scalar, V: Vector<T>> Solver<'a, T, V> for GMRES<T, V> {
    fn set_callable(&mut self, c: &'a impl crate::callable::Callable<T, V>, p: &'a V) {
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