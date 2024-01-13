use crate::{Scalar, vector::Vector, callable::Callable, IndexType};

pub struct SolverStatistics {
    pub niter: IndexType,
    pub nmaxiter: IndexType,
}

pub trait Solver<'a, T: Scalar, V: Vector<T>>: {
    fn set_callable(&mut self, c: &'a impl Callable<T, V>, p: &'a V);
    fn is_callable_set(&self) -> bool;
    fn clear_callable(&mut self);
    fn solve(&self, b: &V) -> Result<V>;
    fn get_statistics(&self) -> &SolverStatistics;
}