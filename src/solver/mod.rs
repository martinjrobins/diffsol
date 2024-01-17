use crate::{Scalar, vector::Vector, callable::Callable, IndexType};
use anyhow::Result;

pub struct SolverStatistics {
    pub niter: IndexType,
    pub nmaxiter: IndexType,
}

pub trait Solver<'a, T: Scalar, V: Vector<T>, C: Callable<T, V>>: {
    fn set_callable(&mut self, c: &'a C, p: &'a V);
    fn is_callable_set(&self) -> bool;
    fn clear_callable(&mut self);
    fn solve(&mut self, b: &V) -> Result<V>;
    fn get_statistics(&self) -> &SolverStatistics;
}