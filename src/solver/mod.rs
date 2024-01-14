use crate::{Scalar, vector::Vector, callable::Callable, IndexType};
use anyhow::Result;

pub struct SolverStatistics {
    pub niter: IndexType,
    pub nmaxiter: IndexType,
}

pub trait Solver<'a, T: Scalar, V: Vector<T>>: {
    fn set_callable(&mut self, c: &'a impl Callable<T, V>, p: &'a V) where Self: Sized;
    fn is_callable_set(&self) -> bool;
    fn clear_callable(&mut self);
    fn solve(&mut self, b: &V) -> Result<V>;
    fn get_statistics(&self) -> &SolverStatistics;
}