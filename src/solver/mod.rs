use crate::{Scalar, vector::Vector, callable::Callable, IndexType};
use anyhow::Result;

pub mod atol;

pub struct SolverStatistics {
    pub niter: IndexType,
    pub nmaxiter: IndexType,
}

pub trait Problem<V: Vector> {
    fn nstates(&self) -> IndexType;
    fn atol(&self) -> Option<&V>;
}

pub trait Options<T: Scalar> {
    fn atol(&self) -> T;
}

pub struct SolverOptions<T: Scalar> {
    pub max_iter: IndexType,
    pub rtol: T,
    pub atol: T,
}

impl <'a, T: Scalar> Default for SolverOptions<T> {
    fn default() -> Self {
        Self {
            max_iter: 100,
            rtol: T::from(1e-6),
            atol: T::from(1e-6),
        }
    }
}

impl <T: Scalar> Options<T> for SolverOptions<T> {
    fn atol(&self) -> T {
        self.atol
    }
}



pub struct SolverProblem<V: Vector, C: Callable<V>> {
    pub f: C,
    pub p: V,
    pub atol: Option<V>,
}

impl<'a, V: Vector, C: Callable<V>> Problem<V> for SolverProblem<V, C> {
    fn atol(&self) -> Option<&V> {
        self.atol
    }
    fn nstates(&self) -> IndexType {
        self.f.nstates()
    }
}

impl<'a, V: Vector, C: Callable<V>> SolverProblem<V, C> {
    pub fn new(f: &'a C, p: &'a V) -> Self {
        Self {
            f,
            p,
            atol: None,
        }
    }
}

impl<'a, V: Vector, C: Callable<V>> Clone for SolverProblem<V, C> {
    fn clone(&self) -> Self {
        Self {
            f: self.f,
            p: self.p,
            atol: self.atol,
        }
    }
}





pub trait Solver<'a, V: Vector, C: Callable<V>> {
    fn options(&self) -> Option<&SolverOptions<V::T>>;
    fn set_options(&mut self, options: SolverOptions<V::T>);
    fn problem(&self) -> Option<&SolverProblem<V, C>>;
    fn set_problem(&mut self, state: &V, problem: SolverProblem<V, C>);
    fn solve(&mut self, state: V) -> Result<V>;
    fn get_statistics(&self) -> &SolverStatistics;
}