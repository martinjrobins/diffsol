use crate::{Scalar, vector::Vector, callable::Callable, IndexType};
use anyhow::Result;

pub mod atol;

pub struct SolverStatistics {
    pub niter: IndexType,
    pub nmaxiter: IndexType,
}

pub trait Problem<T: Scalar, V: Vector<T>> {
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



pub struct SolverProblem<'a, T: Scalar, V: Vector<T>, C: Callable<T, V>> {
    pub f: &'a C,
    pub p: &'a V,
    pub atol: Option<&'a V>,
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T: Scalar, V: Vector<T>, C: Callable<T, V>> Problem<T, V> for SolverProblem<'a, T, V, C> {
    fn atol(&self) -> Option<&V> {
        self.atol
    }
    fn nstates(&self) -> IndexType {
        self.f.nstates()
    }
}

impl<'a, T: Scalar, V: Vector<T>, C: Callable<T, V>> SolverProblem<'a, T, V, C> {
    pub fn new(f: &'a C, p: &'a V) -> Self {
        Self {
            f,
            p,
            atol: None,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, T: Scalar, V: Vector<T>, C: Callable<T, V>> Clone for SolverProblem<'a, T, V, C> {
    fn clone(&self) -> Self {
        Self {
            f: self.f,
            p: self.p,
            atol: self.atol,
            _phantom: std::marker::PhantomData,
        }
    }
}





pub trait Solver<'a, T: Scalar, V: Vector<T>, C: Callable<T, V>>: {
    fn options(&self) -> Option<&SolverOptions<T>>;
    fn set_options(&mut self, options: SolverOptions<T>);
    fn problem(&self) -> Option<&SolverProblem<'a, T, V, C>>;
    fn set_problem(&mut self, state: &V, problem: SolverProblem<'a, T, V, C>);
    fn solve(&mut self, state: V) -> Result<V>;
    fn get_statistics(&self) -> &SolverStatistics;
}