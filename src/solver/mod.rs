use std::rc::Rc;

use crate::{Scalar, vector::Vector, callable::Callable, IndexType};
use anyhow::Result;


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




pub struct SolverProblem<C: Callable> {
    pub f: Rc<C>,
    pub p: C::V,
    pub atol: C::V,
    pub rtol: C::T,
}


impl<C: Callable> SolverProblem<C> {
    pub fn new(f: Rc<C>, p: C::V) -> Self {
        let n = f.nstates();
        let rtol = C::T::from(1e-6);
        let atol = C::V::from_element(n, C::T::from(1e-6));
        Self {
            f,
            p,
            rtol,
            atol,
        }
    }
}

pub trait Solver<C: Callable> {
    fn problem(&self) -> Option<&SolverProblem<C>>;
    fn set_problem(&mut self, state: &C::V, problem: Rc<SolverProblem<C>>);
    fn solve(&mut self, state: &C::V) -> Result<C::V> {
        let mut state = state.clone();
        self.solve_in_place(&mut state)?;
        Ok(state)
    }
    fn solve_in_place(&mut self, state: &mut C::V) -> Result<()>;
}

pub trait IterativeSolver<C: Callable>: Solver<C> {
    fn set_max_iter(&mut self, max_iter: usize);
    fn max_iter(&self) -> usize;
    fn niter(&self) -> usize;
}