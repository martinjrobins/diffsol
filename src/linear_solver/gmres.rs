use anyhow::Result;

use crate::{vector::VectorRef, Callable, Solver, SolverStatistics, Vector};


pub struct GMRES<V: Vector, C: Callable<V>> 
where 
    for <'a> &'a V: VectorRef<V>,
{
    _phantom: std::marker::PhantomData<(V, C)>,
}

impl<V: Vector, C: Callable<V>> GMRES<V, C> 
where
    for <'a> &'a V: VectorRef<V>,
{
    // ...
}

// implement default for gmres
impl<V: Vector, C: Callable<V>> Default for GMRES<V, C> 
where
    for <'a> &'a V: VectorRef<V>,
{
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl <'a, V: Vector, C: Callable<V>> Solver<'a, V, C> for GMRES<V, C> 
where
    for <'b> &'b V: VectorRef<V>,
{

    fn options(&self) -> Option<&crate::solver::SolverOptions<V::T>> {
        todo!()
    }

    fn set_options(&mut self, options: crate::solver::SolverOptions<V::T>) {
        todo!()
    }

    fn problem(&self) -> Option<&crate::solver::SolverProblem<'a, V, C>> {
        todo!()
    }

    fn set_problem(&mut self, state: &V, problem: crate::solver::SolverProblem<'a, V, C>) {
        todo!()
    }

    fn get_statistics(&self) -> &SolverStatistics {
        todo!()
    }

    fn solve(&mut self, state: V) -> Result<V> {
        todo!()
    }
} 