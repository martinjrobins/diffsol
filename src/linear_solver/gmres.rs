use anyhow::Result;

use crate::{solver::SolverProblem, vector::VectorRef, Callable, Solver};


pub struct GMRES<C: Callable> 
where 
    for <'a> &'a C::V: VectorRef<C::V>,
{
    _phantom: std::marker::PhantomData<C>,
}

impl<C: Callable> GMRES<C> 
where
    for <'a> &'a C::V: VectorRef<C::V>,
{
    // ...
}

// implement default for gmres
impl<C: Callable> Default for GMRES<C> 
where
    for <'a> &'a C::V: VectorRef<C::V>,
{
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl <C: Callable> Solver<C> for GMRES<C> 
where
    for <'b> &'b C::V: VectorRef<C::V>,
{
    fn problem(&self) -> Option<&SolverProblem<C>> {
        todo!()
    }

    fn set_problem(&mut self, _state: &<C as Callable>::V, _problem: std::rc::Rc<crate::solver::SolverProblem<C>>) {
        todo!() 
    }

    fn solve_in_place(&mut self, _state: &mut C::V) -> Result<()> {
        todo!()
    }

} 