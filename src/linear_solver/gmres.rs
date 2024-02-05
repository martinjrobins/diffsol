use std::rc::Rc;

use anyhow::Result;

use crate::{vector::VectorRef, LinearOp, Solver, SolverProblem};


pub struct GMRES<C: LinearOp> 
where 
    for <'a> &'a C::V: VectorRef<C::V>,
{
    _phantom: std::marker::PhantomData<C>,
}

impl<C: LinearOp> GMRES<C> 
where
    for <'a> &'a C::V: VectorRef<C::V>,
{
    // ...
}

// implement default for gmres
impl<C: LinearOp> Default for GMRES<C> 
where
    for <'a> &'a C::V: VectorRef<C::V>,
{
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl <C: LinearOp> Solver<C> for GMRES<C> 
where
    for <'b> &'b C::V: VectorRef<C::V>,
{
    fn problem(&self) -> Option<&SolverProblem<C>> {
        todo!()
    }

    fn solve_in_place(&mut self, _state: &mut C::V) -> Result<()> {
        todo!()
    }

    fn set_problem(&mut self, _problem: Rc<SolverProblem<C>>) {
        todo!()
    }
}