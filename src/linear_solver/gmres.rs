use anyhow::Result;

use crate::{LinearSolver, vector::VectorRef, op::LinearOp, SolverProblem};


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

impl <C: LinearOp> LinearSolver<C> for GMRES<C> 
where
    for <'b> &'b C::V: VectorRef<C::V>,
{

    fn problem(&self) -> Option<&SolverProblem<C>> {
        todo!()
    } 
    fn problem_mut(&mut self) -> Option<&mut SolverProblem<C>> {
        todo!()
    }

    fn take_problem(&mut self) -> Option<SolverProblem<C>> {
        todo!()
    }
        
    fn solve_in_place(&mut self, _state: &mut C::V) -> Result<()> {
        todo!()
    }

    fn set_problem(&mut self, _problem: SolverProblem<C>) {
        todo!()
    }
}