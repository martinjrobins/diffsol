use anyhow::Result;

use crate::{op::LinearOp, vector::VectorRef, LinearSolver, SolverProblem};

pub struct GMRES<C: LinearOp>
where
    for<'a> &'a C::V: VectorRef<C::V>,
{
    _phantom: std::marker::PhantomData<C>,
}

impl<C: LinearOp> GMRES<C>
where
    for<'a> &'a C::V: VectorRef<C::V>,
{
    // ...
}

// implement default for gmres
impl<C: LinearOp> Default for GMRES<C>
where
    for<'a> &'a C::V: VectorRef<C::V>,
{
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<C: LinearOp> LinearSolver<C> for GMRES<C>
where
    for<'b> &'b C::V: VectorRef<C::V>,
{

    fn set_linearisation(&mut self, x: &<C as crate::op::Op>::V, t: <C as crate::op::Op>::T) {
        todo!()
    }

    fn solve_in_place(&self, _state: &mut C::V) -> Result<()> {
        todo!()
    }

    fn set_problem(&mut self, _problem: &SolverProblem<C>) {
        todo!()
    }
}
