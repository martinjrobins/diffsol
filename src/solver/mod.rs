use std::rc::Rc;

use crate::{callable::{linearise::LinearisedOp, Op}, IndexType, NonLinearOp, Vector};
use anyhow::Result;


pub struct SolverStatistics {
    pub niter: IndexType,
    pub nmaxiter: IndexType,
}


pub struct SolverProblem<C: Op> {
    pub f: Rc<C>,
    pub p: C::V,
    pub t: C::T,
    pub atol: C::V,
    pub rtol: C::T,
}

impl<C: Op> SolverProblem<C> {
    pub fn new(f: Rc<C>, p: C::V, t: C::T) -> Self {
        let n = f.nstates();
        let rtol = C::T::from(1e-6);
        let atol = C::V::from_element(n, C::T::from(1e-6));
        Self {
            f,
            p,
            t,
            rtol,
            atol,
        }
    }
}

impl<C: NonLinearOp> SolverProblem<C> {
    pub fn linearise(&self, x: &C::V) -> SolverProblem<LinearisedOp<C>> {
        let linearised_f= Rc::new(LinearisedOp::new(self.f.clone(), x));
        SolverProblem::new(linearised_f, self.p.clone(), self.t)
    }
    
}

pub trait Solver<C: Op> {
    fn set_problem(&mut self, problem: Rc<SolverProblem<C>>);
    fn is_problem_set(&self) -> bool;
    fn clear_problem(&mut self);
    fn solve(&mut self, state: &C::V) -> Result<C::V> {
        let mut state = state.clone();
        self.solve_in_place(&mut state)?;
        Ok(state)
    }
    fn solve_in_place(&mut self, state: &mut C::V) -> Result<()>;
}

pub trait NonLinearSolver<C: NonLinearOp>: Solver<C> + IterativeSolver<C> {
    fn update_problem(&mut self, problem: Rc<SolverProblem<C>>);
}

pub trait IterativeSolver<C: NonLinearOp>: Solver<C> {
    fn set_max_iter(&mut self, max_iter: usize);
    fn max_iter(&self) -> usize;
    fn niter(&self) -> usize;
}