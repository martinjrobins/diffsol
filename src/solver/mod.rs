use std::rc::Rc;

use crate::{callable::{linearise::LinearisedOp, ConstantOp, LinearOp, Op}, ode_solver::OdeSolverProblem, IndexType, NonLinearOp};
use anyhow::{Result, anyhow};


pub struct SolverStatistics {
    pub niter: IndexType,
    pub nmaxiter: IndexType,
}


pub struct SolverProblem<C: Op> {
    pub f: Rc<C>,
    pub p: Rc<C::V>,
    pub t: C::T,
    pub atol: Rc<C::V>,
    pub rtol: C::T,
}

impl<C: Op> SolverProblem<C> {
    pub fn new(f: Rc<C>, p: Rc<C::V>, t: C::T, atol: Rc<C::V>, rtol: C::T) -> Self {
        Self {
            f,
            p,
            t,
            rtol,
            atol,
        }
    }
    pub fn new_from_ode_problem(f: Rc<C>, other: &OdeSolverProblem<impl NonLinearOp<M = C::M, V = C::V, T = C::T>, impl LinearOp<M = C::M, V = C::V, T = C::T> , impl ConstantOp<M = C::M, V = C::V, T = C::T>>) -> Self {
        Self {
            f,
            p: other.p.clone(),
            t: other.t0,
            rtol: other.rtol,
            atol: other.atol.clone(),
        }
    }
    pub fn new_from_problem<C2>(f: Rc<C>, other: &SolverProblem<C2>) -> Self 
    where
        C2: Op<M = C::M, V = C::V, T = C::T>,
    {
        Self {
            f,
            p: other.p.clone(),
            t: other.t,
            rtol: other.rtol,
            atol: other.atol.clone(),
        }
    }
}

pub struct LinearSolveSolution<V> {
    pub x: V,
    pub b: V,
}

impl <V> LinearSolveSolution<V> {
    pub fn new(b: V, x: V) -> Self {
        Self { x, b }
    }
}

pub struct NonLinearSolveSolution<V> {
    pub x0: V,
    pub x: V,
}

impl <V> NonLinearSolveSolution<V> {
    pub fn new(x0: V, x: V) -> Self {
        Self { x0, x }
    }
}

impl<C: NonLinearOp> SolverProblem<C> {
    pub fn linearise(&self, x: &C::V) -> SolverProblem<LinearisedOp<C>> {
        let linearised_f= Rc::new(LinearisedOp::new(self.f.clone(), x));
        SolverProblem::new_from_problem(linearised_f, self)
    }
    
}


pub trait LinearSolver<C: Op> {
    fn set_problem(&mut self, problem: SolverProblem<C>);
    fn problem(&self) -> Option<&SolverProblem<C>>;
    fn problem_mut(&mut self) -> Option<&mut SolverProblem<C>>;
    fn take_problem(&mut self) -> Option<SolverProblem<C>>;
    fn reset(&mut self) {
        if let Some(problem) = self.take_problem() {
            self.set_problem(problem);
        }
    }
    fn solve(&mut self, b: &C::V) -> Result<C::V> {
        let mut b = b.clone();
        self.solve_in_place(&mut b)?;
        Ok(b)
    }
    
    fn solve_in_place(&mut self, b: &mut C::V) -> Result<()>;
}

pub trait NonLinearSolver<C: Op> {
    fn set_problem(&mut self, problem: SolverProblem<C>);
    fn problem(&self) -> Option<&SolverProblem<C>>;
    fn problem_mut(&mut self) -> Option<&mut SolverProblem<C>>;
    fn take_problem(&mut self) -> Option<SolverProblem<C>>;
    fn reset(&mut self) {
        if let Some(problem) = self.take_problem() {
            self.set_problem(problem);
        }
    }
    fn set_time(&mut self, t: C::T) -> Result<()> {
        self.problem_mut().ok_or_else(|| anyhow!("No problem set"))?.t = t;
        Ok(())
    }
    fn solve(&mut self, state: &C::V) -> Result<C::V> {
        let mut state = state.clone();
        self.solve_in_place(&mut state)?;
        Ok(state)
    }
    fn solve_in_place(&mut self, state: &mut C::V) -> Result<()>;
    fn set_max_iter(&mut self, max_iter: usize);
    fn max_iter(&self) -> usize;
    fn niter(&self) -> usize;
}
