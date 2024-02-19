use std::rc::Rc;

use crate::{op::{linearise::LinearisedOp, ConstantOp, LinearOp, Op}, ode_solver::OdeSolverProblem, IndexType, NonLinearOp};


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



impl<C: NonLinearOp> SolverProblem<C> {
    pub fn linearise(&self, x: &C::V) -> SolverProblem<LinearisedOp<C>> {
        let linearised_f= Rc::new(LinearisedOp::new(self.f.clone(), x));
        SolverProblem::new_from_problem(linearised_f, self)
    }
    
}





