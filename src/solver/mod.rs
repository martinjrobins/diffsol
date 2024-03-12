use std::rc::Rc;

use crate::{ode_solver::OdeSolverProblem, op::{linearise::LinearisedOp, ConstantOp, LinearOp, Op}, IndexType, NonLinearOp, OdeEquations};


pub struct SolverStatistics {
    pub niter: IndexType,
    pub nmaxiter: IndexType,
}


pub struct SolverProblem<C: Op> {
    pub f: Rc<C>,
    pub t: C::T,
    pub atol: Rc<C::V>,
    pub rtol: C::T,
}

impl<C: Op> SolverProblem<C> {
    pub fn new(f: Rc<C>, t: C::T, atol: Rc<C::V>, rtol: C::T) -> Self {
        Self {
            f,
            t,
            rtol,
            atol,
        }
    }
    pub fn new_from_ode_problem(f: Rc<C>, other: &OdeSolverProblem<impl OdeEquations<T=C::T, V=C::V, M=C::M>>) -> Self {
        Self {
            f,
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





