use std::rc::Rc;

use crate::{
    op::{linearise::LinearisedOp, Op},
    IndexType, NonLinearOp, OdeEquations, OdeSolverProblem,
};

pub struct SolverStatistics {
    pub niter: IndexType,
    pub nmaxiter: IndexType,
}

/// A generic linear or nonlinear solver problem, containing the function to solve $f(t, y)$, the current time $t$, and the relative and absolute tolerances.
pub struct SolverProblem<C: Op> {
    pub f: Rc<C>,
    pub t: C::T,
    pub atol: Rc<C::V>,
    pub rtol: C::T,
}

impl<C: Op> SolverProblem<C> {
    pub fn new(f: Rc<C>, t: C::T, atol: Rc<C::V>, rtol: C::T) -> Self {
        Self { f, t, rtol, atol }
    }
    pub fn new_from_ode_problem(
        f: Rc<C>,
        other: &OdeSolverProblem<impl OdeEquations<T = C::T, V = C::V, M = C::M>>,
    ) -> Self {
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
    /// Create a new solver problem from a nonlinear operator that solves for the linearised operator.
    /// That is, if the original function is $f(t, y)$, this function creates a new problem $f'$ that solves $f' = J(x) v$, where $J(x)$ is the Jacobian of $f$ at $x$.
    pub fn linearise(&self, x: &C::V) -> SolverProblem<LinearisedOp<C>> {
        let linearised_f = Rc::new(LinearisedOp::new(self.f.clone(), x));
        SolverProblem::new_from_problem(linearised_f, self)
    }
}
