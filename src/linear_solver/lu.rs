use std::rc::Rc;

use nalgebra::{DVector, Dyn, DMatrix};
use anyhow::Result;

use crate::{callable::LinearOp, Scalar, Solver, SolverProblem};

pub struct LU<T: Scalar> {
    lu: Option<nalgebra::LU<T, Dyn, Dyn>>,
}

impl<T: Scalar> Default for LU<T> {
    fn default() -> Self {
        Self {
            lu: None,
        }
    }
}

impl<T: Scalar, C: LinearOp<M = DMatrix<T>, V = DVector<T>, T = T>> Solver<C> for LU<T> {
    fn is_problem_set(&self) -> bool {
        self.lu.is_some()
    }

    fn clear_problem(&mut self) {
        self.lu = None;
    }
    
    fn solve_in_place(&mut self, state: &mut C::V) -> Result<()> {
        if self.lu.is_none() {
            return Err(anyhow::anyhow!("LU not initialized"));
        }
        let lu = self.lu.as_ref().unwrap();
        match lu.solve_mut(state) {
            true => Ok(()),
            false => Err(anyhow::anyhow!("LU solve failed")),
        }
    }

    fn set_problem(&mut self, problem: Rc<SolverProblem<C>>) {
        self.lu = Some(nalgebra::LU::new(problem.f.jacobian(&problem.p, problem.t)));
    }
}
    
