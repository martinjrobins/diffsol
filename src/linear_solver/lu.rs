use std::rc::Rc;

use nalgebra::{DVector, Dyn, DMatrix};
use anyhow::Result;

use crate::{Scalar, Callable, Jacobian, Solver, SolverProblem};

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


impl<T: Scalar, C: Callable<V = DVector<T>> + Jacobian<M = DMatrix<T>>> Solver<C> for LU<T> {

    fn problem(&self) -> Option<&SolverProblem<C>> {
        None
    }
    fn set_problem(&mut self, state: &DVector<T>, problem: Rc<SolverProblem<C>>) {
        self.lu = Some(nalgebra::LU::new(problem.f.jacobian(state, &problem.p)));
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
}
