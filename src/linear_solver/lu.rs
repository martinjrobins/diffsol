use nalgebra::{DVector, Dyn, DMatrix};
use anyhow::Result;

use crate::{Scalar, callable::{Callable, Jacobian}, solver::{Solver, SolverOptions, SolverProblem}, matrix::Matrix};

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


impl<'a, T: Scalar, C: Callable<T, DVector<T>> + Jacobian<T, DVector<T>, DMatrix<T>>> Solver<'a, T, DVector<T>, C> for LU<T> {
    
    fn options(&self) -> Option<&SolverOptions<T>> {
       None 
    }
    
    fn problem(&self) -> Option<&SolverProblem<'a, T, DVector<T>, C>> {
        None
    }
    
    fn set_options(&mut self, options: SolverOptions<T>) {}
    
    fn set_problem(&mut self, problem: SolverProblem<'a, T, DVector<T>, C>) {
        self.lu = Some(nalgebra::LU::new(problem.f.jacobian(problem.p)));
    }
    

    fn solve(&mut self, b: &DVector<T>) -> Result<DVector<T>> {
        if self.lu.is_none() {
            return Err(anyhow::anyhow!("LU not initialized"));
        }
        let lu = self.lu.as_ref().unwrap();
        match lu.solve(b) {
            Some(x) => Ok(x),
            None => Err(anyhow::anyhow!("LU solve failed")),
        }
    }

    fn get_statistics(&self) -> &crate::solver::SolverStatistics {
        todo!()
    }
}
