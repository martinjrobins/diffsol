use nalgebra::{DVector, Dyn};
use anyhow::Result;

use crate::{Scalar, callable::Callable, solver::Solver};

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


impl<'a, T: Scalar> Solver<'a, T, DVector<T>> for LU<T> {

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
    
    fn is_callable_set(&self) -> bool {
        self.lu.is_some()
    }
    
    fn clear_callable(&mut self) {
        self.lu = None;
    }

    fn set_callable(&mut self, c: &'a impl Callable<T, DVector<T>>, p: &'a DVector<T>) {
        self.lu = Some(nalgebra::LU::new(c.jacobian(p)));
    }
}
