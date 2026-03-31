use std::sync::{Arc, Mutex};

use crate::{error::DiffsolJsError, ode::Ode};

pub struct InitialConditionSolverOptions {
    ode: Arc<Mutex<Ode>>,
}
impl InitialConditionSolverOptions {
    pub(crate) fn new(ode: Arc<Mutex<Ode>>) -> Self {
        Self { ode }
    }
    pub(crate) fn guard(&self) -> Result<std::sync::MutexGuard<'_, Ode>, DiffsolJsError> {
        self.ode.lock().map_err(|_| {
            DiffsolJsError::from(diffsol::error::DiffsolError::Other(
                "Failed to acquire lock on ODE solver".to_string(),
            ))
        })
    }
}

impl InitialConditionSolverOptions {
    pub(crate) fn get_use_linesearch(&self) -> Result<bool, DiffsolJsError> {
        Ok(self.guard()?.solve.ic_use_linesearch())
    }
    pub(crate) fn set_use_linesearch(&self, value: bool) -> Result<(), DiffsolJsError> {
        self.guard()?.solve.set_ic_use_linesearch(value);
        Ok(())
    }
    pub(crate) fn get_max_linesearch_iterations(&self) -> Result<usize, DiffsolJsError> {
        Ok(self.guard()?.solve.ic_max_linesearch_iterations())
    }
    pub(crate) fn set_max_linesearch_iterations(&self, value: usize) -> Result<(), DiffsolJsError> {
        self.guard()?.solve.set_ic_max_linesearch_iterations(value);
        Ok(())
    }
    pub(crate) fn get_max_newton_iterations(&self) -> Result<usize, DiffsolJsError> {
        Ok(self.guard()?.solve.ic_max_newton_iterations())
    }
    pub(crate) fn set_max_newton_iterations(&self, value: usize) -> Result<(), DiffsolJsError> {
        self.guard()?.solve.set_ic_max_newton_iterations(value);
        Ok(())
    }
    pub(crate) fn get_max_linear_solver_setups(&self) -> Result<usize, DiffsolJsError> {
        Ok(self.guard()?.solve.ic_max_linear_solver_setups())
    }
    pub(crate) fn set_max_linear_solver_setups(&self, value: usize) -> Result<(), DiffsolJsError> {
        self.guard()?.solve.set_ic_max_linear_solver_setups(value);
        Ok(())
    }
    pub(crate) fn get_step_reduction_factor(&self) -> Result<f64, DiffsolJsError> {
        Ok(self.guard()?.solve.ic_step_reduction_factor())
    }
    pub(crate) fn set_step_reduction_factor(&self, value: f64) -> Result<(), DiffsolJsError> {
        self.guard()?.solve.set_ic_step_reduction_factor(value);
        Ok(())
    }
    pub(crate) fn get_armijo_constant(&self) -> Result<f64, DiffsolJsError> {
        Ok(self.guard()?.solve.ic_armijo_constant())
    }
    pub(crate) fn set_armijo_constant(&self, value: f64) -> Result<(), DiffsolJsError> {
        self.guard()?.solve.set_ic_armijo_constant(value);
        Ok(())
    }
}
