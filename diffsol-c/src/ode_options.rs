use std::sync::{Arc, Mutex};

use crate::{error::DiffsolJsError, ode::Ode};

pub struct OdeSolverOptions {
    ode: Arc<Mutex<Ode>>,
}
impl OdeSolverOptions {
    pub(crate) fn new(ode: Arc<Mutex<Ode>>) -> Self {
        Self { ode }
    }
    pub(crate) fn guard(&self) -> Result<std::sync::MutexGuard<'_, Ode>, DiffsolJsError> {
        self.ode.lock().map_err(|_| {
            DiffsolJsError::from(diffsol::error::DiffsolError::Other(
                "Failed to acquire lock on Ode object".to_string(),
            ))
        })
    }
}

impl OdeSolverOptions {
    pub(crate) fn get_max_nonlinear_solver_iterations(&self) -> Result<usize, DiffsolJsError> {
        Ok(self.guard()?.solve.ode_max_nonlinear_solver_iterations())
    }
    pub(crate) fn set_max_nonlinear_solver_iterations(
        &self,
        value: usize,
    ) -> Result<(), DiffsolJsError> {
        self.guard()?
            .solve
            .set_ode_max_nonlinear_solver_iterations(value);
        Ok(())
    }
    pub(crate) fn get_max_error_test_failures(&self) -> Result<usize, DiffsolJsError> {
        Ok(self.guard()?.solve.ode_max_error_test_failures())
    }
    pub(crate) fn set_max_error_test_failures(&self, value: usize) -> Result<(), DiffsolJsError> {
        self.guard()?.solve.set_ode_max_error_test_failures(value);
        Ok(())
    }
    pub(crate) fn get_update_jacobian_after_steps(&self) -> Result<usize, DiffsolJsError> {
        Ok(self.guard()?.solve.ode_update_jacobian_after_steps())
    }
    pub(crate) fn set_update_jacobian_after_steps(
        &self,
        value: usize,
    ) -> Result<(), DiffsolJsError> {
        self.guard()?
            .solve
            .set_ode_update_jacobian_after_steps(value);
        Ok(())
    }
    pub(crate) fn get_update_rhs_jacobian_after_steps(&self) -> Result<usize, DiffsolJsError> {
        Ok(self.guard()?.solve.ode_update_rhs_jacobian_after_steps())
    }
    pub(crate) fn set_update_rhs_jacobian_after_steps(
        &self,
        value: usize,
    ) -> Result<(), DiffsolJsError> {
        self.guard()?
            .solve
            .set_ode_update_rhs_jacobian_after_steps(value);
        Ok(())
    }
    pub(crate) fn get_threshold_to_update_jacobian(&self) -> Result<f64, DiffsolJsError> {
        Ok(self.guard()?.solve.ode_threshold_to_update_jacobian())
    }
    pub(crate) fn set_threshold_to_update_jacobian(
        &self,
        value: f64,
    ) -> Result<(), DiffsolJsError> {
        self.guard()?
            .solve
            .set_ode_threshold_to_update_jacobian(value);
        Ok(())
    }
    pub(crate) fn get_threshold_to_update_rhs_jacobian(&self) -> Result<f64, DiffsolJsError> {
        Ok(self.guard()?.solve.ode_threshold_to_update_rhs_jacobian())
    }
    pub(crate) fn set_threshold_to_update_rhs_jacobian(
        &self,
        value: f64,
    ) -> Result<(), DiffsolJsError> {
        self.guard()?
            .solve
            .set_ode_threshold_to_update_rhs_jacobian(value);
        Ok(())
    }
    pub(crate) fn get_min_timestep(&self) -> Result<f64, DiffsolJsError> {
        Ok(self.guard()?.solve.ode_min_timestep())
    }
    pub(crate) fn set_min_timestep(&self, value: f64) -> Result<(), DiffsolJsError> {
        self.guard()?.solve.set_ode_min_timestep(value);
        Ok(())
    }
}
