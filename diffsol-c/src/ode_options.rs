use std::sync::{Arc, Mutex};

use serde::{Serialize, Serializer, ser::SerializeStruct};

use crate::{error::DiffsolJsError, ode::Ode};

#[derive(Clone)]
pub struct OdeSolverOptions {
    ode: Arc<Mutex<Ode>>,
}
impl OdeSolverOptions {
    pub(crate) fn new(ode: Arc<Mutex<Ode>>) -> Self {
        Self { ode }
    }
    fn guard(&self) -> Result<std::sync::MutexGuard<'_, Ode>, DiffsolJsError> {
        self.ode.lock().map_err(|_| {
            DiffsolJsError::from(diffsol::error::DiffsolError::Other(
                "Failed to acquire lock on Ode object".to_string(),
            ))
        })
    }
}

impl OdeSolverOptions {
    pub fn get_max_nonlinear_solver_iterations(&self) -> Result<usize, DiffsolJsError> {
        Ok(self.guard()?.solve.ode_max_nonlinear_solver_iterations())
    }
    pub fn set_max_nonlinear_solver_iterations(&self, value: usize) -> Result<(), DiffsolJsError> {
        self.guard()?
            .solve
            .set_ode_max_nonlinear_solver_iterations(value);
        Ok(())
    }
    pub fn get_max_error_test_failures(&self) -> Result<usize, DiffsolJsError> {
        Ok(self.guard()?.solve.ode_max_error_test_failures())
    }
    pub fn set_max_error_test_failures(&self, value: usize) -> Result<(), DiffsolJsError> {
        self.guard()?.solve.set_ode_max_error_test_failures(value);
        Ok(())
    }
    pub fn get_update_jacobian_after_steps(&self) -> Result<usize, DiffsolJsError> {
        Ok(self.guard()?.solve.ode_update_jacobian_after_steps())
    }
    pub fn set_update_jacobian_after_steps(&self, value: usize) -> Result<(), DiffsolJsError> {
        self.guard()?
            .solve
            .set_ode_update_jacobian_after_steps(value);
        Ok(())
    }
    pub fn get_update_rhs_jacobian_after_steps(&self) -> Result<usize, DiffsolJsError> {
        Ok(self.guard()?.solve.ode_update_rhs_jacobian_after_steps())
    }
    pub fn set_update_rhs_jacobian_after_steps(&self, value: usize) -> Result<(), DiffsolJsError> {
        self.guard()?
            .solve
            .set_ode_update_rhs_jacobian_after_steps(value);
        Ok(())
    }
    pub fn get_threshold_to_update_jacobian(&self) -> Result<f64, DiffsolJsError> {
        Ok(self.guard()?.solve.ode_threshold_to_update_jacobian())
    }
    pub fn set_threshold_to_update_jacobian(&self, value: f64) -> Result<(), DiffsolJsError> {
        self.guard()?
            .solve
            .set_ode_threshold_to_update_jacobian(value);
        Ok(())
    }
    pub fn get_threshold_to_update_rhs_jacobian(&self) -> Result<f64, DiffsolJsError> {
        Ok(self.guard()?.solve.ode_threshold_to_update_rhs_jacobian())
    }
    pub fn set_threshold_to_update_rhs_jacobian(&self, value: f64) -> Result<(), DiffsolJsError> {
        self.guard()?
            .solve
            .set_ode_threshold_to_update_rhs_jacobian(value);
        Ok(())
    }
    pub fn get_min_timestep(&self) -> Result<f64, DiffsolJsError> {
        Ok(self.guard()?.solve.ode_min_timestep())
    }
    pub fn set_min_timestep(&self, value: f64) -> Result<(), DiffsolJsError> {
        self.guard()?.solve.set_ode_min_timestep(value);
        Ok(())
    }
}

impl Serialize for OdeSolverOptions {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("OdeSolverOptions", 7)?;
        state.serialize_field(
            "max_nonlinear_solver_iterations",
            &self
                .get_max_nonlinear_solver_iterations()
                .map_err(serde::ser::Error::custom)?,
        )?;
        state.serialize_field(
            "max_error_test_failures",
            &self
                .get_max_error_test_failures()
                .map_err(serde::ser::Error::custom)?,
        )?;
        state.serialize_field(
            "update_jacobian_after_steps",
            &self
                .get_update_jacobian_after_steps()
                .map_err(serde::ser::Error::custom)?,
        )?;
        state.serialize_field(
            "update_rhs_jacobian_after_steps",
            &self
                .get_update_rhs_jacobian_after_steps()
                .map_err(serde::ser::Error::custom)?,
        )?;
        state.serialize_field(
            "threshold_to_update_jacobian",
            &self
                .get_threshold_to_update_jacobian()
                .map_err(serde::ser::Error::custom)?,
        )?;
        state.serialize_field(
            "threshold_to_update_rhs_jacobian",
            &self
                .get_threshold_to_update_rhs_jacobian()
                .map_err(serde::ser::Error::custom)?,
        )?;
        state.serialize_field(
            "min_timestep",
            &self.get_min_timestep().map_err(serde::ser::Error::custom)?,
        )?;
        state.end()
    }
}
