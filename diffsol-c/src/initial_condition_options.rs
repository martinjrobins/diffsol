use std::sync::{Arc, Mutex};

use serde::{Serialize, Serializer, ser::SerializeStruct};

use crate::{error::DiffsolJsError, ode::Ode};

#[derive(Clone)]
pub struct InitialConditionSolverOptions {
    ode: Arc<Mutex<Ode>>,
}
impl InitialConditionSolverOptions {
    pub(crate) fn new(ode: Arc<Mutex<Ode>>) -> Self {
        Self { ode }
    }
    fn guard(&self) -> Result<std::sync::MutexGuard<'_, Ode>, DiffsolJsError> {
        self.ode.lock().map_err(|_| {
            DiffsolJsError::from(diffsol::error::DiffsolError::Other(
                "Failed to acquire lock on ODE solver".to_string(),
            ))
        })
    }
}

impl InitialConditionSolverOptions {
    pub fn get_use_linesearch(&self) -> Result<bool, DiffsolJsError> {
        Ok(self.guard()?.solve.ic_use_linesearch())
    }
    pub fn set_use_linesearch(&self, value: bool) -> Result<(), DiffsolJsError> {
        self.guard()?.solve.set_ic_use_linesearch(value);
        Ok(())
    }
    pub fn get_max_linesearch_iterations(&self) -> Result<usize, DiffsolJsError> {
        Ok(self.guard()?.solve.ic_max_linesearch_iterations())
    }
    pub fn set_max_linesearch_iterations(&self, value: usize) -> Result<(), DiffsolJsError> {
        self.guard()?.solve.set_ic_max_linesearch_iterations(value);
        Ok(())
    }
    pub fn get_max_newton_iterations(&self) -> Result<usize, DiffsolJsError> {
        Ok(self.guard()?.solve.ic_max_newton_iterations())
    }
    pub fn set_max_newton_iterations(&self, value: usize) -> Result<(), DiffsolJsError> {
        self.guard()?.solve.set_ic_max_newton_iterations(value);
        Ok(())
    }
    pub fn get_max_linear_solver_setups(&self) -> Result<usize, DiffsolJsError> {
        Ok(self.guard()?.solve.ic_max_linear_solver_setups())
    }
    pub fn set_max_linear_solver_setups(&self, value: usize) -> Result<(), DiffsolJsError> {
        self.guard()?.solve.set_ic_max_linear_solver_setups(value);
        Ok(())
    }
    pub fn get_step_reduction_factor(&self) -> Result<f64, DiffsolJsError> {
        Ok(self.guard()?.solve.ic_step_reduction_factor())
    }
    pub fn set_step_reduction_factor(&self, value: f64) -> Result<(), DiffsolJsError> {
        self.guard()?.solve.set_ic_step_reduction_factor(value);
        Ok(())
    }
    pub fn get_armijo_constant(&self) -> Result<f64, DiffsolJsError> {
        Ok(self.guard()?.solve.ic_armijo_constant())
    }
    pub fn set_armijo_constant(&self, value: f64) -> Result<(), DiffsolJsError> {
        self.guard()?.solve.set_ic_armijo_constant(value);
        Ok(())
    }
}

impl Serialize for InitialConditionSolverOptions {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("InitialConditionSolverOptions", 6)?;
        state.serialize_field(
            "use_linesearch",
            &self
                .get_use_linesearch()
                .map_err(serde::ser::Error::custom)?,
        )?;
        state.serialize_field(
            "max_linesearch_iterations",
            &self
                .get_max_linesearch_iterations()
                .map_err(serde::ser::Error::custom)?,
        )?;
        state.serialize_field(
            "max_newton_iterations",
            &self
                .get_max_newton_iterations()
                .map_err(serde::ser::Error::custom)?,
        )?;
        state.serialize_field(
            "max_linear_solver_setups",
            &self
                .get_max_linear_solver_setups()
                .map_err(serde::ser::Error::custom)?,
        )?;
        state.serialize_field(
            "step_reduction_factor",
            &self
                .get_step_reduction_factor()
                .map_err(serde::ser::Error::custom)?,
        )?;
        state.serialize_field(
            "armijo_constant",
            &self
                .get_armijo_constant()
                .map_err(serde::ser::Error::custom)?,
        )?;
        state.end()
    }
}
