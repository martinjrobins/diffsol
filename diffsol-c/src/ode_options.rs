use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize, Serializer};

use crate::{error::DiffsolRtError, ode::Ode, solve::Solve};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OdeSolverOptionsSnapshot {
    pub max_nonlinear_solver_iterations: usize,
    pub max_error_test_failures: usize,
    pub update_jacobian_after_steps: usize,
    pub update_rhs_jacobian_after_steps: usize,
    pub threshold_to_update_jacobian: f64,
    pub threshold_to_update_rhs_jacobian: f64,
    pub min_timestep: f64,
}

impl OdeSolverOptionsSnapshot {
    pub(crate) fn from_solve(solve: &dyn Solve) -> Self {
        Self {
            max_nonlinear_solver_iterations: solve.ode_max_nonlinear_solver_iterations(),
            max_error_test_failures: solve.ode_max_error_test_failures(),
            update_jacobian_after_steps: solve.ode_update_jacobian_after_steps(),
            update_rhs_jacobian_after_steps: solve.ode_update_rhs_jacobian_after_steps(),
            threshold_to_update_jacobian: solve.ode_threshold_to_update_jacobian(),
            threshold_to_update_rhs_jacobian: solve.ode_threshold_to_update_rhs_jacobian(),
            min_timestep: solve.ode_min_timestep(),
        }
    }

    #[cfg_attr(
        not(any(feature = "diffsl-cranelift", feature = "diffsl-llvm")),
        allow(dead_code)
    )]
    pub(crate) fn apply_to_solve(&self, solve: &mut dyn Solve) {
        solve.set_ode_max_nonlinear_solver_iterations(self.max_nonlinear_solver_iterations);
        solve.set_ode_max_error_test_failures(self.max_error_test_failures);
        solve.set_ode_update_jacobian_after_steps(self.update_jacobian_after_steps);
        solve.set_ode_update_rhs_jacobian_after_steps(self.update_rhs_jacobian_after_steps);
        solve.set_ode_threshold_to_update_jacobian(self.threshold_to_update_jacobian);
        solve.set_ode_threshold_to_update_rhs_jacobian(self.threshold_to_update_rhs_jacobian);
        solve.set_ode_min_timestep(self.min_timestep);
    }
}

#[derive(Clone)]
pub struct OdeSolverOptions {
    ode: Arc<Mutex<Ode>>,
}
impl OdeSolverOptions {
    pub(crate) fn new(ode: Arc<Mutex<Ode>>) -> Self {
        Self { ode }
    }
    fn guard(&self) -> Result<std::sync::MutexGuard<'_, Ode>, DiffsolRtError> {
        self.ode.lock().map_err(|_| {
            DiffsolRtError::from(diffsol::error::DiffsolError::Other(
                "Failed to acquire lock on Ode object".to_string(),
            ))
        })
    }
}

impl OdeSolverOptions {
    pub fn get_max_nonlinear_solver_iterations(&self) -> Result<usize, DiffsolRtError> {
        Ok(self.guard()?.solve.ode_max_nonlinear_solver_iterations())
    }
    pub fn set_max_nonlinear_solver_iterations(&self, value: usize) -> Result<(), DiffsolRtError> {
        self.guard()?
            .solve
            .set_ode_max_nonlinear_solver_iterations(value);
        Ok(())
    }
    pub fn get_max_error_test_failures(&self) -> Result<usize, DiffsolRtError> {
        Ok(self.guard()?.solve.ode_max_error_test_failures())
    }
    pub fn set_max_error_test_failures(&self, value: usize) -> Result<(), DiffsolRtError> {
        self.guard()?.solve.set_ode_max_error_test_failures(value);
        Ok(())
    }
    pub fn get_update_jacobian_after_steps(&self) -> Result<usize, DiffsolRtError> {
        Ok(self.guard()?.solve.ode_update_jacobian_after_steps())
    }
    pub fn set_update_jacobian_after_steps(&self, value: usize) -> Result<(), DiffsolRtError> {
        self.guard()?
            .solve
            .set_ode_update_jacobian_after_steps(value);
        Ok(())
    }
    pub fn get_update_rhs_jacobian_after_steps(&self) -> Result<usize, DiffsolRtError> {
        Ok(self.guard()?.solve.ode_update_rhs_jacobian_after_steps())
    }
    pub fn set_update_rhs_jacobian_after_steps(&self, value: usize) -> Result<(), DiffsolRtError> {
        self.guard()?
            .solve
            .set_ode_update_rhs_jacobian_after_steps(value);
        Ok(())
    }
    pub fn get_threshold_to_update_jacobian(&self) -> Result<f64, DiffsolRtError> {
        Ok(self.guard()?.solve.ode_threshold_to_update_jacobian())
    }
    pub fn set_threshold_to_update_jacobian(&self, value: f64) -> Result<(), DiffsolRtError> {
        self.guard()?
            .solve
            .set_ode_threshold_to_update_jacobian(value);
        Ok(())
    }
    pub fn get_threshold_to_update_rhs_jacobian(&self) -> Result<f64, DiffsolRtError> {
        Ok(self.guard()?.solve.ode_threshold_to_update_rhs_jacobian())
    }
    pub fn set_threshold_to_update_rhs_jacobian(&self, value: f64) -> Result<(), DiffsolRtError> {
        self.guard()?
            .solve
            .set_ode_threshold_to_update_rhs_jacobian(value);
        Ok(())
    }
    pub fn get_min_timestep(&self) -> Result<f64, DiffsolRtError> {
        Ok(self.guard()?.solve.ode_min_timestep())
    }
    pub fn set_min_timestep(&self, value: f64) -> Result<(), DiffsolRtError> {
        self.guard()?.solve.set_ode_min_timestep(value);
        Ok(())
    }
}

impl Serialize for OdeSolverOptions {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let guard = self.guard().map_err(serde::ser::Error::custom)?;
        OdeSolverOptionsSnapshot::from_solve(guard.solve.as_ref()).serialize(serializer)
    }
}

#[cfg(all(test, any(feature = "diffsl-cranelift", feature = "diffsl-llvm")))]
mod tests {
    use crate::{
        jit::JitBackendType,
        linear_solver_type::LinearSolverType,
        matrix_type::MatrixType,
        ode::OdeWrapper,
        ode_solver_type::OdeSolverType,
        scalar_type::ScalarType,
        test_support::{available_jit_backends, logistic_diffsl_code},
    };

    use super::OdeSolverOptions;

    fn make_options(jit_backend: JitBackendType) -> OdeSolverOptions {
        OdeWrapper::new_jit(
            logistic_diffsl_code(),
            jit_backend,
            ScalarType::F64,
            MatrixType::NalgebraDense,
            LinearSolverType::Default,
            OdeSolverType::Bdf,
        )
        .unwrap()
        .get_options()
    }

    #[test]
    fn ode_solver_options_roundtrip_and_serialize() {
        for jit_backend in available_jit_backends() {
            let options = make_options(jit_backend);
            options.set_max_nonlinear_solver_iterations(17).unwrap();
            options.set_max_error_test_failures(19).unwrap();
            options.set_update_jacobian_after_steps(23).unwrap();
            options.set_update_rhs_jacobian_after_steps(29).unwrap();
            options.set_threshold_to_update_jacobian(1e-3).unwrap();
            options.set_threshold_to_update_rhs_jacobian(2e-3).unwrap();
            options.set_min_timestep(1e-4).unwrap();

            assert_eq!(options.get_max_nonlinear_solver_iterations().unwrap(), 17);
            assert_eq!(options.get_max_error_test_failures().unwrap(), 19);
            assert_eq!(options.get_update_jacobian_after_steps().unwrap(), 23);
            assert_eq!(options.get_update_rhs_jacobian_after_steps().unwrap(), 29);
            assert_eq!(options.get_threshold_to_update_jacobian().unwrap(), 1e-3);
            assert_eq!(
                options.get_threshold_to_update_rhs_jacobian().unwrap(),
                2e-3
            );
            assert_eq!(options.get_min_timestep().unwrap(), 1e-4);

            let value = serde_json::to_value(&options).unwrap();
            assert_eq!(value["max_nonlinear_solver_iterations"], 17);
            assert_eq!(value["max_error_test_failures"], 19);
            assert_eq!(value["update_jacobian_after_steps"], 23);
            assert_eq!(value["update_rhs_jacobian_after_steps"], 29);
            assert_eq!(value["threshold_to_update_jacobian"], 1e-3);
            assert_eq!(value["threshold_to_update_rhs_jacobian"], 2e-3);
            assert_eq!(value["min_timestep"], 1e-4);
        }
    }
}
