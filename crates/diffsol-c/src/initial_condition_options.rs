use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize, Serializer};

use crate::{error::DiffsolRtError, ode::Ode, solve::Solve};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InitialConditionSolverOptionsSnapshot {
    pub use_linesearch: bool,
    pub max_linesearch_iterations: usize,
    pub max_newton_iterations: usize,
    pub max_linear_solver_setups: usize,
    pub step_reduction_factor: f64,
    pub armijo_constant: f64,
}

impl InitialConditionSolverOptionsSnapshot {
    pub(crate) fn from_solve(solve: &dyn Solve) -> Self {
        Self {
            use_linesearch: solve.ic_use_linesearch(),
            max_linesearch_iterations: solve.ic_max_linesearch_iterations(),
            max_newton_iterations: solve.ic_max_newton_iterations(),
            max_linear_solver_setups: solve.ic_max_linear_solver_setups(),
            step_reduction_factor: solve.ic_step_reduction_factor(),
            armijo_constant: solve.ic_armijo_constant(),
        }
    }

    pub(crate) fn apply_to_solve(&self, solve: &mut dyn Solve) {
        solve.set_ic_use_linesearch(self.use_linesearch);
        solve.set_ic_max_linesearch_iterations(self.max_linesearch_iterations);
        solve.set_ic_max_newton_iterations(self.max_newton_iterations);
        solve.set_ic_max_linear_solver_setups(self.max_linear_solver_setups);
        solve.set_ic_step_reduction_factor(self.step_reduction_factor);
        solve.set_ic_armijo_constant(self.armijo_constant);
    }
}

#[derive(Clone)]
pub struct InitialConditionSolverOptions {
    ode: Arc<Mutex<Ode>>,
}
impl InitialConditionSolverOptions {
    pub(crate) fn new(ode: Arc<Mutex<Ode>>) -> Self {
        Self { ode }
    }
    fn guard(&self) -> Result<std::sync::MutexGuard<'_, Ode>, DiffsolRtError> {
        self.ode.lock().map_err(|_| {
            DiffsolRtError::from(diffsol::error::DiffsolError::Other(
                "Failed to acquire lock on ODE solver".to_string(),
            ))
        })
    }
}

impl InitialConditionSolverOptions {
    pub fn get_use_linesearch(&self) -> Result<bool, DiffsolRtError> {
        Ok(self.guard()?.solve.ic_use_linesearch())
    }
    pub fn set_use_linesearch(&self, value: bool) -> Result<(), DiffsolRtError> {
        self.guard()?.solve.set_ic_use_linesearch(value);
        Ok(())
    }
    pub fn get_max_linesearch_iterations(&self) -> Result<usize, DiffsolRtError> {
        Ok(self.guard()?.solve.ic_max_linesearch_iterations())
    }
    pub fn set_max_linesearch_iterations(&self, value: usize) -> Result<(), DiffsolRtError> {
        self.guard()?.solve.set_ic_max_linesearch_iterations(value);
        Ok(())
    }
    pub fn get_max_newton_iterations(&self) -> Result<usize, DiffsolRtError> {
        Ok(self.guard()?.solve.ic_max_newton_iterations())
    }
    pub fn set_max_newton_iterations(&self, value: usize) -> Result<(), DiffsolRtError> {
        self.guard()?.solve.set_ic_max_newton_iterations(value);
        Ok(())
    }
    pub fn get_max_linear_solver_setups(&self) -> Result<usize, DiffsolRtError> {
        Ok(self.guard()?.solve.ic_max_linear_solver_setups())
    }
    pub fn set_max_linear_solver_setups(&self, value: usize) -> Result<(), DiffsolRtError> {
        self.guard()?.solve.set_ic_max_linear_solver_setups(value);
        Ok(())
    }
    pub fn get_step_reduction_factor(&self) -> Result<f64, DiffsolRtError> {
        Ok(self.guard()?.solve.ic_step_reduction_factor())
    }
    pub fn set_step_reduction_factor(&self, value: f64) -> Result<(), DiffsolRtError> {
        self.guard()?.solve.set_ic_step_reduction_factor(value);
        Ok(())
    }
    pub fn get_armijo_constant(&self) -> Result<f64, DiffsolRtError> {
        Ok(self.guard()?.solve.ic_armijo_constant())
    }
    pub fn set_armijo_constant(&self, value: f64) -> Result<(), DiffsolRtError> {
        self.guard()?.solve.set_ic_armijo_constant(value);
        Ok(())
    }
}

impl Serialize for InitialConditionSolverOptions {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let guard = self.guard().map_err(serde::ser::Error::custom)?;
        InitialConditionSolverOptionsSnapshot::from_solve(guard.solve.as_ref())
            .serialize(serializer)
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

    use super::InitialConditionSolverOptions;

    fn make_options(jit_backend: JitBackendType) -> InitialConditionSolverOptions {
        OdeWrapper::new_jit(
            logistic_diffsl_code(),
            jit_backend,
            ScalarType::F64,
            MatrixType::NalgebraDense,
            LinearSolverType::Default,
            OdeSolverType::Bdf,
        )
        .unwrap()
        .get_ic_options()
    }

    #[test]
    fn initial_condition_options_roundtrip_and_serialize() {
        for jit_backend in available_jit_backends() {
            let options = make_options(jit_backend);
            options.set_use_linesearch(true).unwrap();
            options.set_max_linesearch_iterations(13).unwrap();
            options.set_max_newton_iterations(17).unwrap();
            options.set_max_linear_solver_setups(19).unwrap();
            options.set_step_reduction_factor(0.5).unwrap();
            options.set_armijo_constant(1e-4).unwrap();

            assert!(options.get_use_linesearch().unwrap());
            assert_eq!(options.get_max_linesearch_iterations().unwrap(), 13);
            assert_eq!(options.get_max_newton_iterations().unwrap(), 17);
            assert_eq!(options.get_max_linear_solver_setups().unwrap(), 19);
            assert_eq!(options.get_step_reduction_factor().unwrap(), 0.5);
            assert_eq!(options.get_armijo_constant().unwrap(), 1e-4);

            let value = serde_json::to_value(&options).unwrap();
            assert_eq!(value["use_linesearch"], true);
            assert_eq!(value["max_linesearch_iterations"], 13);
            assert_eq!(value["max_newton_iterations"], 17);
            assert_eq!(value["max_linear_solver_setups"], 19);
            assert_eq!(value["step_reduction_factor"], 0.5);
            assert_eq!(value["armijo_constant"], 1e-4);
        }
    }
}
