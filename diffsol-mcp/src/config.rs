#[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
use diffsol_c::{default_enabled_jit_backend, LinearSolverType, OdeSolverType};
use diffsol_c::{JitBackendType, MatrixType, OdeWrapper, ScalarType};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::DiffsolMcpError;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default, PartialEq)]
pub struct OdeOptionsInput {
    pub max_nonlinear_solver_iterations: Option<usize>,
    pub max_error_test_failures: Option<usize>,
    pub update_jacobian_after_steps: Option<usize>,
    pub update_rhs_jacobian_after_steps: Option<usize>,
    pub threshold_to_update_jacobian: Option<f64>,
    pub threshold_to_update_rhs_jacobian: Option<f64>,
    pub min_timestep: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default, PartialEq)]
pub struct InitialConditionOptionsInput {
    pub use_linesearch: Option<bool>,
    pub max_linesearch_iterations: Option<usize>,
    pub max_newton_iterations: Option<usize>,
    pub max_linear_solver_setups: Option<usize>,
    pub step_reduction_factor: Option<f64>,
    pub armijo_constant: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default, PartialEq)]
pub struct ProblemConfigInput {
    pub jit_backend: Option<JitBackendType>,
    pub matrix_type: Option<MatrixType>,
    pub scalar_type: Option<ScalarType>,
    pub rtol: Option<f64>,
    pub atol: Option<f64>,
    pub ode_options: Option<OdeOptionsInput>,
    pub initial_condition_options: Option<InitialConditionOptionsInput>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct ProblemConfig {
    pub jit_backend: JitBackendType,
    pub matrix_type: MatrixType,
    pub scalar_type: ScalarType,
    pub rtol: f64,
    pub atol: f64,
    pub ode_options: Value,
    pub initial_condition_options: Value,
}

impl ProblemConfig {
    pub fn build_ode(
        code: &str,
        input: Option<&ProblemConfigInput>,
    ) -> Result<OdeWrapper, DiffsolMcpError> {
        #[cfg(not(any(feature = "diffsl-cranelift", feature = "diffsl-llvm")))]
        {
            let _ = code;
            let _ = input;
            return Err(DiffsolMcpError::NoJitBackendEnabled);
        }

        #[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
        {
            let input = input.cloned().unwrap_or_default();
            let jit_backend = resolve_jit_backend(input.jit_backend)?;
            let matrix_type = input.matrix_type.unwrap_or(MatrixType::NalgebraDense);
            let scalar_type = input.scalar_type.unwrap_or(ScalarType::F64);

            let ode = OdeWrapper::new_jit(
                code,
                jit_backend.into(),
                scalar_type,
                matrix_type,
                LinearSolverType::Default,
                OdeSolverType::Bdf,
            )?;

            Self::apply_input_to(&ode, &input)?;
            Ok(ode)
        }
    }

    pub fn from_ode(ode: &OdeWrapper) -> Result<Self, DiffsolMcpError> {
        Ok(Self {
            jit_backend: ode.get_jit_backend()?.ok_or_else(|| {
                DiffsolMcpError::invalid_input("problem is not backed by a JIT backend")
            })?,
            matrix_type: ode.get_matrix_type()?,
            scalar_type: ode.get_scalar_type()?,
            rtol: ode.get_rtol()?,
            atol: ode.get_atol()?,
            ode_options: serde_json::to_value(ode.get_options())?,
            initial_condition_options: serde_json::to_value(ode.get_ic_options())?,
        })
    }

    #[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
    fn apply_input_to(ode: &OdeWrapper, input: &ProblemConfigInput) -> Result<(), DiffsolMcpError> {
        if let Some(rtol) = input.rtol {
            ode.set_rtol(rtol)?;
        }
        if let Some(atol) = input.atol {
            ode.set_atol(atol)?;
        }

        let ode_options = ode.get_options();
        if let Some(ode_input) = input.ode_options.as_ref() {
            if let Some(value) = ode_input.max_nonlinear_solver_iterations {
                ode_options.set_max_nonlinear_solver_iterations(value)?;
            }
            if let Some(value) = ode_input.max_error_test_failures {
                ode_options.set_max_error_test_failures(value)?;
            }
            if let Some(value) = ode_input.update_jacobian_after_steps {
                ode_options.set_update_jacobian_after_steps(value)?;
            }
            if let Some(value) = ode_input.update_rhs_jacobian_after_steps {
                ode_options.set_update_rhs_jacobian_after_steps(value)?;
            }
            if let Some(value) = ode_input.threshold_to_update_jacobian {
                ode_options.set_threshold_to_update_jacobian(value)?;
            }
            if let Some(value) = ode_input.threshold_to_update_rhs_jacobian {
                ode_options.set_threshold_to_update_rhs_jacobian(value)?;
            }
            if let Some(value) = ode_input.min_timestep {
                ode_options.set_min_timestep(value)?;
            }
        }

        let ic_options = ode.get_ic_options();
        if let Some(ic_input) = input.initial_condition_options.as_ref() {
            if let Some(value) = ic_input.use_linesearch {
                ic_options.set_use_linesearch(value)?;
            }
            if let Some(value) = ic_input.max_linesearch_iterations {
                ic_options.set_max_linesearch_iterations(value)?;
            }
            if let Some(value) = ic_input.max_newton_iterations {
                ic_options.set_max_newton_iterations(value)?;
            }
            if let Some(value) = ic_input.max_linear_solver_setups {
                ic_options.set_max_linear_solver_setups(value)?;
            }
            if let Some(value) = ic_input.step_reduction_factor {
                ic_options.set_step_reduction_factor(value)?;
            }
            if let Some(value) = ic_input.armijo_constant {
                ic_options.set_armijo_constant(value)?;
            }
        }

        Ok(())
    }
}

#[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
fn resolve_jit_backend(config: Option<JitBackendType>) -> Result<JitBackendType, DiffsolMcpError> {
    if let Some(configured) = config {
        return Ok(configured);
    }

    match default_enabled_jit_backend() {
        Some(backend) => Ok(backend),
        None => {
            if cfg!(all(feature = "diffsl-cranelift", feature = "diffsl-llvm")) {
                Err(DiffsolMcpError::AmbiguousJitBackend)
            } else {
                Err(DiffsolMcpError::NoJitBackendEnabled)
            }
        }
    }
}
