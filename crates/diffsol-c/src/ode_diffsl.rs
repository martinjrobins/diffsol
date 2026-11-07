//! DiffSL equations: the [OdeWrapper] constructors, the solver factories, and the
//! [SolveEqn] impls for each code generation backend.

#[cfg(feature = "diffsl-external-dynamic")]
use std::path::PathBuf;

use diffsol::{
    error::DiffsolError, matrix::Matrix, DefaultDenseMatrix, DiffSl, ObjectModule, Vector,
};
#[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
use diffsol::{CodegenModule, CodegenModuleCompile, CodegenModuleJit, OdeBuilder};

use crate::error::DiffsolRtError;
#[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
use crate::jit::JitBackendType;
use crate::linear_solver_type::LinearSolverType;
use crate::matrix_type::MatrixType;
use crate::ode::OdeWrapper;
use crate::ode_solver_type::OdeSolverType;
#[cfg(feature = "external")]
use crate::scalar_type::ExternalScalar;
use crate::scalar_type::{Scalar, ScalarType};
use crate::solve::{GenericSolve, Solve};
use crate::solve_eqn::{unsupported, EqnRequirement, SolveEqn};

/// A [`GenericSolve`] over DiffSL equations compiled with the code generation backend `CG`.
pub(crate) type DiffSlSolve<M, CG> = GenericSolve<M, DiffSl<M, CG>>;

impl OdeWrapper {
    /// Construct an ODE solver backed by externally-provided DiffSL symbols.
    #[cfg(feature = "external")]
    pub fn new_external(
        rhs_state_deps: Vec<(usize, usize)>,
        rhs_input_deps: Vec<(usize, usize)>,
        mass_state_deps: Vec<(usize, usize)>,
        scalar_type: ScalarType,
        matrix_type: MatrixType,
        linear_solver: LinearSolverType,
        ode_solver: OdeSolverType,
    ) -> Result<Self, DiffsolRtError> {
        let solve = solve_factory_external(
            rhs_state_deps,
            rhs_input_deps,
            mass_state_deps,
            matrix_type,
            scalar_type,
        )?;
        Self::build(
            String::new(),
            scalar_type,
            solve,
            None,
            linear_solver,
            ode_solver,
        )
    }

    /// Construct an ODE solver backed by DiffSL symbols loaded from a dynamic library.
    #[cfg(feature = "diffsl-external-dynamic")]
    #[allow(clippy::too_many_arguments)]
    pub fn new_external_dynamic(
        path: impl Into<std::path::PathBuf>,
        rhs_state_deps: Vec<(usize, usize)>,
        rhs_input_deps: Vec<(usize, usize)>,
        mass_state_deps: Vec<(usize, usize)>,
        scalar_type: ScalarType,
        matrix_type: MatrixType,
        linear_solver: LinearSolverType,
        ode_solver: OdeSolverType,
    ) -> Result<Self, DiffsolRtError> {
        let solve = solve_factory_external_dynamic(
            path.into(),
            rhs_state_deps,
            rhs_input_deps,
            mass_state_deps,
            matrix_type,
            scalar_type,
        )?;
        Self::build(
            String::new(),
            scalar_type,
            solve,
            None,
            linear_solver,
            ode_solver,
        )
    }

    /// Construct an ODE solver by JIT-compiling DiffSL code immediately.
    #[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
    pub fn new_jit(
        code: &str,
        jit_backend: JitBackendType,
        scalar_type: ScalarType,
        matrix_type: MatrixType,
        linear_solver: LinearSolverType,
        ode_solver: OdeSolverType,
    ) -> Result<Self, DiffsolRtError> {
        let solve = solve_factory_jit(code, jit_backend, matrix_type, scalar_type)?;
        Self::build(
            code.to_owned(),
            scalar_type,
            solve,
            Some(jit_backend),
            linear_solver,
            ode_solver,
        )
    }
}

// Public factory method for generating an instance based on matrix type
#[cfg(feature = "external")]
pub(crate) fn solve_factory_external(
    rhs_state_deps: Vec<(usize, usize)>,
    rhs_input_deps: Vec<(usize, usize)>,
    mass_state_deps: Vec<(usize, usize)>,
    matrix_type: MatrixType,
    scalar_type: ScalarType,
) -> Result<Box<dyn Solve>, DiffsolRtError> {
    // the C ABI only carries the rhs and mass patterns; the out, reset and stop operators' are
    // left empty, which is the same "no non-zeros" contract the other lists already use
    let deps = diffsol::DiffSlDeps {
        rhs_state: rhs_state_deps,
        rhs_input: rhs_input_deps,
        mass_state: mass_state_deps,
        ..Default::default()
    };
    let solve: Box<dyn Solve> = match matrix_type {
        MatrixType::NalgebraDense => match scalar_type {
            #[cfg(feature = "diffsl-external-f32")]
            ScalarType::F32 => Box::new(DiffSlSolve::<
                diffsol::NalgebraMat<f32>,
                diffsl::ExternalModule<f32>,
            >::from_external(deps, false)?),
            #[cfg(feature = "diffsl-external-f64")]
            ScalarType::F64 => Box::new(DiffSlSolve::<
                diffsol::NalgebraMat<f64>,
                diffsl::ExternalModule<f64>,
            >::from_external(deps, false)?),
            _ => {
                return Err(DiffsolRtError::from(DiffsolError::Other(
                    "Unsupported scalar type for NalgebraDense".to_string(),
                )));
            }
        },
        MatrixType::FaerDense => match scalar_type {
            #[cfg(feature = "diffsl-external-f32")]
            ScalarType::F32 => Box::new(DiffSlSolve::<
                diffsol::FaerMat<f32>,
                diffsl::ExternalModule<f32>,
            >::from_external(deps, false)?),
            #[cfg(feature = "diffsl-external-f64")]
            ScalarType::F64 => Box::new(DiffSlSolve::<
                diffsol::FaerMat<f64>,
                diffsl::ExternalModule<f64>,
            >::from_external(deps, false)?),
            _ => {
                return Err(DiffsolRtError::from(DiffsolError::Other(
                    "Unsupported scalar type for FaerDense".to_string(),
                )));
            }
        },
        MatrixType::FaerSparse => match scalar_type {
            #[cfg(feature = "diffsl-external-f32")]
            ScalarType::F32 => Box::new(DiffSlSolve::<
                diffsol::FaerSparseMat<f32>,
                diffsl::ExternalModule<f32>,
            >::from_external(deps, false)?),
            #[cfg(feature = "diffsl-external-f64")]
            ScalarType::F64 => Box::new(DiffSlSolve::<
                diffsol::FaerSparseMat<f64>,
                diffsl::ExternalModule<f64>,
            >::from_external(deps, false)?),
            _ => {
                return Err(DiffsolRtError::from(DiffsolError::Other(
                    "Unsupported scalar type for FaerSparse".to_string(),
                )));
            }
        },
    };
    Ok(solve)
}

#[cfg(feature = "diffsl-external-dynamic")]
pub(crate) fn solve_factory_external_dynamic(
    path: PathBuf,
    rhs_state_deps: Vec<(usize, usize)>,
    rhs_input_deps: Vec<(usize, usize)>,
    mass_state_deps: Vec<(usize, usize)>,
    matrix_type: MatrixType,
    scalar_type: ScalarType,
) -> Result<Box<dyn Solve>, DiffsolRtError> {
    // see `solve_factory_external`: the C ABI carries only the rhs and mass patterns
    let deps = diffsol::DiffSlDeps {
        rhs_state: rhs_state_deps,
        rhs_input: rhs_input_deps,
        mass_state: mass_state_deps,
        ..Default::default()
    };
    let solve: Box<dyn Solve> = match matrix_type {
        MatrixType::NalgebraDense => match scalar_type {
            ScalarType::F32 => Box::new(DiffSlSolve::<
                diffsol::NalgebraMat<f32>,
                diffsl::ExternalDynModule<f32>,
            >::from_external_dynamic(path, deps, false)?),
            ScalarType::F64 => Box::new(DiffSlSolve::<
                diffsol::NalgebraMat<f64>,
                diffsl::ExternalDynModule<f64>,
            >::from_external_dynamic(path, deps, false)?),
        },
        MatrixType::FaerDense => match scalar_type {
            ScalarType::F32 => Box::new(DiffSlSolve::<
                diffsol::FaerMat<f32>,
                diffsl::ExternalDynModule<f32>,
            >::from_external_dynamic(path, deps, false)?),
            ScalarType::F64 => Box::new(DiffSlSolve::<
                diffsol::FaerMat<f64>,
                diffsl::ExternalDynModule<f64>,
            >::from_external_dynamic(path, deps, false)?),
        },
        MatrixType::FaerSparse => match scalar_type {
            ScalarType::F32 => Box::new(DiffSlSolve::<
                diffsol::FaerSparseMat<f32>,
                diffsl::ExternalDynModule<f32>,
            >::from_external_dynamic(path, deps, false)?),
            ScalarType::F64 => Box::new(DiffSlSolve::<
                diffsol::FaerSparseMat<f64>,
                diffsl::ExternalDynModule<f64>,
            >::from_external_dynamic(path, deps, false)?),
        },
    };
    Ok(solve)
}

#[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
pub(crate) fn solve_factory_jit(
    code: &str,
    jit_backend: JitBackendType,
    matrix_type: MatrixType,
    scalar_type: ScalarType,
) -> Result<Box<dyn Solve>, DiffsolRtError> {
    match jit_backend {
        #[cfg(feature = "diffsl-cranelift")]
        JitBackendType::Cranelift => solve_factory_with_jit_backend::<diffsol::CraneliftJitModule>(
            code,
            matrix_type,
            scalar_type,
        ),
        #[cfg(feature = "diffsl-llvm")]
        JitBackendType::Llvm => {
            solve_factory_with_jit_backend::<diffsol::LlvmModule>(code, matrix_type, scalar_type)
        }
    }
}

#[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
fn solve_factory_with_jit_backend<CG>(
    code: &str,
    matrix_type: MatrixType,
    scalar_type: ScalarType,
) -> Result<Box<dyn Solve>, DiffsolRtError>
where
    CG: CodegenModule + CodegenModuleJit + CodegenModuleCompile,
    DiffSl<diffsol::NalgebraMat<f32>, CG>: SolveEqn,
    DiffSl<diffsol::NalgebraMat<f64>, CG>: SolveEqn,
    DiffSl<diffsol::FaerMat<f32>, CG>: SolveEqn,
    DiffSl<diffsol::FaerMat<f64>, CG>: SolveEqn,
    DiffSl<diffsol::FaerSparseMat<f32>, CG>: SolveEqn,
    DiffSl<diffsol::FaerSparseMat<f64>, CG>: SolveEqn,
{
    let solve: Box<dyn Solve> = match matrix_type {
        MatrixType::NalgebraDense => match scalar_type {
            ScalarType::F32 => {
                let problem =
                    OdeBuilder::<diffsol::NalgebraMat<f32>>::new().build_from_diffsl::<CG>(code)?;
                Box::new(GenericSolve { problem })
            }
            ScalarType::F64 => {
                let problem =
                    OdeBuilder::<diffsol::NalgebraMat<f64>>::new().build_from_diffsl::<CG>(code)?;
                Box::new(GenericSolve { problem })
            }
        },
        MatrixType::FaerDense => match scalar_type {
            ScalarType::F32 => {
                let problem =
                    OdeBuilder::<diffsol::FaerMat<f32>>::new().build_from_diffsl::<CG>(code)?;
                Box::new(GenericSolve { problem })
            }
            ScalarType::F64 => {
                let problem =
                    OdeBuilder::<diffsol::FaerMat<f64>>::new().build_from_diffsl::<CG>(code)?;
                Box::new(GenericSolve { problem })
            }
        },
        MatrixType::FaerSparse => match scalar_type {
            ScalarType::F32 => {
                let problem = OdeBuilder::<diffsol::FaerSparseMat<f32>>::new()
                    .build_from_diffsl::<CG>(code)?;
                Box::new(GenericSolve { problem })
            }
            ScalarType::F64 => {
                let problem = OdeBuilder::<diffsol::FaerSparseMat<f64>>::new()
                    .build_from_diffsl::<CG>(code)?;
                Box::new(GenericSolve { problem })
            }
        },
    };
    Ok(solve)
}

pub(crate) fn solve_factory_from_serialized_diffsl(
    serialized_diffsl: &[u8],
    matrix_type: MatrixType,
    scalar_type: ScalarType,
) -> Result<Box<dyn Solve>, DiffsolRtError> {
    let solve: Box<dyn Solve> = match matrix_type {
        MatrixType::NalgebraDense => match scalar_type {
            ScalarType::F32 => Box::new(
                DiffSlSolve::<diffsol::NalgebraMat<f32>, ObjectModule>::from_serialized_diffsl(
                    serialized_diffsl,
                )?,
            ),
            ScalarType::F64 => Box::new(
                DiffSlSolve::<diffsol::NalgebraMat<f64>, ObjectModule>::from_serialized_diffsl(
                    serialized_diffsl,
                )?,
            ),
        },
        MatrixType::FaerDense => match scalar_type {
            ScalarType::F32 => Box::new(
                DiffSlSolve::<diffsol::FaerMat<f32>, ObjectModule>::from_serialized_diffsl(
                    serialized_diffsl,
                )?,
            ),
            ScalarType::F64 => Box::new(
                DiffSlSolve::<diffsol::FaerMat<f64>, ObjectModule>::from_serialized_diffsl(
                    serialized_diffsl,
                )?,
            ),
        },
        MatrixType::FaerSparse => match scalar_type {
            ScalarType::F32 => Box::new(
                DiffSlSolve::<diffsol::FaerSparseMat<f32>, ObjectModule>::from_serialized_diffsl(
                    serialized_diffsl,
                )?,
            ),
            ScalarType::F64 => Box::new(
                DiffSlSolve::<diffsol::FaerSparseMat<f64>, ObjectModule>::from_serialized_diffsl(
                    serialized_diffsl,
                )?,
            ),
        },
    };
    Ok(solve)
}

#[cfg(feature = "external")]
impl<M> DiffSlSolve<M, diffsl::ExternalModule<M::T>>
where
    M: Matrix<T: ExternalScalar>,
    M::V: Vector + DefaultDenseMatrix,
{
    pub fn from_external(
        deps: diffsol::DiffSlDeps,
        include_sensitivities: bool,
    ) -> Result<Self, DiffsolRtError> {
        let eqn = DiffSl::<M, diffsl::ExternalModule<M::T>>::from_external(
            M::C::default(),
            deps,
            include_sensitivities,
        )?;
        Self::from_eqn(eqn)
    }
}

#[cfg(feature = "diffsl-external-dynamic")]
impl<M> DiffSlSolve<M, diffsl::ExternalDynModule<M::T>>
where
    M: Matrix<T: Scalar>,
    M::V: Vector + DefaultDenseMatrix,
{
    pub fn from_external_dynamic(
        path: impl Into<PathBuf>,
        deps: diffsol::DiffSlDeps,
        include_sensitivities: bool,
    ) -> Result<Self, DiffsolRtError> {
        let eqn = DiffSl::<M, diffsl::ExternalDynModule<M::T>>::from_external_dynamic(
            path,
            M::C::default(),
            deps,
            include_sensitivities,
        )?;
        Self::from_eqn(eqn)
    }
}

impl<M> DiffSlSolve<M, ObjectModule>
where
    M: Matrix<T: Scalar>,
    M::V: Vector + DefaultDenseMatrix,
{
    pub fn from_serialized_diffsl(serialized_diffsl: &[u8]) -> Result<Self, DiffsolRtError> {
        let eqn = serde_json::from_slice::<DiffSl<M, ObjectModule>>(serialized_diffsl)
            .map_err(|e| DiffsolRtError::from(DiffsolError::Other(e.to_string())))?;
        Self::from_eqn(eqn)
    }
}

fn serialize_json<T: serde::Serialize>(eqn: &T) -> Result<Vec<u8>, DiffsolRtError> {
    serde_json::to_vec(eqn).map_err(|e| DiffsolRtError::from(DiffsolError::Other(e.to_string())))
}

/// The requirement check of DiffSL equations, which is the same for each backend.
fn check_diffsl_requirement(requirement: EqnRequirement) -> Result<(), DiffsolRtError> {
    // sensitivity analysis of DiffSL equations is not supported on Windows
    let needs_sens = matches!(
        requirement,
        EqnRequirement::FwdSens | EqnRequirement::Adjoint(_)
    );
    if needs_sens && cfg!(target_os = "windows") {
        return Err(DiffsolError::Other(
            "Sensitivity analysis is not supported on Windows, please use a linux or macOS system."
                .to_string(),
        )
        .into());
    }
    Ok(())
}

/// Implement [`SolveEqn`] for `DiffSl<M, $cg>`, where the scalar type of `M` satisfies
/// `$scalar`. The serialization depends on the backend `$cg`, so each backend gives its own
/// `serialized` body, with `$eqn` the equations.
macro_rules! impl_solve_eqn_for_diffsl {
    ($cg:ty, $scalar:path, |$eqn:ident| $serialized:expr) => {
        impl<M> SolveEqn for DiffSl<M, $cg>
        where
            M: Matrix<T: $scalar>,
            M::V: Vector + DefaultDenseMatrix,
        {
            fn serialized(&self) -> Result<Vec<u8>, DiffsolRtError> {
                let $eqn = self;
                $serialized
            }

            fn check_requirement(&self, requirement: EqnRequirement) -> Result<(), DiffsolRtError> {
                check_diffsl_requirement(requirement)
            }
        }
    };
}

#[cfg(feature = "diffsl-llvm")]
impl_solve_eqn_for_diffsl!(diffsol::LlvmModule, Scalar, |eqn| serialize_json(eqn));

impl_solve_eqn_for_diffsl!(diffsol::ObjectModule, Scalar, |eqn| serialize_json(eqn));

#[cfg(feature = "diffsl-cranelift")]
impl_solve_eqn_for_diffsl!(diffsol::CraneliftJitModule, Scalar, |_eqn| unsupported(
    "ODE serialization is not supported for Cranelift-backed solvers"
));

#[cfg(feature = "external")]
impl_solve_eqn_for_diffsl!(diffsl::ExternalModule<M::T>, ExternalScalar, |_eqn| {
    unsupported("ODE serialization is only supported for JIT-backed solvers")
});

#[cfg(feature = "diffsl-external-dynamic")]
impl_solve_eqn_for_diffsl!(diffsl::ExternalDynModule<M::T>, Scalar, |_eqn| unsupported(
    "ODE serialization is only supported for JIT-backed solvers"
));

#[cfg(all(test, any(feature = "diffsl-cranelift", feature = "diffsl-llvm")))]
mod tests {
    use diffsol::MatrixCommon;
    use diffsol::{
        CodegenModuleCompile, CodegenModuleJit, Context, OdeBuilder, OdeEquations, Vector,
    };

    #[cfg(feature = "diffsl-llvm")]
    use crate::test_support::{hybrid_logistic_state_dr, logistic_state_dr};
    use crate::{
        host_array::FromHostArray,
        linear_solver_type::LinearSolverType,
        matrix_type::MatrixType,
        ode_solver_type::OdeSolverType,
        scalar_type::ScalarType,
        test_support::{
            assert_close, hybrid_logistic_diffsl_code, hybrid_logistic_state, logistic_diffsl_code,
            logistic_state, two_param_logistic_diffsl_code, LOGISTIC_X0,
        },
    };

    use super::{solve_factory_with_jit_backend, DiffSlSolve};
    use crate::solve::{GenericSolve, Solve};
    use crate::solve_eqn::SolveEqn;

    fn make_generic_solve<T, CG>() -> DiffSlSolve<diffsol::NalgebraMat<T>, CG>
    where
        T: crate::scalar_type::Scalar + diffsol::NalgebraScalar,
        CG: diffsol::CodegenModule + CodegenModuleJit + CodegenModuleCompile,
        diffsol::DiffSl<diffsol::NalgebraMat<T>, CG>: SolveEqn,
        diffsol::NalgebraMat<T>: diffsol::matrix::Matrix,
        <diffsol::NalgebraMat<T> as MatrixCommon>::T: crate::scalar_type::Scalar,
        <diffsol::NalgebraMat<T> as MatrixCommon>::V: diffsol::DefaultDenseMatrix,
    {
        let problem = OdeBuilder::<diffsol::NalgebraMat<T>>::new()
            .build_from_diffsl::<CG>(logistic_diffsl_code())
            .unwrap();
        GenericSolve { problem }
    }

    fn assert_factory_supports_all_matrix_and_scalar_types<CG>()
    where
        CG: diffsol::CodegenModule + CodegenModuleJit + CodegenModuleCompile,
        diffsol::DiffSl<diffsol::NalgebraMat<f32>, CG>: SolveEqn,
        diffsol::DiffSl<diffsol::NalgebraMat<f64>, CG>: SolveEqn,
        diffsol::DiffSl<diffsol::FaerMat<f32>, CG>: SolveEqn,
        diffsol::DiffSl<diffsol::FaerMat<f64>, CG>: SolveEqn,
        diffsol::DiffSl<diffsol::FaerSparseMat<f32>, CG>: SolveEqn,
        diffsol::DiffSl<diffsol::FaerSparseMat<f64>, CG>: SolveEqn,
    {
        for matrix_type in [
            MatrixType::NalgebraDense,
            MatrixType::FaerDense,
            MatrixType::FaerSparse,
        ] {
            for scalar_type in [ScalarType::F32, ScalarType::F64] {
                assert!(solve_factory_with_jit_backend::<CG>(
                    logistic_diffsl_code(),
                    matrix_type,
                    scalar_type,
                )
                .is_ok());
            }
        }
    }

    fn assert_solve_metadata_and_helpers<T, CG>(tol: f64)
    where
        T: crate::scalar_type::Scalar + diffsol::NalgebraScalar,
        CG: diffsol::CodegenModule + CodegenModuleJit + CodegenModuleCompile,
        diffsol::DiffSl<diffsol::NalgebraMat<T>, CG>: SolveEqn,
        diffsol::NalgebraMat<T>: diffsol::matrix::Matrix,
        <diffsol::NalgebraMat<T> as MatrixCommon>::T: crate::scalar_type::Scalar,
        <diffsol::NalgebraMat<T> as MatrixCommon>::V: diffsol::DefaultDenseMatrix,
        DiffSlSolve<diffsol::NalgebraMat<T>, CG>: Solve,
    {
        let mut solve = make_generic_solve::<T, CG>();
        assert_eq!(solve.matrix_type(), MatrixType::NalgebraDense);
        assert_eq!(solve.nstates(), 1);
        assert_eq!(solve.nparams(), 1);
        assert_eq!(solve.nout(), 1);
        assert!(!solve.has_stop());
        assert!(solve.check(LinearSolverType::Default).is_ok());
        assert!(solve.check(LinearSolverType::Lu).is_ok());
        assert!(solve.check(LinearSolverType::Klu).is_err());

        solve.set_atol(1e-5);
        solve.set_rtol(1e-4);
        assert_close(solve.atol(), 1e-5, tol, "solve atol");
        assert_close(solve.rtol(), 1e-4, tol, "solve rtol");
        solve.set_t0(0.125);
        solve.set_h0(0.25);
        solve.set_integrate_out(true);
        assert_close(solve.t0(), 0.125, tol, "solve t0");
        assert_close(solve.h0(), 0.25, tol, "solve h0");
        assert!(solve.integrate_out());

        solve.set_sens_rtol(Some(1e-3));
        solve.set_sens_atol(Some(1e-4));
        solve.set_out_rtol(Some(2e-3));
        solve.set_out_atol(Some(2e-4));
        solve.set_param_rtol(Some(3e-3));
        solve.set_param_atol(Some(3e-4));
        assert_close(solve.sens_rtol().unwrap(), 1e-3, tol, "solve sens rtol");
        assert_close(solve.sens_atol().unwrap(), 1e-4, tol, "solve sens atol");
        assert_close(solve.out_rtol().unwrap(), 2e-3, tol, "solve out rtol");
        assert_close(solve.out_atol().unwrap(), 2e-4, tol, "solve out atol");
        assert_close(solve.param_rtol().unwrap(), 3e-3, tol, "solve param rtol");
        assert_close(solve.param_atol().unwrap(), 3e-4, tol, "solve param atol");

        solve.set_sens_rtol(None);
        solve.set_sens_atol(None);
        solve.set_out_rtol(None);
        solve.set_out_atol(None);
        solve.set_param_rtol(None);
        solve.set_param_atol(None);
        assert_eq!(solve.sens_rtol(), None);
        assert_eq!(solve.sens_atol(), None);
        assert_eq!(solve.out_rtol(), None);
        assert_eq!(solve.out_atol(), None);
        assert_eq!(solve.param_rtol(), None);
        assert_eq!(solve.param_atol(), None);

        let y0 = Vec::<f64>::from_host_array(solve.y0(&[2.0]).unwrap()).unwrap();
        assert_eq!(y0.len(), 1);
        assert_close(y0[0], LOGISTIC_X0, tol, "y0[0]");

        let rhs = Vec::<f64>::from_host_array(solve.rhs(&[2.0], 0.0, &[0.25]).unwrap()).unwrap();
        assert_eq!(rhs.len(), 1);
        assert_close(rhs[0], 0.375, tol, "solve rhs");

        let jac_mul =
            Vec::<f64>::from_host_array(solve.rhs_jac_mul(&[2.0], 0.0, &[0.25], &[3.0]).unwrap())
                .unwrap();
        assert_eq!(jac_mul.len(), 1);
        assert_close(jac_mul[0], 3.0, tol, "solve rhs jac mul");
    }

    fn assert_solve_runtime_paths<T, CG>(tol: f64)
    where
        T: crate::scalar_type::Scalar + diffsol::NalgebraScalar,
        CG: diffsol::CodegenModule + CodegenModuleJit + CodegenModuleCompile,
        diffsol::DiffSl<diffsol::NalgebraMat<T>, CG>: SolveEqn,
        diffsol::NalgebraMat<T>: diffsol::matrix::Matrix,
        <diffsol::NalgebraMat<T> as MatrixCommon>::T: crate::scalar_type::Scalar,
        <diffsol::NalgebraMat<T> as MatrixCommon>::V: diffsol::DefaultDenseMatrix,
        DiffSlSolve<diffsol::NalgebraMat<T>, CG>: Solve,
    {
        let mut solve = make_generic_solve::<T, CG>();
        let soln = solve
            .solve(OdeSolverType::Bdf, LinearSolverType::Lu, &[2.0], 1.0)
            .unwrap();
        let ts = Vec::<f64>::from_host_array(soln.get_ts()).unwrap();
        let ys = Vec::<Vec<f64>>::from_host_array(soln.get_ys()).unwrap();
        assert_close(*ts.last().unwrap(), 1.0, tol, "solve final time");
        assert_close(
            ys[0][ts.len() - 1],
            logistic_state(LOGISTIC_X0, 2.0, 1.0),
            tol,
            "solve final value",
        );

        let mut solve = make_generic_solve::<T, CG>();
        let dense = solve
            .solve_dense(
                OdeSolverType::Tsit45,
                LinearSolverType::Lu,
                &[2.0],
                &[0.25, 0.5, 1.0],
            )
            .unwrap();
        let ts = Vec::<f64>::from_host_array(dense.get_ts()).unwrap();
        let ys = Vec::<Vec<f64>>::from_host_array(dense.get_ys()).unwrap();
        assert_eq!(ts, vec![0.25, 0.5, 1.0]);
        for (i, &t) in ts.iter().enumerate() {
            assert_close(
                ys[0][i],
                logistic_state(LOGISTIC_X0, 2.0, t),
                tol,
                &format!("solve_dense[{i}]"),
            );
        }

        let mut solve = make_generic_solve::<T, CG>();
        let err = match solve.solve(OdeSolverType::Bdf, LinearSolverType::Default, &[], 1.0) {
            Ok(_) => panic!("expected parameter count mismatch"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("Expecting 1 params but got 0"));

        let hybrid_problem = OdeBuilder::<diffsol::NalgebraMat<T>>::new()
            .build_from_diffsl::<CG>(hybrid_logistic_diffsl_code())
            .unwrap();
        let mut hybrid_solve = GenericSolve {
            problem: hybrid_problem,
        };
        let hybrid = hybrid_solve
            .solve(OdeSolverType::Bdf, LinearSolverType::Lu, &[2.0], 2.0)
            .unwrap();
        let hybrid_ts = Vec::<f64>::from_host_array(hybrid.get_ts()).unwrap();
        let hybrid_ys = Vec::<Vec<f64>>::from_host_array(hybrid.get_ys()).unwrap();
        assert_close(*hybrid_ts.last().unwrap(), 2.0, tol, "solve final time");
        assert_close(
            hybrid_ys[0][hybrid_ts.len() - 1],
            hybrid_logistic_state(2.0, 2.0),
            tol,
            "solve final value",
        );

        let hybrid_problem = OdeBuilder::<diffsol::NalgebraMat<T>>::new()
            .build_from_diffsl::<CG>(hybrid_logistic_diffsl_code())
            .unwrap();
        let mut hybrid_solve = GenericSolve {
            problem: hybrid_problem,
        };
        let hybrid_dense = hybrid_solve
            .solve_dense(
                OdeSolverType::Tsit45,
                LinearSolverType::Lu,
                &[2.0],
                &[0.5, 1.0, 1.5, 2.0],
            )
            .unwrap();
        let hybrid_dense_ts = Vec::<f64>::from_host_array(hybrid_dense.get_ts()).unwrap();
        let hybrid_dense_ys = Vec::<Vec<f64>>::from_host_array(hybrid_dense.get_ys()).unwrap();
        assert_eq!(hybrid_dense_ts, vec![0.5, 1.0, 1.5, 2.0]);
        for (i, &t) in hybrid_dense_ts.iter().enumerate() {
            assert_close(
                hybrid_dense_ys[0][i],
                hybrid_logistic_state(2.0, t),
                tol,
                &format!("solve_dense[{i}]"),
            );
        }
    }

    #[cfg(feature = "diffsl-llvm")]
    fn assert_solve_sensitivity_paths() {
        let t_eval = [0.25, 0.5, 1.0];

        let mut solve = make_generic_solve::<f64, diffsol::LlvmModule>();
        let sens = solve
            .solve_fwd_sens(OdeSolverType::Bdf, LinearSolverType::Lu, &[2.0], &t_eval)
            .unwrap();
        let sens_values = sens.get_sens();
        assert_eq!(sens_values.len(), 1);
        let sens_matrix =
            Vec::<Vec<f64>>::from_host_array(sens_values.into_iter().next().unwrap()).unwrap();
        for (i, &t) in t_eval.iter().enumerate() {
            assert_close(
                sens_matrix[0][i],
                logistic_state_dr(LOGISTIC_X0, 2.0, t),
                5e-4,
                &format!("solve_fwd_sens[{i}]"),
            );
        }

        let hybrid_problem = OdeBuilder::<diffsol::NalgebraMat<f64>>::new()
            .build_from_diffsl::<diffsol::LlvmModule>(hybrid_logistic_diffsl_code())
            .unwrap();
        let mut solve = GenericSolve {
            problem: hybrid_problem,
        };
        let hybrid_sens = solve
            .solve_fwd_sens(OdeSolverType::Bdf, LinearSolverType::Lu, &[2.0], &t_eval)
            .unwrap();
        let sens_values = hybrid_sens.get_sens();
        let sens_matrix =
            Vec::<Vec<f64>>::from_host_array(sens_values.into_iter().next().unwrap()).unwrap();
        for (i, &t) in t_eval.iter().enumerate() {
            assert_close(
                sens_matrix[0][i],
                hybrid_logistic_state_dr(2.0, t),
                5e-4,
                &format!("solve_fwd_sens[{i}]"),
            );
        }
    }

    #[cfg(feature = "diffsl-cranelift")]
    fn assert_sens_atol_round_trips_with_multiple_params() {
        // two params make sens_atol batched (nbatch = 2), which get_index rejects
        let problem = OdeBuilder::<diffsol::NalgebraMat<f64>>::new()
            .build_from_diffsl::<diffsol::CraneliftJitModule>(two_param_logistic_diffsl_code())
            .unwrap();
        let mut solve = GenericSolve { problem };
        assert_eq!(solve.nparams(), 2);

        solve.set_sens_atol(Some(1e-4));
        let sens_atol = solve.problem.sens_atol.as_ref().unwrap();
        assert_eq!(sens_atol.context().nbatch(), 2);
        assert_eq!(sens_atol.len(), solve.nstates());
        assert_close(solve.sens_atol().unwrap(), 1e-4, 1e-12, "sens atol");
    }

    #[cfg(feature = "diffsl-cranelift")]
    #[test]
    fn solve_factory_supports_all_jit_matrix_and_scalar_types_for_cranelift() {
        assert_factory_supports_all_matrix_and_scalar_types::<diffsol::CraneliftJitModule>();
    }

    #[cfg(feature = "diffsl-cranelift")]
    #[test]
    fn solve_trait_helpers_and_runtime_paths_for_cranelift() {
        assert_solve_metadata_and_helpers::<f64, diffsol::CraneliftJitModule>(1e-12);
        assert_solve_runtime_paths::<f64, diffsol::CraneliftJitModule>(5e-4);
        assert_solve_metadata_and_helpers::<f32, diffsol::CraneliftJitModule>(5e-3);
        assert_solve_runtime_paths::<f32, diffsol::CraneliftJitModule>(5e-3);
    }

    #[cfg(feature = "diffsl-cranelift")]
    #[test]
    fn sens_atol_round_trips_with_multiple_params_for_cranelift() {
        assert_sens_atol_round_trips_with_multiple_params();
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn solve_trait_helpers_and_runtime_paths_for_llvm() {
        assert_solve_metadata_and_helpers::<f64, diffsol::LlvmModule>(1e-12);
        assert_solve_runtime_paths::<f64, diffsol::LlvmModule>(5e-4);
        assert_solve_metadata_and_helpers::<f32, diffsol::LlvmModule>(5e-3);
        assert_solve_runtime_paths::<f32, diffsol::LlvmModule>(5e-3);
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn solve_trait_sensitivity_paths_for_llvm() {
        assert_solve_sensitivity_paths();
    }

    #[cfg(feature = "diffsl-cranelift")]
    #[test]
    fn setup_problem_validates_parameter_count_for_cranelift() {
        let mut solve = make_generic_solve::<f64, diffsol::CraneliftJitModule>();
        let err = solve.setup_problem(&[]).unwrap_err();
        assert!(err.to_string().contains("Expecting 1 params but got 0"));

        solve.setup_problem(&[2.0]).unwrap();
        let mut params = solve.problem.context().vector_zeros(1);
        solve.problem.eqn.get_params(&mut params);
        assert_eq!(params.get_index(0), 2.0);
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn setup_problem_validates_parameter_count_for_llvm() {
        let mut solve = make_generic_solve::<f64, diffsol::LlvmModule>();
        let err = solve.setup_problem(&[]).unwrap_err();
        assert!(err.to_string().contains("Expecting 1 params but got 0"));

        solve.setup_problem(&[2.0]).unwrap();
        let mut params = solve.problem.context().vector_zeros(1);
        solve.problem.eqn.get_params(&mut params);
        assert_eq!(params.get_index(0), 2.0);
    }
}
