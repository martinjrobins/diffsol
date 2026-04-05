// Validation used by py_solve to determine that solver type and matrix type
// combinations are valid.

use diffsol::error::DiffsolError;
use diffsol::{FaerScalar, Scalar};

use crate::{
    error::DiffsolJsError, linear_solver_type::LinearSolverType, matrix_type::MatrixKind,
    matrix_type::MatrixType,
};

pub(crate) fn validate_linear_solver<
    M: diffsol::Matrix + LuValidator<M> + KluValidator<M> + MatrixKind,
>(
    linear_solver: LinearSolverType,
) -> Result<(), DiffsolJsError> {
    match linear_solver {
        LinearSolverType::Default => Ok(()),
        LinearSolverType::Lu => {
            if !<M as LuValidator<M>>::valid() {
                return Err(DiffsolError::Other(format!(
                    "Lu solver not supported for {}",
                    MatrixType::from_diffsol::<M>().get_name()
                ))
                .into());
            }
            Ok(())
        }
        LinearSolverType::Klu => {
            if !<M as KluValidator<M>>::valid() {
                return Err(DiffsolError::Other(format!(
                    "Klu solver not supported for {}",
                    MatrixType::from_diffsol::<M>().get_name()
                ))
                .into());
            }
            Ok(())
        }
    }
}

pub(crate) trait KluValidator<M: diffsol::Matrix> {
    type LS: diffsol::LinearSolver<M>;
    fn valid() -> bool {
        false
    }
}

// Known issue: FaerSparseMat currently only supports KLU for f64
#[cfg(feature = "suitesparse")]
impl KluValidator<diffsol::FaerSparseMat<f64>> for diffsol::FaerSparseMat<f64> {
    type LS = diffsol::KLU<diffsol::FaerSparseMat<f64>>;
    fn valid() -> bool {
        true
    }
}

#[cfg(feature = "suitesparse")]
impl KluValidator<diffsol::FaerSparseMat<f32>> for diffsol::FaerSparseMat<f32> {
    type LS = diffsol::FaerSparseLU<f32>;
    fn valid() -> bool {
        false
    }
}

#[cfg(not(feature = "suitesparse"))]
impl<T: FaerScalar> KluValidator<diffsol::FaerSparseMat<T>> for diffsol::FaerSparseMat<T> {
    type LS = diffsol::FaerSparseLU<T>;
    fn valid() -> bool {
        false
    }
}

impl<T: Scalar> KluValidator<diffsol::NalgebraMat<T>> for diffsol::NalgebraMat<T> {
    type LS = diffsol::NalgebraLU<T>;
    fn valid() -> bool {
        false
    }
}

impl<T: FaerScalar> KluValidator<diffsol::FaerMat<T>> for diffsol::FaerMat<T> {
    type LS = diffsol::FaerLU<T>;
    fn valid() -> bool {
        false
    }
}

pub(crate) trait LuValidator<M: diffsol::Matrix> {
    type LS: diffsol::LinearSolver<M>;
    fn valid() -> bool {
        false
    }
}

impl<T: Scalar> LuValidator<diffsol::NalgebraMat<T>> for diffsol::NalgebraMat<T> {
    type LS = diffsol::NalgebraLU<T>;
    fn valid() -> bool {
        true
    }
}

impl<T: FaerScalar> LuValidator<diffsol::FaerMat<T>> for diffsol::FaerMat<T> {
    type LS = diffsol::FaerLU<T>;
    fn valid() -> bool {
        true
    }
}

impl<T: FaerScalar> LuValidator<diffsol::FaerSparseMat<T>> for diffsol::FaerSparseMat<T> {
    type LS = diffsol::FaerSparseLU<T>;
    fn valid() -> bool {
        true
    }
}
