// Matrix type Python enum

use diffsol::{FaerScalar, Matrix, NalgebraScalar};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Enumerates the possible matrix types for diffsol
///
/// :attr nalgebra_dense: dense matrix using nalgebra crate (https://nalgebra.rs/)
/// :attr faer_dense: dense matrix using faer crate (https://faer.veganb.tw/)
/// :attr faer_sparse: sparse matrix using faer crate (https://faer.veganb.tw/)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MatrixType {
    NalgebraDense,
    FaerDense,
    FaerSparse,
}

// Internal trait to determine runtime MatrixType from a compile-time diffsol matrix type
pub(crate) trait MatrixKind {
    const MATRIX_TYPE: MatrixType;
}

impl<T: NalgebraScalar> MatrixKind for diffsol::NalgebraMat<T> {
    const MATRIX_TYPE: MatrixType = MatrixType::NalgebraDense;
}

impl<T: FaerScalar> MatrixKind for diffsol::FaerMat<T> {
    const MATRIX_TYPE: MatrixType = MatrixType::FaerDense;
}

impl<T: FaerScalar> MatrixKind for diffsol::FaerSparseMat<T> {
    const MATRIX_TYPE: MatrixType = MatrixType::FaerSparse;
}

impl MatrixType {
    pub(crate) fn get_name(&self) -> &str {
        match self {
            MatrixType::NalgebraDense => "nalgebra_dense",
            MatrixType::FaerDense => "faer_dense",
            MatrixType::FaerSparse => "faer_sparse",
        }
    }

    // Determine runtime matrix type compiled diffsol matrix type
    pub(crate) fn from_diffsol<M: Matrix + MatrixKind>() -> Self {
        M::MATRIX_TYPE
    }
}
