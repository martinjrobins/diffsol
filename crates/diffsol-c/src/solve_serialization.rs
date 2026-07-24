use crate::error::DiffsolRtError;
#[cfg(feature = "external")]
use crate::scalar_type::ExternalScalar;
use crate::scalar_type::Scalar;
use crate::solve::GenericSolve;
use diffsol::ObjectModule;
use diffsol::{
    error::DiffsolError, matrix::MatrixHost, CodegenModule, DefaultDenseMatrix, Vector, VectorHost,
};

pub(crate) fn unsupported_serialization_error(message: &str) -> Result<Vec<u8>, DiffsolRtError> {
    Err(DiffsolError::Other(message.to_string()).into())
}

pub(crate) trait SolveSerialization<M>: CodegenModule
where
    M: MatrixHost<T: Scalar>,
    M::V: Vector + VectorHost + DefaultDenseMatrix,
{
    fn serialized_diffsl(solve: &GenericSolve<M, Self>) -> Result<Vec<u8>, DiffsolRtError>
    where
        Self: Sized;
}

#[cfg(feature = "external")]
impl<M> SolveSerialization<M> for diffsl::ExternalModule<M::T>
where
    M: MatrixHost<T: ExternalScalar>,
    M::V: Vector + VectorHost + DefaultDenseMatrix,
{
    fn serialized_diffsl(_solve: &GenericSolve<M, Self>) -> Result<Vec<u8>, DiffsolRtError> {
        unsupported_serialization_error(
            "ODE serialization is only supported for JIT-backed solvers",
        )
    }
}

#[cfg(feature = "diffsl-external-dynamic")]
impl<M> SolveSerialization<M> for diffsl::ExternalDynModule<M::T>
where
    M: MatrixHost<T: Scalar>,
    M::V: Vector + VectorHost + DefaultDenseMatrix,
{
    fn serialized_diffsl(_solve: &GenericSolve<M, Self>) -> Result<Vec<u8>, DiffsolRtError> {
        unsupported_serialization_error(
            "ODE serialization is only supported for JIT-backed solvers",
        )
    }
}

#[cfg(feature = "diffsl-llvm")]
impl<M> SolveSerialization<M> for diffsol::LlvmModule
where
    M: MatrixHost<T: Scalar>,
    M::V: Vector + VectorHost + DefaultDenseMatrix,
{
    fn serialized_diffsl(solve: &GenericSolve<M, Self>) -> Result<Vec<u8>, DiffsolRtError> {
        solve.serialize_eqn()
    }
}

#[cfg(feature = "diffsl-cranelift")]
impl<M> SolveSerialization<M> for diffsol::CraneliftJitModule
where
    M: MatrixHost<T: Scalar>,
    M::V: Vector + VectorHost + DefaultDenseMatrix,
{
    fn serialized_diffsl(_solve: &GenericSolve<M, Self>) -> Result<Vec<u8>, DiffsolRtError> {
        unsupported_serialization_error(
            "ODE serialization is not supported for Cranelift-backed solvers",
        )
    }
}

impl<M> SolveSerialization<M> for ObjectModule
where
    M: MatrixHost<T: Scalar>,
    M::V: Vector + VectorHost + DefaultDenseMatrix,
{
    fn serialized_diffsl(solve: &GenericSolve<M, Self>) -> Result<Vec<u8>, DiffsolRtError> {
        solve.serialize_eqn()
    }
}
