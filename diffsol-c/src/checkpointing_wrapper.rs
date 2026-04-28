use std::{
    any::Any,
    sync::{Arc, Mutex, MutexGuard},
};

use diffsol::{
    error::DiffsolError, CheckpointingPath, CodegenModule, DefaultDenseMatrix, DiffSl, Matrix,
    OdeEquations, OdeSolverMethod, OdeSolverState, Op, VectorHost,
};

use crate::{
    error::DiffsolRtError,
    linear_solver_type::LinearSolverType,
    matrix_type::{MatrixKind, MatrixType},
    ode_solver_type::OdeSolverType,
    scalar_type::{Scalar, ScalarType, ToScalarType},
};

pub(crate) trait CheckpointingData: Any {
    fn as_any(&self) -> &dyn Any;
    fn forward_method(&self) -> OdeSolverType;
    fn forward_linear_solver(&self) -> LinearSolverType;
    fn params(&self) -> &[f64];
}

pub(crate) struct TypedCheckpointing<M, CG, State>
where
    M: Matrix<T: Scalar>,
    CG: diffsol::CodegenModule,
    DiffSl<M, CG>: OdeEquations,
    State: OdeSolverState<<DiffSl<M, CG> as Op>::V>,
{
    pub(crate) checkpointing: CheckpointingPath<DiffSl<M, CG>, State>,
    pub(crate) params: Vec<f64>,
    pub(crate) matrix_type: MatrixType,
    pub(crate) scalar_type: ScalarType,
    pub(crate) forward_method: OdeSolverType,
    pub(crate) forward_linear_solver: LinearSolverType,
}

impl<M, CG, State> TypedCheckpointing<M, CG, State>
where
    M: Matrix<T: Scalar>,
    CG: diffsol::CodegenModule,
    DiffSl<M, CG>: OdeEquations,
    State: OdeSolverState<<DiffSl<M, CG> as Op>::V>,
{
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        checkpointing: CheckpointingPath<DiffSl<M, CG>, State>,
        params: Vec<f64>,
        matrix_type: MatrixType,
        scalar_type: ScalarType,
        forward_method: OdeSolverType,
        forward_linear_solver: LinearSolverType,
    ) -> Self {
        Self {
            checkpointing,
            params,
            matrix_type,
            scalar_type,
            forward_method,
            forward_linear_solver,
        }
    }
}

impl<M, CG, State> CheckpointingData for TypedCheckpointing<M, CG, State>
where
    M: Matrix<T: Scalar> + 'static,
    CG: diffsol::CodegenModule + 'static,
    DiffSl<M, CG>: OdeEquations,
    State: OdeSolverState<<DiffSl<M, CG> as Op>::V> + 'static,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn forward_method(&self) -> OdeSolverType {
        self.forward_method
    }

    fn forward_linear_solver(&self) -> LinearSolverType {
        self.forward_linear_solver
    }

    fn params(&self) -> &[f64] {
        &self.params
    }
}

#[derive(Clone)]
pub struct CheckpointingWrapper(Arc<Mutex<Box<dyn CheckpointingData>>>);

impl CheckpointingWrapper {
    pub(crate) fn new(checkpointing: Box<dyn CheckpointingData>) -> Self {
        Self(Arc::new(Mutex::new(checkpointing)))
    }

    pub(crate) fn wrap_checkpointing<M, CG, State>(
        checkpointing: CheckpointingPath<DiffSl<M, CG>, State>,
        params: &[f64],
        method: OdeSolverType,
        linear_solver: LinearSolverType,
    ) -> Self
    where
        M: Matrix<T: Scalar> + MatrixKind + 'static,
        CG: CodegenModule + 'static,
        M::V: VectorHost + DefaultDenseMatrix + Send + Sync + 'static,
        State: OdeSolverState<M::V> + 'static,
    {
        Self::new(Box::new(TypedCheckpointing::<M, CG, State>::new(
            checkpointing,
            params.to_vec(),
            MatrixType::from_diffsol::<M>(),
            M::T::scalar_type(),
            method,
            linear_solver,
        )))
    }

    fn guard(&self) -> Result<MutexGuard<'_, Box<dyn CheckpointingData>>, DiffsolRtError> {
        self.0
            .lock()
            .map_err(|_| DiffsolError::Other("Checkpointing mutex poisoned".to_string()).into())
    }

    pub(crate) fn forward_settings(
        &self,
    ) -> Result<(OdeSolverType, LinearSolverType), DiffsolRtError> {
        let guard = self.guard()?;
        Ok((guard.forward_method(), guard.forward_linear_solver()))
    }

    pub(crate) fn params(&self) -> Result<Vec<f64>, DiffsolRtError> {
        let guard = self.guard()?;
        Ok(guard.params().to_vec())
    }

    #[allow(dead_code)]
    pub(crate) fn checkpointing_for_solver<'solver, M, CG, S>(
        &self,
        _solver: &S,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
    ) -> Result<CheckpointingPath<DiffSl<M, CG>, S::State>, DiffsolError>
    where
        M: Matrix<T: Scalar> + MatrixKind + 'static,
        CG: CodegenModule + 'static,
        M::V: VectorHost + DefaultDenseMatrix + Send + Sync + 'static,
        S: OdeSolverMethod<'solver, DiffSl<M, CG>>,
        S::State: Clone + 'static,
    {
        let (_, checkpointing) = self
            .clone_typed::<M, CG, S::State>(
                MatrixType::from_diffsol::<M>(),
                M::T::scalar_type(),
                method,
                linear_solver,
            )
            .map_err(|err| DiffsolError::Other(err.to_string()))?;
        Ok(checkpointing)
    }

    pub(crate) fn clone_typed<M, CG, State>(
        &self,
        expected_matrix_type: MatrixType,
        expected_scalar_type: ScalarType,
        expected_forward_method: OdeSolverType,
        expected_forward_linear_solver: LinearSolverType,
    ) -> Result<(Vec<f64>, CheckpointingPath<DiffSl<M, CG>, State>), DiffsolRtError>
    where
        M: Matrix<T: Scalar> + 'static,
        CG: diffsol::CodegenModule + 'static,
        DiffSl<M, CG>: OdeEquations,
        State: OdeSolverState<<DiffSl<M, CG> as Op>::V> + Clone + 'static,
    {
        let guard = self.guard()?;
        let typed = guard
            .as_any()
            .downcast_ref::<TypedCheckpointing<M, CG, State>>()
            .ok_or_else(|| {
                DiffsolRtError::from(DiffsolError::Other(
                    "Checkpointing object is incompatible with this ODE solver".to_string(),
                ))
            })?;

        if typed.matrix_type != expected_matrix_type
            || typed.scalar_type != expected_scalar_type
            || typed.forward_method != expected_forward_method
            || typed.forward_linear_solver != expected_forward_linear_solver
        {
            return Err(DiffsolRtError::from(DiffsolError::Other(
                "Checkpointing object was created with different solver settings".to_string(),
            )));
        }

        Ok((typed.params.clone(), typed.checkpointing.clone()))
    }
}
