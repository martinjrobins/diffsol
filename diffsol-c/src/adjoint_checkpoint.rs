use std::{
    any::Any,
    sync::{Arc, Mutex, MutexGuard},
};

use diffsol::{CheckpointingPath, DiffSl, DiffsolError, OdeEquations};

use crate::{
    error::DiffsolRtError, linear_solver_type::LinearSolverType,
    ode_solver_tag::OdeSolverMethodTag, ode_solver_type::OdeSolverType, scalar_type::Scalar,
};

pub(crate) trait AdjointCheckpoint: Any + Send {
    fn as_any(&self) -> &dyn Any;
    fn method(&self) -> OdeSolverType;
    fn linear_solver(&self) -> LinearSolverType;
    fn params(&self) -> &[f64];
}

impl dyn AdjointCheckpoint + '_ {
    pub(crate) fn data<M, CG, Tag>(
        &self,
    ) -> Result<&AdjointCheckpointData<M, CG, Tag>, DiffsolError>
    where
        M: diffsol::Matrix<T: Scalar> + 'static,
        CG: diffsol::CodegenModule + 'static,
        DiffSl<M, CG>: OdeEquations<V = M::V> + 'static,
        Tag: OdeSolverMethodTag<M, CG> + 'static,
    {
        self.as_any()
            .downcast_ref::<AdjointCheckpointData<M, CG, Tag>>()
            .ok_or_else(|| {
                DiffsolError::Other(
                    "Adjoint checkpoint is incompatible with this ODE solver".to_string(),
                )
            })
    }
}

pub(crate) struct AdjointCheckpointData<M, CG, Tag>
where
    M: diffsol::Matrix<T: Scalar>,
    CG: diffsol::CodegenModule,
    DiffSl<M, CG>: OdeEquations<V = M::V>,
    Tag: OdeSolverMethodTag<M, CG>,
{
    pub(crate) checkpointing: CheckpointingPath<DiffSl<M, CG>, Tag::State>,
    params: Vec<f64>,
    method: OdeSolverType,
    linear_solver: LinearSolverType,
}

impl<M, CG, Tag> AdjointCheckpointData<M, CG, Tag>
where
    M: diffsol::Matrix<T: Scalar>,
    CG: diffsol::CodegenModule,
    DiffSl<M, CG>: OdeEquations<V = M::V>,
    Tag: OdeSolverMethodTag<M, CG>,
{
    pub(crate) fn new(
        checkpointing: CheckpointingPath<DiffSl<M, CG>, Tag::State>,
        params: Vec<f64>,
        method: OdeSolverType,
        linear_solver: LinearSolverType,
    ) -> Self {
        Self {
            checkpointing,
            params,
            method,
            linear_solver,
        }
    }
}

impl<M, CG, Tag> AdjointCheckpoint for AdjointCheckpointData<M, CG, Tag>
where
    M: diffsol::Matrix<T: Scalar> + 'static,
    CG: diffsol::CodegenModule + 'static,
    DiffSl<M, CG>: OdeEquations<V = M::V> + 'static,
    Tag: OdeSolverMethodTag<M, CG> + 'static,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn method(&self) -> OdeSolverType {
        self.method
    }

    fn linear_solver(&self) -> LinearSolverType {
        self.linear_solver
    }

    fn params(&self) -> &[f64] {
        &self.params
    }
}

#[derive(Clone)]
pub struct AdjointCheckpointWrapper(Arc<Mutex<Box<dyn AdjointCheckpoint>>>);

impl AdjointCheckpointWrapper {
    pub(crate) fn new(checkpoint: Box<dyn AdjointCheckpoint>) -> Self {
        Self(Arc::new(Mutex::new(checkpoint)))
    }

    pub(crate) fn guard(
        &self,
    ) -> Result<MutexGuard<'_, Box<dyn AdjointCheckpoint>>, DiffsolRtError> {
        self.0.lock().map_err(|_| {
            DiffsolError::Other("Adjoint checkpoint mutex poisoned".to_string()).into()
        })
    }
}
