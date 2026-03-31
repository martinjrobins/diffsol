use thiserror::Error;
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum DiffsolMcpError {
    #[error("{0}")]
    Message(String),
    #[error("problem `{0}` was not found")]
    ProblemNotFound(Uuid),
    #[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
    #[error("DiffSL JIT backend is ambiguous; specify `jit_backend` in the problem config")]
    AmbiguousJitBackend,
    #[error("no DiffSL JIT backend is enabled for this build")]
    NoJitBackendEnabled,
    #[error("either `final_time` or a non-empty `t_eval` must be provided")]
    MissingSolveTarget,
    #[error("`t_eval` must not be empty")]
    EmptyTEval,
    #[error("resource URI `{0}` is not recognised")]
    UnknownResource(String),
    #[error("invalid problem resource URI `{0}`")]
    InvalidProblemUri(String),
    #[error(transparent)]
    Diffsol(#[from] diffsol_c::DiffsolJsError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

impl DiffsolMcpError {
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }
}
