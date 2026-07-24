use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum JitBackendType {
    #[cfg(feature = "diffsl-cranelift")]
    Cranelift,
    #[cfg(feature = "diffsl-llvm")]
    Llvm,
}

#[allow(dead_code)]
pub fn default_enabled_jit_backend() -> Option<JitBackendType> {
    #[cfg(all(feature = "diffsl-cranelift", not(feature = "diffsl-llvm")))]
    let backend = Some(JitBackendType::Cranelift);
    #[cfg(feature = "diffsl-llvm")]
    let backend = Some(JitBackendType::Llvm);
    #[cfg(all(not(feature = "diffsl-cranelift"), not(feature = "diffsl-llvm")))]
    let backend = None;

    backend
}
