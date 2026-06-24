use crate::DiffsolError;

/// Context for the nalgebra backend.
///
/// Batching (`nbatch > 1`) is not supported by this backend; use the CUDA
/// backend instead.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct NalgebraContext {}

impl NalgebraContext {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for NalgebraContext {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::Context for NalgebraContext {
    fn clone_with_nbatch(&self, nbatch: usize) -> Result<Self, DiffsolError> {
        if nbatch != 1 {
            Err(DiffsolError::Other(
                "NalgebraContext does not support batching (nbatch > 1). Use the CUDA backend instead."
                    .to_string(),
            ))
        } else {
            Ok(Self {})
        }
    }
}
