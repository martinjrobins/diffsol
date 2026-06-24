use faer::{get_global_parallelism, Par};

use crate::DiffsolError;

/// Context for the faer backend.
///
/// Carries the faer parallelism configuration.  Batching (`nbatch > 1`) is
/// not supported by this backend; use the CUDA backend instead.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FaerContext {
    pub par: Par,
}

impl FaerContext {
    pub fn new() -> Self {
        Self {
            par: get_global_parallelism(),
        }
    }
}

impl Default for FaerContext {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::Context for FaerContext {
    fn clone_with_nbatch(&self, nbatch: usize) -> Result<Self, DiffsolError> {
        if nbatch != 1 {
            Err(DiffsolError::Other(
                "FaerContext does not support batching (nbatch > 1). Use the CUDA backend instead."
                    .to_string(),
            ))
        } else {
            Ok(Self { par: self.par })
        }
    }
}
