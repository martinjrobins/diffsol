use faer::{get_global_parallelism, Par};

/// Context for the faer backend.
///
/// Carries the batch count `nbatch` which determines how many independent
/// ODE systems are solved simultaneously, as well as the faer parallelism
/// configuration.  All vectors and matrices created with this context share
/// the same batch dimension.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FaerContext {
    pub par: Par,
    nbatch: usize,
}

impl FaerContext {
    pub fn new() -> Self {
        Self {
            par: get_global_parallelism(),
            nbatch: 1,
        }
    }
    pub fn with_nbatch(nbatch: usize) -> Self {
        assert!(nbatch > 0, "nbatch must be > 0");
        Self {
            par: get_global_parallelism(),
            nbatch,
        }
    }
}

impl Default for FaerContext {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::Context for FaerContext {
    fn nbatch(&self) -> usize {
        self.nbatch
    }
    fn clone_with_nbatch(&self, nbatch: usize) -> Self {
        assert!(nbatch > 0, "nbatch must be > 0");
        Self {
            par: self.par,
            nbatch,
        }
    }
}
