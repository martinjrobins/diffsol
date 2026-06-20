/// Context for the nalgebra backend.
///
/// Carries the batch count `nbatch` which determines how many independent
/// ODE systems are solved simultaneously.  All vectors and matrices created
/// with this context share the same batch dimension.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct NalgebraContext {
    nbatch: usize,
}

impl NalgebraContext {
    pub fn new() -> Self {
        Self { nbatch: 1 }
    }
    pub fn with_nbatch(nbatch: usize) -> Self {
        assert!(nbatch > 0, "nbatch must be > 0");
        Self { nbatch }
    }
}

impl Default for NalgebraContext {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::Context for NalgebraContext {
    fn nbatch(&self) -> usize {
        self.nbatch
    }
    fn clone_with_nbatch(&self, nbatch: usize) -> Self {
        assert!(nbatch > 0, "nbatch must be > 0");
        Self { nbatch }
    }
}
