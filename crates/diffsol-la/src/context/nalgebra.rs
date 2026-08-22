/// Context for the nalgebra backend.
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

    fn clone_with_nbatch(&self, nbatch: usize) -> Result<Self, crate::LaError> {
        assert!(nbatch > 0, "nbatch must be > 0");
        Ok(Self { nbatch })
    }
}
