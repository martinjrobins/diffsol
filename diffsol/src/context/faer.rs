use faer::{get_global_parallelism, Par};

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
        Self {
            par: self.par,
            nbatch,
        }
    }
}
