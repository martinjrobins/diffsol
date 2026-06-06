use faer::{get_global_parallelism, Par};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FaerContext {
    pub par: Par,
    pub nbatch: usize,
}

impl FaerContext {
    pub const fn new(par: Par, nbatch: usize) -> Self {
        Self { par, nbatch }
    }
}

impl Default for FaerContext {
    fn default() -> Self {
        Self {
            par: get_global_parallelism(),
            nbatch: 1,
        }
    }
}
