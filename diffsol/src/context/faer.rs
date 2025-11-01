use faer::{get_global_parallelism, Par};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FaerContext {
    par: Par,
}

impl Default for FaerContext {
    fn default() -> Self {
        Self {
            par: get_global_parallelism(),
        }
    }
}
