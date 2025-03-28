use faer::{get_global_parallelism, Par};

#[derive(Clone, Debug, PartialEq)]
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
