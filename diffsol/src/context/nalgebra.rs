#[derive(Copy, Clone, Debug, PartialEq)]
pub struct NalgebraContext {
    pub nbatch: usize,
}

impl NalgebraContext {
    pub const fn new(nbatch: usize) -> Self {
        Self { nbatch }
    }
}

impl Default for NalgebraContext {
    fn default() -> Self {
        Self { nbatch: 1 }
    }
}
