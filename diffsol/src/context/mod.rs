/// defines the current execution and allocation context of an operator / vector / matrix
/// for example:
/// - threading model (e.g. single-threaded, multi-threaded, GPU)
/// - custom allocators, host/device memory
/// - etc.
/// 
/// It will generally be the case that all the operators / vectors / matrices for the current ode problem
/// share the same context
pub trait Context: Clone + Default {}

impl<T: Clone + Default> Context for T {}