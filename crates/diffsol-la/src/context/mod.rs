use crate::{error::LaError, DefaultDenseMatrix, Matrix, Vector};

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "nalgebra")]
pub mod nalgebra;

#[cfg(feature = "faer")]
pub mod faer;

/// defines the current execution and allocation context of an operator / vector / matrix
/// for example:
/// - threading model (e.g. single-threaded, multi-threaded, GPU)
/// - custom allocators, host/device memory
/// - etc.
///
/// It will generally be the case that all the operators / vectors / matrices for the current ode problem
/// share the same context
pub trait Context: Clone + Default {
    /// Returns the batch count for this context.
    ///
    /// When `nbatch > 1`, vectors and matrices store data for `nbatch`
    /// independent ODE systems simultaneously.  Operations between operands
    /// with different batch counts use broadcast semantics: an operand with
    /// `nbatch == 1` is applied to all batches of the other operand.
    fn nbatch(&self) -> usize {
        1
    }
    /// Creates a new context with the given batch count.
    ///
    /// Other properties of the context (e.g. CUDA stream, faer parallelism)
    /// are preserved.
    ///
    /// Returns an error if the backend does not support batching (i.e.
    /// `nbatch > 1` for CPU backends such as faer and nalgebra).
    fn clone_with_nbatch(&self, nbatch: usize) -> Result<Self, LaError>;
    /// Panics if the two batch counts are incompatible.
    ///
    /// Compatibility rule: two batches are compatible if they are equal, or
    /// if either one is `1` (broadcast).  Only panics when both are `> 1`
    /// and differ.
    fn assert_compatible_nbatch(&self, other_nbatch: usize, op: &str) {
        let self_nbatch = self.nbatch();
        if self_nbatch != other_nbatch && self_nbatch != 1 && other_nbatch != 1 {
            panic!(
                "incompatible nbatch in {}: lhs={}, rhs={}",
                op, self_nbatch, other_nbatch
            );
        }
    }
    fn vector_from_element<V: Vector<C = Self>>(&self, len: usize, value: V::T) -> V {
        V::from_element(len, value, self.clone())
    }
    fn vector_from_vec<V: Vector<C = Self>>(&self, vec: Vec<V::T>) -> V {
        V::from_vec(vec, self.clone())
    }
    fn vector_zeros<V: Vector<C = Self>>(&self, len: usize) -> V {
        V::zeros(len, self.clone())
    }
    fn dense_mat_zeros<V: Vector<C = Self> + DefaultDenseMatrix>(
        &self,
        rows: usize,
        cols: usize,
    ) -> <V as DefaultDenseMatrix>::M {
        <<V as DefaultDenseMatrix>::M as Matrix>::zeros(rows, cols, self.clone())
    }
}
