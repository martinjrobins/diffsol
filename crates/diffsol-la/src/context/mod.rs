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
/// Kept out of line so that the panic's formatting does not stop
/// [`Context::assert_compatible_nbatch`] from being inlined into hot operations.
#[cold]
#[inline(never)]
fn incompatible_nbatch(lhs: usize, rhs: usize, op: &str) -> ! {
    panic!("incompatible nbatch in {}: lhs={}, rhs={}", op, lhs, rhs);
}

/// Source batch feeding destination batch `dest` of `dest_nbatch`, for an operand holding
/// `src_nbatch` batches.
///
/// Broadcast is grouped: each source batch is repeated across `dest_nbatch / src_nbatch`
/// *contiguous* destination batches, so `src_nbatch == 2` against `dest_nbatch == 4` reads
/// batches `[0, 0, 1, 1]`.  The four cases this covers:
/// - `src_nbatch == dest_nbatch` gives `dest`
/// - `src_nbatch == 1` gives `0`
/// - `src_nbatch == B`, `dest_nbatch == B * P` gives `dest / P`
/// - `dest_nbatch < src_nbatch` gives the first batch of each group, i.e. `0` when
///   `dest_nbatch == 1` (see [`Context::assert_compatible_nbatch`] for when that arises)
#[inline]
pub(crate) fn broadcast_batch(dest: usize, src_nbatch: usize, dest_nbatch: usize) -> usize {
    dest * src_nbatch / dest_nbatch
}

pub trait Context: Clone + Default {
    /// Returns the batch count for this context.
    ///
    /// When `nbatch > 1`, vectors and matrices store data for `nbatch`
    /// independent ODE systems simultaneously.  Operations between operands
    /// with different batch counts use broadcast semantics: an operand with
    /// `nbatch == B` is applied to `B * P` batches of the other operand, repeating each
    /// of its batches over `P` contiguous batches (`nbatch == 1` is the `B == 1` case,
    /// applied to every batch of the other operand).
    #[inline]
    fn nbatch(&self) -> usize {
        1
    }
    /// Creates a new context with the given batch count.
    ///
    /// Other properties of the context (e.g. CUDA stream, faer parallelism)
    /// are preserved.
    ///
    /// Returns an error when a backend cannot create a requested batch context.
    fn clone_with_nbatch(&self, nbatch: usize) -> Result<Self, LaError>;
    /// Panics if the two batch counts are incompatible.
    ///
    /// Compatibility rule: two batch counts are compatible if the larger is an exact
    /// multiple of the smaller, so the smaller operand broadcasts over exact repeat
    /// groups.  Equal counts and `1` against `N` are the special cases of it; `2` against
    /// `3` panics.
    ///
    /// In-place operations write `self.nbatch()` batches: a right-hand side with fewer
    /// batches is broadcast over them (see [`broadcast_batch`]), and where the left-hand
    /// side has fewer batches than the right only the first right-hand side batch of each
    /// group is used.
    #[inline]
    fn assert_compatible_nbatch(&self, other_nbatch: usize, op: &str) {
        let self_nbatch = self.nbatch();
        let (min, max) = if self_nbatch < other_nbatch {
            (self_nbatch, other_nbatch)
        } else {
            (other_nbatch, self_nbatch)
        };
        if min == 0 || !max.is_multiple_of(min) {
            incompatible_nbatch(self_nbatch, other_nbatch, op);
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
