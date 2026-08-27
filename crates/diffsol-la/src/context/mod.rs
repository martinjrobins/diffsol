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

/// The divisibility half of the compatibility rule, kept out of line so that
/// [`Context::assert_compatible_nbatch`] inlines as three comparisons and nothing else.
#[cold]
#[inline(never)]
fn assert_grouped_nbatch(lhs: usize, rhs: usize, op: &str) {
    let (min, max) = if lhs < rhs { (lhs, rhs) } else { (rhs, lhs) };
    if !max.is_multiple_of(min) {
        incompatible_nbatch(lhs, rhs, op);
    }
}

/// The divisibility half of the broadcast-into rule, kept out of line for the same reason as
/// [`assert_grouped_nbatch`].
#[cold]
#[inline(never)]
fn assert_divides_nbatch(src_nbatch: usize, dest_nbatch: usize, op: &str) {
    if src_nbatch == 0 || !dest_nbatch.is_multiple_of(src_nbatch) {
        incompatible_nbatch(dest_nbatch, src_nbatch, op);
    }
}

/// Source batch feeding destination batch `dest` of `dest_nbatch`, for an operand holding
/// `src_nbatch` batches. Broadcast is grouped: each source batch is repeated across `dest_nbatch / src_nbatch`
/// *contiguous* destination batches
#[inline]
pub(crate) fn broadcast_batch(dest: usize, src_nbatch: usize, dest_nbatch: usize) -> usize {
    // matching batch counts are by far the common case, and returning `dest` unchanged keeps
    // the caller's index provably in range -- without this branch the division loses that and
    // the elementwise loops downstream deoptimise (~6% on lin_alg_ops axpy/faer/100)
    if src_nbatch == dest_nbatch {
        return dest;
    }
    dest * src_nbatch / dest_nbatch
}

/// Runs `f` out of line and marked unlikely.
///
/// Used for the broadcasting tails of the elementwise macros: they are cold next to the
/// equal-batch paths above them, and leaving them inline costs the hot path its own inlining
/// (worth ~15% on lin_alg_ops copy_from_view/faer/100).
#[cold]
#[inline(never)]
pub(crate) fn cold_call<R, F: FnOnce() -> R>(f: F) -> R {
    f()
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
    /// Panics if the two batch counts cannot appear in the same operation.
    ///
    /// Compatibility rule: two batch counts are compatible if the larger is an exact
    /// multiple of the smaller, so the smaller operand broadcasts over exact repeat
    /// groups.  Equal counts and `1` against `N` are the special cases of it; `2` against
    /// `3` panics.
    ///
    /// This is the rule for operations that *allocate* their result: the result carries
    /// `max(nbatch)` and both operands broadcast into it (see [`broadcast_batch`]), so every
    /// input batch is read whichever operand is the narrower one.  Operations that write into
    /// an operand chosen by position use [`Context::assert_broadcastable_into`] instead.
    #[inline]
    fn assert_compatible_nbatch(&self, other_nbatch: usize, op: &str) {
        let self_nbatch = self.nbatch();
        // equal counts and `1` against `N` are the overwhelmingly common cases, and three
        // comparisons are cheaper than the division the general rule needs -- which is worth
        // ~1ns per call on the smallest ops (see lin_alg_ops axpy/faer/100)
        if self_nbatch == other_nbatch || self_nbatch == 1 || other_nbatch == 1 {
            return;
        }
        assert_grouped_nbatch(self_nbatch, other_nbatch, op);
    }
    /// Panics unless an operand holding `src_nbatch` batches broadcasts *into* `self`'s
    /// batches, i.e. unless `self.nbatch()` is an exact multiple of `src_nbatch`.
    ///
    /// This is the rule for operations with a fixed destination -- assignments, `copy_from`,
    /// `axpy`, `gemv` into `y`, a linear solve in place: the destination's batch count governs
    /// the loop, so a source holding *more* batches would have batches silently ignored
    /// rather than broadcast.  `self` here is always the destination.
    #[inline]
    fn assert_broadcastable_into(&self, src_nbatch: usize, op: &str) {
        let nbatch = self.nbatch();
        if src_nbatch == nbatch || src_nbatch == 1 {
            return;
        }
        assert_divides_nbatch(src_nbatch, nbatch, op);
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
