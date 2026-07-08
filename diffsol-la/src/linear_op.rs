use crate::{Context, IndexType, Matrix, Scalar, Vector};

/// A linear operator `A` for use with the [crate::LinearSolver] trait.
///
/// This is a minimal, time-unaware description of a linear operator: it can
/// report its shape and context, materialise itself as a matrix, and optionally
/// provide a sparsity pattern. It is deliberately decoupled from the richer,
/// time-aware operator traits in the `diffsol` crate.
///
/// A solver for `Ax = b` uses [Self::sparsity] to set up the matrix storage and
/// [Self::matrix_inplace] to populate the matrix `A` prior to factorisation.
pub trait LinearOp {
    type T: Scalar;
    type V: Vector<T = Self::T, C = Self::C>;
    type M: Matrix<T = Self::T, V = Self::V, C = Self::C>;
    type C: Context;

    /// Number of rows of the operator (i.e. length of the output vector `y`).
    fn nrows(&self) -> IndexType;

    /// Number of columns of the operator (i.e. length of the input vector `x`).
    fn ncols(&self) -> IndexType;

    /// The context associated with this operator.
    fn context(&self) -> &Self::C;

    /// Compute the matrix representation of the operator `A` and store it in `y`.
    ///
    /// `y` is assumed to have been initialised with the sparsity pattern
    /// returned by [Self::sparsity].
    fn matrix_inplace(&self, y: &mut Self::M);

    /// The sparsity pattern of the operator's matrix, or `None` if dense.
    fn sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        None
    }
}
