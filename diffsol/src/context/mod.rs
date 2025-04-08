use crate::{DefaultDenseMatrix, Matrix, Vector};

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

impl<T: Clone + Default> Context for T {}
