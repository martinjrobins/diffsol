use crate::{error::LaError, LinearOp, Matrix};

#[cfg(feature = "nalgebra")]
pub mod nalgebra;

#[cfg(feature = "faer")]
pub mod faer;

#[cfg(feature = "suitesparse")]
pub mod suitesparse;

#[cfg(feature = "cuda")]
pub mod cuda;

pub use faer::lu::LU as FaerLU;
pub use nalgebra::lu::LU as NalgebraLU;

/// A solver for the linear problem `Ax = b`, where `A` is a [LinearOp].
pub trait LinearSolver<M: Matrix>: Default {
    /// Set the point at which the linear operator `A` is evaluated and factorise it.
    ///
    /// The operator is assumed to have the same sparsity as that given to
    /// [Self::set_sparsity].
    fn set_linearisation<C: LinearOp<V = M::V, T = M::T, M = M, C = M::C>>(&mut self, op: &C);

    /// Set the sparsity of the problem to be solved, any previous problem is discarded.
    ///
    /// Any internal state of the solver is reset. This function will normally set
    /// the sparsity pattern of the matrix to be solved.
    fn set_sparsity<C: LinearOp<V = M::V, T = M::T, M = M, C = M::C>>(&mut self, op: &C);

    /// Solve the problem `Ax = b` and return the solution `x`.
    ///
    /// Panics if [Self::set_linearisation] has not been called previously.
    fn solve(&self, b: &M::V) -> Result<M::V, LaError> {
        let mut b = b.clone();
        self.solve_in_place(&mut b)?;
        Ok(b)
    }

    fn solve_in_place(&self, b: &mut M::V) -> Result<(), LaError>;
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::{IndexType, LinearOp, Matrix, Vector};

    /// A simple diagonal [LinearOp] used for testing the linear solvers.
    pub struct DiagonalOp<M: Matrix> {
        matrix: M,
    }

    /// Create a 2x2 diagonal operator `A = diag(value, value)`.
    pub fn diagonal_op<M: Matrix>(value: f64) -> DiagonalOp<M> {
        use num_traits::FromPrimitive;
        let v = M::T::from_f64(value).unwrap();
        let diag = M::V::from_vec(vec![v, v], Default::default());
        DiagonalOp {
            matrix: M::from_diagonal(&diag),
        }
    }

    impl<M: Matrix> LinearOp for DiagonalOp<M> {
        type T = M::T;
        type V = M::V;
        type M = M;
        type C = M::C;

        fn nrows(&self) -> IndexType {
            self.matrix.nrows()
        }
        fn ncols(&self) -> IndexType {
            self.matrix.ncols()
        }
        fn context(&self) -> &Self::C {
            self.matrix.context()
        }
        fn matrix_inplace(&self, y: &mut Self::M) {
            y.copy_from(&self.matrix);
        }
        fn sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
            self.matrix.sparsity().map(|s| {
                use crate::matrix::sparsity::MatrixSparsityRef;
                s.to_owned()
            })
        }
    }
}
