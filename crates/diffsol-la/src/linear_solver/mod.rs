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
pub use nalgebra::lu::LU as NalgebraNativeLU;
pub use nalgebra::reusable_lu::ReusableLU as NalgebraLU;

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
    use super::LinearSolver;
    use crate::{Context, IndexType, LinearOp, Matrix, Vector};

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

    /// Create a batched 2x2 diagonal operator with `A_b = diag(values[b], values[b])`.
    pub fn batched_diagonal_op<M: Matrix>(values: &[f64], ctx: M::C) -> DiagonalOp<M> {
        use num_traits::FromPrimitive;
        assert_eq!(values.len(), ctx.nbatch());
        let diag = M::V::from_vec(
            values
                .iter()
                .flat_map(|&v| [M::T::from_f64(v).unwrap(); 2])
                .collect(),
            ctx,
        );
        DiagonalOp {
            matrix: M::from_diagonal(&diag),
        }
    }

    /// One factorization per group: `A` carries 2 batches and the right-hand side 4, so
    /// batches 0 and 1 solve against `A_0` and batches 2 and 3 against `A_1`.  A cyclic
    /// mapping would pair batch 1 with `A_1` and fail.
    pub fn test_grouped_lu_solve<M: Matrix, LS: LinearSolver<M>>(ctx2: M::C) {
        use num_traits::FromPrimitive;
        let f = |x: f64| M::T::from_f64(x).unwrap();
        let wide = ctx2.clone_with_nbatch(4).unwrap();
        let op = batched_diagonal_op::<M>(&[2.0, 5.0], ctx2);
        let mut s = LS::default();
        s.set_sparsity(&op);
        s.set_linearisation(&op);
        let b = M::V::from_vec((1..=8).map(|i| f(i as f64)).collect(), wide.clone());
        let x = s.solve(&b).unwrap();
        let expected = M::V::from_vec(
            vec![
                f(0.5),
                f(1.0),
                f(1.5),
                f(2.0),
                f(1.0),
                f(1.2),
                f(1.4),
                f(1.6),
            ],
            wide,
        );
        x.assert_eq_st(&expected, f(1e-10));
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
