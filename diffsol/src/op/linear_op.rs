use super::Op;
use crate::{Matrix, Vector};
use num_traits::{One, Zero};

/// LinearOp is a trait for linear operators (i.e. they only depend linearly on the input `x`), see [crate::NonLinearOp] for a non-linear op.
///
/// An example of a linear operator is a matrix-vector product `y = A(t) * x`, where `A(t)` is a matrix.
/// It extends the [Op] trait with methods for calling the operator via a GEMV-like operation (i.e. `y = t * A * x + beta * y`), and for computing the matrix representation of the operator.
pub trait LinearOp: Op {
    /// Compute the operator `y = A(t) * x` at a given state and time, the default implementation uses [Self::gemv_inplace].
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        let beta = Self::T::zero();
        self.gemv_inplace(x, t, beta, y);
    }

    /// Compute the operator via a GEMV operation (i.e. `y = A(t) * x + beta * y`)
    fn gemv_inplace(&self, x: &Self::V, t: Self::T, beta: Self::T, y: &mut Self::V);

    /// Compute the matrix representation of the operator `A(t)` and return it.
    /// See [Self::matrix_inplace] for a non-allocating version.
    fn matrix(&self, t: Self::T) -> Self::M {
        let mut y = Self::M::new_from_sparsity(
            self.nstates(),
            self.nstates(),
            self.sparsity(),
            self.context().clone(),
        );
        self.matrix_inplace(t, &mut y);
        y
    }

    /// Compute the matrix representation of the operator `A(t)` and store it in the matrix `y`.
    /// The default implementation of this method computes the matrix using [Self::gemv_inplace],
    /// but it can be overriden for more efficient implementations.
    fn matrix_inplace(&self, t: Self::T, y: &mut Self::M) {
        self._default_matrix_inplace(t, y);
    }

    /// Default implementation of the matrix computation, see [Self::matrix_inplace].
    fn _default_matrix_inplace(&self, t: Self::T, y: &mut Self::M) {
        let mut v = Self::V::zeros(self.nstates(), self.context().clone());
        let mut col = Self::V::zeros(self.nout(), self.context().clone());
        for j in 0..self.nstates() {
            v.set_index(j, Self::T::one());
            self.call_inplace(&v, t, &mut col);
            y.set_column(j, &col);
            v.set_index(j, Self::T::zero());
        }
    }

    fn sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        None
    }
}

pub trait LinearOpTranspose: LinearOp {
    /// Compute the transpose of the operator via a GEMV operation (i.e. `y = A(t)^T * x + beta * y`)
    fn gemv_transpose_inplace(&self, _x: &Self::V, _t: Self::T, _beta: Self::T, _y: &mut Self::V);

    /// Compute the transpose of the operator `y = A(t)^T * x` at a given state and time, the default implementation uses [Self::gemv_transpose_inplace].
    fn call_transpose_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        let beta = Self::T::zero();
        self.gemv_transpose_inplace(x, t, beta, y);
    }

    /// Compute the matrix representation of the transpose of the operator `A(t)^T` and store it in the matrix `y`.
    /// The default implementation of this method computes the matrix using [Self::gemv_transpose_inplace],
    /// but it can be overriden for more efficient implementations.
    fn transpose_inplace(&self, t: Self::T, y: &mut Self::M) {
        self._default_transpose_inplace(t, y);
    }

    /// Default implementation of the tranpose computation, see [Self::transpose_inplace].
    fn _default_transpose_inplace(&self, t: Self::T, y: &mut Self::M) {
        let mut v = Self::V::zeros(self.nstates(), self.context().clone());
        let mut col = Self::V::zeros(self.nout(), self.context().clone());
        for j in 0..self.nstates() {
            v.set_index(j, Self::T::one());
            self.call_transpose_inplace(&v, t, &mut col);
            y.set_column(j, &col);
            v.set_index(j, Self::T::zero());
        }
    }
    fn transpose_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        None
    }
}

pub trait LinearOpSens: LinearOp {
    /// Compute the product of the gradient of F wrt a parameter vector p with a given vector `J_p(t) * x * v`.
    /// Note that the vector v is of size nparams() and the result is of size nstates().
    /// Default implementation returns zero and panics if nparams() is not zero.
    fn sens_mul_inplace(&self, _x: &Self::V, _t: Self::T, _v: &Self::V, _y: &mut Self::V);

    /// Compute the product of the partial gradient of F wrt a parameter vector p with a given vector `\parial F/\partial p(x, t) * v`, and return the result.
    /// Use `[Self::sens_mul_inplace]` to for a non-allocating version.
    fn sens_mul(&self, x: &Self::V, t: Self::T, v: &Self::V) -> Self::V {
        let mut y = Self::V::zeros(self.nstates(), self.context().clone());
        self.sens_mul_inplace(x, t, v, &mut y);
        y
    }

    /// Compute the gradient of the operator wrt a parameter vector p and store it in the matrix `y`.
    /// `y` should have been previously initialised using the output of [Self::sens_sparsity].
    /// The default implementation of this method computes the gradient using [Self::sens_mul_inplace],
    /// but it can be overriden for more efficient implementations.
    fn sens_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        self._default_sens_inplace(x, t, y);
    }

    /// Default implementation of the gradient computation (this is the default for [Self::sens_inplace]).
    fn _default_sens_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        let mut v = Self::V::zeros(self.nparams(), self.context().clone());
        let mut col = Self::V::zeros(self.nout(), self.context().clone());
        for j in 0..self.nparams() {
            v.set_index(j, Self::T::one());
            self.sens_mul_inplace(x, t, &v, &mut col);
            y.set_column(j, &col);
            v.set_index(j, Self::T::zero());
        }
    }

    /// Compute the gradient of the operator wrt a parameter vector p and return it.
    /// See [Self::sens_inplace] for a non-allocating version.
    fn sens(&self, x: &Self::V, t: Self::T) -> Self::M {
        let n = self.nstates();
        let m = self.nparams();
        let mut y = Self::M::new_from_sparsity(n, m, self.sens_sparsity(), self.context().clone());
        self.sens_inplace(x, t, &mut y);
        y
    }

    fn sens_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        None
    }
}
