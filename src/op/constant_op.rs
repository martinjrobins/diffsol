use super::Op;
use crate::{Matrix, MatrixSparsityRef, Vector};
use num_traits::{One, Zero};

pub trait ConstantOp: Op {
    fn call_inplace(&self, t: Self::T, y: &mut Self::V);
    fn call(&self, t: Self::T) -> Self::V {
        let mut y = Self::V::zeros(self.nout());
        self.call_inplace(t, &mut y);
        y
    }
}

pub trait ConstantOpSens: ConstantOp {
    /// Compute the product of the gradient of F wrt a parameter vector p with a given vector `J_p(x, t) * v`.
    /// Note that the vector v is of size nparams() and the result is of size nstates().
    fn sens_mul_inplace(&self, _t: Self::T, _v: &Self::V, _y: &mut Self::V);

    /// Compute the gradient of the operator wrt a parameter vector p and store it in the matrix `y`.
    /// `y` should have been previously initialised using the output of [`Op::sparsity`].
    /// The default implementation of this method computes the gradient using [Self::sens_mul_inplace],
    /// but it can be overriden for more efficient implementations.
    fn sens_inplace(&self, t: Self::T, y: &mut Self::M) {
        self._default_sens_inplace(t, y);
    }

    /// Default implementation of the gradient computation (this is the default for [Self::sens_inplace]).
    fn _default_sens_inplace(&self, t: Self::T, y: &mut Self::M) {
        let mut v = Self::V::zeros(self.nparams());
        let mut col = Self::V::zeros(self.nout());
        for j in 0..self.nparams() {
            v[j] = Self::T::one();
            self.sens_mul_inplace(t, &v, &mut col);
            y.set_column(j, &col);
            v[j] = Self::T::zero();
        }
    }

    /// Compute the gradient of the operator wrt a parameter vector p and return it.
    /// See [Self::sens_inplace] for a non-allocating version.
    fn sens(&self, t: Self::T) -> Self::M {
        let n = self.nstates();
        let m = self.nparams();
        let mut y = Self::M::new_from_sparsity(n, m, self.sparsity_sens().map(|s| s.to_owned()));
        self.sens_inplace(t, &mut y);
        y
    }
}

pub trait ConstantOpSensAdjoint: ConstantOp {
    /// Compute the product of the transpose of the gradient of F wrt a parameter vector p with a given vector `-J_p^T(x, t) * v`.
    /// Note that the vector v is of size nstates() and the result is of size nparam().
    fn sens_mul_transpose_inplace(&self, _t: Self::T, _v: &Self::V, _y: &mut Self::V);
}