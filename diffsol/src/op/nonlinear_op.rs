use super::Op;
use crate::{Matrix, Vector};
use num_traits::{One, Zero};

// NonLinearOp is a trait that defines a nonlinear operator or function `F` that maps an input vector `x` to an output vector `y`, (i.e. `y = F(x, t)`).
// It extends the [Op] trait with methods for computing the operator and its Jacobian.
//
// The operator is defined by the [Self::call_inplace] method, which computes the function `F(x, t)` at a given state and time.
// The Jacobian is defined by the [Self::jac_mul_inplace] method, which computes the product of the Jacobian with a given vector `J(x, t) * v`.
pub trait NonLinearOp: Op {
    /// Compute the operator `F(x, t)` at a given state and time.
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V);

    /// Compute the operator `F(x, t)` at a given state and time, and return the result.
    /// Use `[Self::call_inplace]` to for a non-allocating version.
    fn call(&self, x: &Self::V, t: Self::T) -> Self::V {
        let mut y = Self::V::zeros(self.nout(), self.context().clone());
        self.call_inplace(x, t, &mut y);
        y
    }
}

pub trait NonLinearOpSens: NonLinearOp {
    /// Compute the product of the gradient of F wrt a parameter vector p with a given vector `J_p(x, t) * v`.
    /// Note that the vector v is of size nparams() and the result is of size nstates().
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
pub trait NonLinearOpSensAdjoint: NonLinearOp {
    /// Compute the product of the negative tramspose of the gradient of F wrt a parameter vector p with a given vector `-J_p(x, t)^T * v`.
    fn sens_transpose_mul_inplace(&self, _x: &Self::V, _t: Self::T, _v: &Self::V, _y: &mut Self::V);

    /// Compute the negative transpose of the gradient of the operator wrt a parameter vector p and return it.
    /// See [Self::sens_adjoint_inplace] for a non-allocating version.
    fn sens_adjoint(&self, x: &Self::V, t: Self::T) -> Self::M {
        let n = self.nstates();
        let mut y =
            Self::M::new_from_sparsity(n, n, self.sens_adjoint_sparsity(), self.context().clone());
        self.sens_adjoint_inplace(x, t, &mut y);
        y
    }

    /// Compute the negative transpose of the gradient of the operator wrt a parameter vector p and store it in the matrix `y`.
    /// `y` should have been previously initialised using the output of [Self::sens_adjoint_sparsity].
    /// The default implementation of this method computes the gradient using [Self::sens_transpose_mul_inplace],
    /// but it can be overriden for more efficient implementations.
    fn sens_adjoint_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        self._default_sens_adjoint_inplace(x, t, y);
    }

    /// Default implementation of the gradient computation (this is the default for [Self::sens_adjoint_inplace]).
    fn _default_sens_adjoint_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        let mut v = Self::V::zeros(self.nstates(), self.context().clone());
        let mut col = Self::V::zeros(self.nout(), self.context().clone());
        for j in 0..self.nstates() {
            v.set_index(j, Self::T::one());
            self.sens_transpose_mul_inplace(x, t, &v, &mut col);
            y.set_column(j, &col);
            v.set_index(j, Self::T::zero());
        }
    }

    fn sens_adjoint_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        None
    }
}
pub trait NonLinearOpAdjoint: NonLinearOp {
    /// Compute the product of the transpose of the Jacobian with a given vector `-J(x, t)^T * v`.
    /// The default implementation fails with a panic, as this method is not implemented by default
    /// and should be implemented by the user if needed.
    fn jac_transpose_mul_inplace(&self, _x: &Self::V, _t: Self::T, _v: &Self::V, _y: &mut Self::V);

    /// Compute the Adjoint matrix `-J^T(x, t)` of the operator and store it in the matrix `y`.
    /// `y` should have been previously initialised using the output of [`Self::adjoint_sparsity`].
    /// The default implementation of this method computes the Jacobian using [Self::jac_transpose_mul_inplace],
    /// but it can be overriden for more efficient implementations.
    fn adjoint_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        self._default_adjoint_inplace(x, t, y);
    }

    /// Default implementation of the Adjoint computation (this is the default for [Self::adjoint_inplace]).
    fn _default_adjoint_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        let mut v = Self::V::zeros(self.nstates(), self.context().clone());
        let mut col = Self::V::zeros(self.nout(), self.context().clone());
        for j in 0..self.nstates() {
            v.set_index(j, Self::T::one());
            self.jac_transpose_mul_inplace(x, t, &v, &mut col);
            y.set_column(j, &col);
            v.set_index(j, Self::T::zero());
        }
    }

    /// Compute the Adjoint matrix `-J^T(x, t)` of the operator and return it.
    /// See [Self::adjoint_inplace] for a non-allocating version.
    fn adjoint(&self, x: &Self::V, t: Self::T) -> Self::M {
        let n = self.nstates();
        let mut y =
            Self::M::new_from_sparsity(n, n, self.adjoint_sparsity(), self.context().clone());
        self.adjoint_inplace(x, t, &mut y);
        y
    }
    /// Return sparsity information (if available)
    fn adjoint_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        None
    }
}
pub trait NonLinearOpJacobian: NonLinearOp {
    /// Compute the product of the Jacobian with a given vector `J(x, t) * v`.
    fn jac_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V);

    /// Compute the product of the Jacobian with a given vector `J(x, t) * v`, and return the result.
    /// Use `[Self::jac_mul_inplace]` to for a non-allocating version.
    fn jac_mul(&self, x: &Self::V, t: Self::T, v: &Self::V) -> Self::V {
        let mut y = Self::V::zeros(self.nstates(), self.context().clone());
        self.jac_mul_inplace(x, t, v, &mut y);
        y
    }

    /// Compute the Jacobian matrix `J(x, t)` of the operator and return it.
    /// See [Self::jacobian_inplace] for a non-allocating version.
    fn jacobian(&self, x: &Self::V, t: Self::T) -> Self::M {
        let n = self.nstates();
        let mut y =
            Self::M::new_from_sparsity(n, n, self.jacobian_sparsity(), self.context().clone());
        self.jacobian_inplace(x, t, &mut y);
        y
    }

    /// Return sparsity information (if available)
    fn jacobian_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        None
    }

    /// Compute the Jacobian matrix `J(x, t)` of the operator and store it in the matrix `y`.
    /// `y` should have been previously initialised using the output of [Self::jacobian_sparsity].
    /// The default implementation of this method computes the Jacobian using [Self::jac_mul_inplace],
    /// but it can be overriden for more efficient implementations.
    fn jacobian_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        self._default_jacobian_inplace(x, t, y);
    }

    /// Default implementation of the Jacobian computation (this is the default for [Self::jacobian_inplace]).
    fn _default_jacobian_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        let mut v = Self::V::zeros(self.nstates(), self.context().clone());
        let mut col = Self::V::zeros(self.nout(), self.context().clone());
        for j in 0..self.nstates() {
            v.set_index(j, Self::T::one());
            self.jac_mul_inplace(x, t, &v, &mut col);
            y.set_column(j, &col);
            v.set_index(j, Self::T::zero());
        }
    }
}
