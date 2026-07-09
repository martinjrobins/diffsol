use diffsol_la::{Context, IndexType, Matrix, Scalar, Vector};

/// A non-linear operator `F` for use with the [crate::NonLinearSolver] trait.
///
/// This is a minimal description of a non-linear function `F` that
/// maps an input vector `x` to an output vector `y` (i.e. `y = F(x)`). It can report
/// its shape and context and evaluate itself in place. It is deliberately decoupled
/// from the richer, time-aware operator traits in the `diffsol` crate.
pub trait NonLinearOp {
    type T: Scalar;
    type V: Vector<T = Self::T, C = Self::C>;
    type M: Matrix<T = Self::T, V = Self::V, C = Self::C>;
    type C: Context;

    /// Return the number of input states of the operator (i.e. length of `x`).
    fn nstates(&self) -> IndexType;

    /// Return the number of outputs of the operator (i.e. length of `y`).
    fn nout(&self) -> IndexType;

    /// The context associated with this operator.
    fn context(&self) -> &Self::C;

    /// Compute the operator `F(x)` and store it in `y`.
    fn call_inplace(&self, x: &Self::V, y: &mut Self::V);

    /// Compute the operator `F(x)` and return the result.
    /// Use [Self::call_inplace] for a non-allocating version.
    fn call(&self, x: &Self::V) -> Self::V {
        let mut y = Self::V::zeros(self.nout(), self.context().clone());
        self.call_inplace(x, &mut y);
        y
    }
}

/// A non-linear operator that can also provide its Jacobian `J = dF/dx`.
pub trait NonLinearOpJacobian: NonLinearOp {
    /// Compute the product of the Jacobian with a given vector `J(x) * v` and store it in `y`.
    fn jac_mul_inplace(&self, x: &Self::V, v: &Self::V, y: &mut Self::V);

    /// Compute the product of the Jacobian with a given vector `J(x) * v` and return the result.
    /// Use [Self::jac_mul_inplace] for a non-allocating version.
    fn jac_mul(&self, x: &Self::V, v: &Self::V) -> Self::V {
        let mut y = Self::V::zeros(self.nstates(), self.context().clone());
        self.jac_mul_inplace(x, v, &mut y);
        y
    }

    /// Compute the Jacobian matrix `J(x)` of the operator and return it.
    /// See [Self::jacobian_inplace] for a non-allocating version.
    fn jacobian(&self, x: &Self::V) -> Self::M {
        let n = self.nstates();
        let mut y =
            Self::M::new_from_sparsity(n, n, self.jacobian_sparsity(), self.context().clone());
        self.jacobian_inplace(x, &mut y);
        y
    }

    /// Compute the Jacobian matrix `J(x)` of the operator and store it in the matrix `y`.
    /// `y` should have been previously initialised using the output of [Self::jacobian_sparsity].
    /// The default implementation computes the Jacobian using [Self::jac_mul_inplace], but it
    /// can be overriden for more efficient implementations.
    fn jacobian_inplace(&self, x: &Self::V, y: &mut Self::M) {
        self._default_jacobian_inplace(x, y);
    }

    /// Default implementation of the Jacobian computation (this is the default for [Self::jacobian_inplace]).
    fn _default_jacobian_inplace(&self, x: &Self::V, y: &mut Self::M) {
        use num_traits::{One, Zero};
        let mut v = Self::V::zeros(self.nstates(), self.context().clone());
        let mut col = Self::V::zeros(self.nout(), self.context().clone());
        for j in 0..self.nstates() {
            v.set_index(j, Self::T::one());
            self.jac_mul_inplace(x, &v, &mut col);
            y.set_column(j, &col);
            v.set_index(j, Self::T::zero());
        }
    }

    /// The sparsity pattern of the operator's Jacobian, or `None` if dense.
    fn jacobian_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        None
    }
}
