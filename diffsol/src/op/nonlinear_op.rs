use super::Op;
use crate::{scale, Matrix, Scalar, Vector};
use num_traits::{One, Signed, Zero};

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

pub trait NonLinearOpTimePartial: NonLinearOp {
    /// Compute the partial time derivative `∂F/∂t(x, t)` and store it in `y`.
    ///
    /// The default implementation estimates the derivative using a central finite difference
    /// at the supplied state and time.
    fn time_derive_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        let eps_sqrt = Self::T::EPSILON.sqrt();
        let h = (Self::T::one() + t.abs()) * eps_sqrt;
        let mut y_plus = Self::V::zeros(self.nout(), self.context().clone());
        let mut y_minus = Self::V::zeros(self.nout(), self.context().clone());
        self.call_inplace(x, t + h, &mut y_plus);
        self.call_inplace(x, t - h, &mut y_minus);
        y.copy_from(&y_plus);
        *y -= &y_minus;
        *y *= scale(Self::T::one() / (h + h));
    }

    /// Compute the partial time derivative `∂F/∂t(x, t)` and return it.
    fn time_derive(&self, x: &Self::V, t: Self::T) -> Self::V {
        let mut y = Self::V::zeros(self.nout(), self.context().clone());
        self.time_derive_inplace(x, t, &mut y);
        y
    }
}

impl<T: NonLinearOp> NonLinearOpTimePartial for T {}

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

#[cfg(test)]
mod tests {
    use crate::{
        context::nalgebra::NalgebraContext, matrix::dense_nalgebra_serial::NalgebraMat,
        DenseMatrix, NonLinearOp, NonLinearOpAdjoint, NonLinearOpJacobian, NonLinearOpSens,
        NonLinearOpSensAdjoint, NonLinearOpTimePartial, Op, Vector,
    };

    type M = NalgebraMat<f64>;

    struct FakeNonLinearOp {
        ctx: NalgebraContext,
    }

    struct TimeDependentFakeNonLinearOp {
        ctx: NalgebraContext,
    }

    impl Op for FakeNonLinearOp {
        type T = f64;
        type V = crate::NalgebraVec<f64>;
        type M = M;
        type C = NalgebraContext;

        fn context(&self) -> &Self::C {
            &self.ctx
        }
        fn nstates(&self) -> usize {
            2
        }
        fn nout(&self) -> usize {
            2
        }
        fn nparams(&self) -> usize {
            2
        }
    }

    impl NonLinearOp for FakeNonLinearOp {
        fn call_inplace(&self, x: &Self::V, _t: Self::T, y: &mut Self::V) {
            y.copy_from(&Self::V::from_vec(
                vec![
                    2.0 * x.get_index(0) + 3.0 * x.get_index(1),
                    -x.get_index(0) + 4.0 * x.get_index(1),
                ],
                NalgebraContext,
            ));
        }
    }

    impl Op for TimeDependentFakeNonLinearOp {
        type T = f64;
        type V = crate::NalgebraVec<f64>;
        type M = M;
        type C = NalgebraContext;

        fn context(&self) -> &Self::C {
            &self.ctx
        }
        fn nstates(&self) -> usize {
            2
        }
        fn nout(&self) -> usize {
            2
        }
        fn nparams(&self) -> usize {
            0
        }
    }

    impl NonLinearOp for TimeDependentFakeNonLinearOp {
        fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
            y.copy_from(&Self::V::from_vec(
                vec![
                    2.0 * x.get_index(0) + t,
                    -x.get_index(0) + 4.0 * x.get_index(1) - 3.0 * t,
                ],
                NalgebraContext,
            ));
        }
    }

    impl NonLinearOpJacobian for FakeNonLinearOp {
        fn jac_mul_inplace(&self, _x: &Self::V, _t: Self::T, v: &Self::V, y: &mut Self::V) {
            y.copy_from(&Self::V::from_vec(
                vec![
                    2.0 * v.get_index(0) + 3.0 * v.get_index(1),
                    -v.get_index(0) + 4.0 * v.get_index(1),
                ],
                NalgebraContext,
            ));
        }
    }

    impl NonLinearOpAdjoint for FakeNonLinearOp {
        fn jac_transpose_mul_inplace(
            &self,
            _x: &Self::V,
            _t: Self::T,
            v: &Self::V,
            y: &mut Self::V,
        ) {
            y.copy_from(&Self::V::from_vec(
                vec![
                    -2.0 * v.get_index(0) + v.get_index(1),
                    -3.0 * v.get_index(0) - 4.0 * v.get_index(1),
                ],
                NalgebraContext,
            ));
        }
    }

    impl NonLinearOpSens for FakeNonLinearOp {
        fn sens_mul_inplace(&self, _x: &Self::V, _t: Self::T, v: &Self::V, y: &mut Self::V) {
            y.copy_from(&Self::V::from_vec(
                vec![
                    v.get_index(0) + 2.0 * v.get_index(1),
                    3.0 * v.get_index(0) + 4.0 * v.get_index(1),
                ],
                NalgebraContext,
            ));
        }
    }

    impl NonLinearOpSensAdjoint for FakeNonLinearOp {
        fn sens_transpose_mul_inplace(
            &self,
            _x: &Self::V,
            _t: Self::T,
            v: &Self::V,
            y: &mut Self::V,
        ) {
            y.copy_from(&Self::V::from_vec(
                vec![
                    -v.get_index(0) - 3.0 * v.get_index(1),
                    -2.0 * v.get_index(0) - 4.0 * v.get_index(1),
                ],
                NalgebraContext,
            ));
        }
    }

    #[test]
    fn nonlinear_op_default_helpers_construct_expected_vectors_and_matrices() {
        let op = FakeNonLinearOp {
            ctx: NalgebraContext,
        };
        let x = crate::NalgebraVec::from_vec(vec![1.0, 2.0], NalgebraContext);
        let v = crate::NalgebraVec::from_vec(vec![3.0, -1.0], NalgebraContext);

        op.call(&x, 0.0).assert_eq_st(
            &crate::NalgebraVec::from_vec(vec![8.0, 7.0], NalgebraContext),
            1e-12,
        );
        op.jac_mul(&x, 0.0, &v).assert_eq_st(
            &crate::NalgebraVec::from_vec(vec![3.0, -7.0], NalgebraContext),
            1e-12,
        );
        op.sens_mul(&x, 0.0, &v).assert_eq_st(
            &crate::NalgebraVec::from_vec(vec![1.0, 5.0], NalgebraContext),
            1e-12,
        );

        let jac = op.jacobian(&x, 0.0);
        assert_eq!(jac.get_index(0, 0), 2.0);
        assert_eq!(jac.get_index(1, 0), -1.0);
        assert_eq!(jac.get_index(0, 1), 3.0);
        assert_eq!(jac.get_index(1, 1), 4.0);

        let adj = op.adjoint(&x, 0.0);
        assert_eq!(adj.get_index(0, 0), -2.0);
        assert_eq!(adj.get_index(1, 0), -3.0);
        assert_eq!(adj.get_index(0, 1), 1.0);
        assert_eq!(adj.get_index(1, 1), -4.0);

        let sens = op.sens(&x, 0.0);
        assert_eq!(sens.get_index(0, 0), 1.0);
        assert_eq!(sens.get_index(1, 0), 3.0);
        assert_eq!(sens.get_index(0, 1), 2.0);
        assert_eq!(sens.get_index(1, 1), 4.0);

        let sens_adj = op.sens_adjoint(&x, 0.0);
        assert_eq!(sens_adj.get_index(0, 0), -1.0);
        assert_eq!(sens_adj.get_index(1, 0), -2.0);
        assert_eq!(sens_adj.get_index(0, 1), -3.0);
        assert_eq!(sens_adj.get_index(1, 1), -4.0);
    }

    #[test]
    fn nonlinear_op_time_partial_default_helper_uses_finite_differences() {
        let op = TimeDependentFakeNonLinearOp {
            ctx: NalgebraContext,
        };
        let x = crate::NalgebraVec::from_vec(vec![1.0, 2.0], NalgebraContext);

        op.time_derive(&x, 0.5).assert_eq_st(
            &crate::NalgebraVec::from_vec(vec![1.0, -3.0], NalgebraContext),
            1e-8,
        );

        let mut y = crate::NalgebraVec::zeros(2, NalgebraContext);
        op.time_derive_inplace(&x, 0.5, &mut y);
        y.assert_eq_st(
            &crate::NalgebraVec::from_vec(vec![1.0, -3.0], NalgebraContext),
            1e-8,
        );
    }
}
