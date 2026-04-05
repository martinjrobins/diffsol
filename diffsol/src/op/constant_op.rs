use super::Op;
use crate::{Matrix, Vector};
use num_traits::{One, Zero};

pub trait ConstantOp: Op {
    fn call_inplace(&self, t: Self::T, y: &mut Self::V);
    fn call(&self, t: Self::T) -> Self::V {
        let mut y = Self::V::zeros(self.nout(), self.context().clone());
        self.call_inplace(t, &mut y);
        y
    }
}

pub trait ConstantOpSens: ConstantOp {
    /// Compute the product of the gradient of F wrt a parameter vector p with a given vector `J_p(x, t) * v`.
    /// Note that the vector v is of size nparams() and the result is of size nstates().
    fn sens_mul_inplace(&self, _t: Self::T, _v: &Self::V, _y: &mut Self::V);

    /// Compute the gradient of the operator wrt a parameter vector p and store it in the matrix `y`.
    /// `y` should have been previously initialised using the output of [Self::sens_sparsity].
    /// The default implementation of this method computes the gradient using [Self::sens_mul_inplace],
    /// but it can be overriden for more efficient implementations.
    fn sens_inplace(&self, t: Self::T, y: &mut Self::M) {
        self._default_sens_inplace(t, y);
    }

    /// Default implementation of the gradient computation (this is the default for [Self::sens_inplace]).
    fn _default_sens_inplace(&self, t: Self::T, y: &mut Self::M) {
        let mut v = Self::V::zeros(self.nparams(), self.context().clone());
        let mut col = Self::V::zeros(self.nout(), self.context().clone());
        for j in 0..self.nparams() {
            v.set_index(j, Self::T::one());
            self.sens_mul_inplace(t, &v, &mut col);
            y.set_column(j, &col);
            v.set_index(j, Self::T::zero());
        }
    }

    /// Compute the gradient of the operator wrt a parameter vector p and return it.
    /// See [Self::sens_inplace] for a non-allocating version.
    fn sens(&self, t: Self::T) -> Self::M {
        let n = self.nstates();
        let m = self.nparams();
        let mut y = Self::M::new_from_sparsity(n, m, self.sens_sparsity(), self.context().clone());
        self.sens_inplace(t, &mut y);
        y
    }
    fn sens_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        None
    }
}

pub trait ConstantOpSensAdjoint: ConstantOp {
    /// Compute the product of the transpose of the gradient of F wrt a parameter vector p with a given vector `-J_p^T(x, t) * v`.
    /// Note that the vector v is of size nstates() and the result is of size nparam().
    fn sens_transpose_mul_inplace(&self, t: Self::T, v: &Self::V, y: &mut Self::V);

    /// Compute the negative transpose of the gradient of the operator wrt a parameter vector p and return it.
    /// See [Self::sens_adjoint_inplace] for a non-allocating version.
    fn sens_adjoint(&self, t: Self::T) -> Self::M {
        let n = self.nstates();
        let mut y =
            Self::M::new_from_sparsity(n, n, self.sens_adjoint_sparsity(), self.context().clone());
        self.sens_adjoint_inplace(t, &mut y);
        y
    }

    /// Compute the negative transpose of the gradient of the operator wrt a parameter vector p and store it in the matrix `y`.
    /// `y` should have been previously initialised using the output of [Self::sens_adjoint_sparsity].
    /// The default implementation of this method computes the gradient using [Self::sens_transpose_mul_inplace],
    /// but it can be overriden for more efficient implementations.
    fn sens_adjoint_inplace(&self, t: Self::T, y: &mut Self::M) {
        self._default_sens_adjoint_inplace(t, y);
    }

    /// Default implementation of the gradient computation (this is the default for [Self::sens_adjoint_inplace]).
    fn _default_sens_adjoint_inplace(&self, t: Self::T, y: &mut Self::M) {
        let mut v = Self::V::zeros(self.nstates(), self.context().clone());
        let mut col = Self::V::zeros(self.nout(), self.context().clone());
        for j in 0..self.nstates() {
            v.set_index(j, Self::T::one());
            self.sens_transpose_mul_inplace(t, &v, &mut col);
            y.set_column(j, &col);
            v.set_index(j, Self::T::zero());
        }
    }

    fn sens_adjoint_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        None
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        context::nalgebra::NalgebraContext,
        matrix::dense_nalgebra_serial::NalgebraMat, ConstantOp, ConstantOpSens,
        ConstantOpSensAdjoint, DenseMatrix, Matrix, Op, Vector,
    };

    type M = NalgebraMat<f64>;

    struct FakeConstantOp {
        ctx: NalgebraContext,
    }

    impl FakeConstantOp {
        fn new() -> Self {
            Self {
                ctx: NalgebraContext,
            }
        }
    }

    impl Op for FakeConstantOp {
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

    impl ConstantOp for FakeConstantOp {
        fn call_inplace(&self, t: Self::T, y: &mut Self::V) {
            y.copy_from(&Self::V::from_vec(vec![1.0 + t, 2.0 - t], self.ctx));
        }
    }

    impl ConstantOpSens for FakeConstantOp {
        fn sens_mul_inplace(&self, _t: Self::T, v: &Self::V, y: &mut Self::V) {
            y.copy_from(&Self::V::from_vec(
                vec![v.get_index(0) + 2.0 * v.get_index(1), 3.0 * v.get_index(0) - v.get_index(1)],
                self.ctx,
            ));
        }
    }

    impl ConstantOpSensAdjoint for FakeConstantOp {
        fn sens_transpose_mul_inplace(&self, _t: Self::T, v: &Self::V, y: &mut Self::V) {
            y.copy_from(&Self::V::from_vec(
                vec![-v.get_index(0) - 3.0 * v.get_index(1), -2.0 * v.get_index(0) + v.get_index(1)],
                self.ctx,
            ));
        }
    }

    #[test]
    fn constant_op_default_helpers_construct_expected_outputs() {
        let op = FakeConstantOp::new();
        let value = op.call(0.25);
        value.assert_eq_st(
            &crate::NalgebraVec::from_vec(vec![1.25, 1.75], NalgebraContext),
            1e-12,
        );

        let sens = op.sens(0.0);
        assert_eq!(sens.get_index(0, 0), 1.0);
        assert_eq!(sens.get_index(1, 0), 3.0);
        assert_eq!(sens.get_index(0, 1), 2.0);
        assert_eq!(sens.get_index(1, 1), -1.0);

        let mut sens_inplace = M::zeros(2, 2, NalgebraContext);
        op.sens_inplace(0.0, &mut sens_inplace);
        assert_eq!(sens_inplace.get_index(0, 0), 1.0);
        assert_eq!(sens_inplace.get_index(1, 0), 3.0);
        assert_eq!(sens_inplace.get_index(0, 1), 2.0);
        assert_eq!(sens_inplace.get_index(1, 1), -1.0);

        let sens_adj = op.sens_adjoint(0.0);
        assert_eq!(sens_adj.get_index(0, 0), -1.0);
        assert_eq!(sens_adj.get_index(1, 0), -2.0);
        assert_eq!(sens_adj.get_index(0, 1), -3.0);
        assert_eq!(sens_adj.get_index(1, 1), 1.0);

        let mut sens_adj_inplace = M::zeros(2, 2, NalgebraContext);
        op.sens_adjoint_inplace(0.0, &mut sens_adj_inplace);
        assert_eq!(sens_adj_inplace.get_index(0, 0), -1.0);
        assert_eq!(sens_adj_inplace.get_index(1, 0), -2.0);
        assert_eq!(sens_adj_inplace.get_index(0, 1), -3.0);
        assert_eq!(sens_adj_inplace.get_index(1, 1), 1.0);
    }
}
