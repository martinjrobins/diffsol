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

#[cfg(test)]
mod tests {
    use crate::{
        context::nalgebra::NalgebraContext, matrix::dense_nalgebra_serial::NalgebraMat,
        matrix::Matrix, DenseMatrix, LinearOp, LinearOpSens, LinearOpTranspose, Op, Vector,
    };

    type M = NalgebraMat<f64>;

    struct FakeLinearOp {
        ctx: NalgebraContext,
    }

    impl Op for FakeLinearOp {
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

    impl LinearOp for FakeLinearOp {
        fn gemv_inplace(&self, x: &Self::V, _t: Self::T, beta: Self::T, y: &mut Self::V) {
            let out = Self::V::from_vec(
                vec![2.0 * x.get_index(0) + 3.0 * x.get_index(1), -x.get_index(0) + 4.0 * x.get_index(1)],
                NalgebraContext,
            );
            y.axpy(1.0, &out, beta);
        }
    }

    impl LinearOpTranspose for FakeLinearOp {
        fn gemv_transpose_inplace(&self, x: &Self::V, _t: Self::T, beta: Self::T, y: &mut Self::V) {
            let out = Self::V::from_vec(
                vec![2.0 * x.get_index(0) - x.get_index(1), 3.0 * x.get_index(0) + 4.0 * x.get_index(1)],
                NalgebraContext,
            );
            y.axpy(1.0, &out, beta);
        }
    }

    impl LinearOpSens for FakeLinearOp {
        fn sens_mul_inplace(&self, _x: &Self::V, _t: Self::T, v: &Self::V, y: &mut Self::V) {
            y.copy_from(&Self::V::from_vec(
                vec![v.get_index(0) + 2.0 * v.get_index(1), 3.0 * v.get_index(0) + 4.0 * v.get_index(1)],
                NalgebraContext,
            ));
        }
    }

    #[test]
    fn linear_op_default_helpers_construct_expected_outputs() {
        let op = FakeLinearOp {
            ctx: NalgebraContext,
        };
        let x = crate::NalgebraVec::from_vec(vec![1.0, 2.0], NalgebraContext);
        let v = crate::NalgebraVec::from_vec(vec![3.0, -1.0], NalgebraContext);
        let mut y = crate::NalgebraVec::from_vec(vec![1.0, 1.0], NalgebraContext);

        op.call_inplace(&x, 0.0, &mut y);
        y.assert_eq_st(
            &crate::NalgebraVec::from_vec(vec![8.0, 7.0], NalgebraContext),
            1e-12,
        );

        let matrix = op.matrix(0.0);
        assert_eq!(matrix.get_index(0, 0), 2.0);
        assert_eq!(matrix.get_index(1, 0), -1.0);
        assert_eq!(matrix.get_index(0, 1), 3.0);
        assert_eq!(matrix.get_index(1, 1), 4.0);

        let mut transpose = M::zeros(2, 2, NalgebraContext);
        op.transpose_inplace(0.0, &mut transpose);
        assert_eq!(transpose.get_index(0, 0), 2.0);
        assert_eq!(transpose.get_index(1, 0), 3.0);
        assert_eq!(transpose.get_index(0, 1), -1.0);
        assert_eq!(transpose.get_index(1, 1), 4.0);

        let mut transpose_call = crate::NalgebraVec::zeros(2, NalgebraContext);
        op.call_transpose_inplace(&x, 0.0, &mut transpose_call);
        transpose_call.assert_eq_st(
            &crate::NalgebraVec::from_vec(vec![0.0, 11.0], NalgebraContext),
            1e-12,
        );

        let sens = op.sens(&x, 0.0);
        assert_eq!(sens.get_index(0, 0), 1.0);
        assert_eq!(sens.get_index(1, 0), 3.0);
        assert_eq!(sens.get_index(0, 1), 2.0);
        assert_eq!(sens.get_index(1, 1), 4.0);

        op.sens_mul(&x, 0.0, &v)
            .assert_eq_st(
                &crate::NalgebraVec::from_vec(vec![1.0, 5.0], NalgebraContext),
                1e-12,
            );
    }
}
