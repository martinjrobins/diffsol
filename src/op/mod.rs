use crate::{Matrix, Scalar, Vector};

use num_traits::{One, Zero};
use serde::Serialize;

pub mod bdf;
pub mod closure;
pub mod constant_closure;
pub mod filter;
pub mod linear_closure;
pub mod linearise;
pub mod matrix;
pub mod sdirk;
pub mod unit;

pub trait Op {
    type T: Scalar;
    type V: Vector<T = Self::T>;
    type M: Matrix<T = Self::T, V = Self::V>;
    fn nstates(&self) -> usize;
    fn nout(&self) -> usize;
    fn nparams(&self) -> usize;

    /// Return sparsity information for the jacobian or matrix (if available)
    fn sparsity(&self) -> Option<&<Self::M as Matrix>::Sparsity> {
        None
    }

    fn statistics(&self) -> OpStatistics {
        OpStatistics::default()
    }
}

#[derive(Default, Clone, Serialize)]
pub struct OpStatistics {
    pub number_of_calls: usize,
    pub number_of_jac_muls: usize,
    pub number_of_matrix_evals: usize,
}

impl OpStatistics {
    pub fn new() -> Self {
        Self {
            number_of_jac_muls: 0,
            number_of_calls: 0,
            number_of_matrix_evals: 0,
        }
    }

    pub fn increment_call(&mut self) {
        self.number_of_calls += 1;
    }

    pub fn increment_jac_mul(&mut self) {
        self.number_of_jac_muls += 1;
    }

    pub fn increment_matrix(&mut self) {
        self.number_of_matrix_evals += 1;
    }
}

// NonLinearOp is a trait for non-linear operators. It extends the Op trait with methods for
// computing the operator and its Jacobian. The operator is defined by the call_inplace method,
// which computes the operator at a given state and time. The Jacobian is defined by the
// jac_mul_inplace method, which computes the product of the Jacobian with a given vector.
pub trait NonLinearOp: Op {
    /// Compute the operator at a given state and time.
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V);

    /// Compute the product of the Jacobian with a given vector.
    fn jac_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V);
    fn call(&self, x: &Self::V, t: Self::T) -> Self::V {
        let mut y = Self::V::zeros(self.nout());
        self.call_inplace(x, t, &mut y);
        y
    }
    fn jac_mul(&self, x: &Self::V, t: Self::T, v: &Self::V) -> Self::V {
        let mut y = Self::V::zeros(self.nstates());
        self.jac_mul_inplace(x, t, v, &mut y);
        y
    }

    /// Compute the Jacobian of the operator and store it in the matrix `y`.
    /// `y` should have been previously initialised using the output of [`Self::sparsity`].
    fn jacobian_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        self._default_jacobian_inplace(x, t, y);
    }

    fn _default_jacobian_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        let mut v = Self::V::zeros(self.nstates());
        let mut col = Self::V::zeros(self.nout());
        for j in 0..self.nstates() {
            v[j] = Self::T::one();
            self.jac_mul_inplace(x, t, &v, &mut col);
            y.set_column(j, &col);
            v[j] = Self::T::zero();
        }
    }

    fn jacobian(&self, x: &Self::V, t: Self::T) -> Self::M {
        let n = self.nstates();
        let mut y = Self::M::new_from_sparsity(n, n, self.sparsity());
        self.jacobian_inplace(x, t, &mut y);
        y
    }
}

pub trait LinearOp: Op {
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        let beta = Self::T::zero();
        self.gemv_inplace(x, t, beta, y);
    }

    fn gemv_inplace(&self, x: &Self::V, t: Self::T, beta: Self::T, y: &mut Self::V);

    fn matrix(&self, t: Self::T) -> Self::M {
        let mut y = Self::M::new_from_sparsity(self.nstates(), self.nstates(), self.sparsity());
        self.matrix_inplace(t, &mut y);
        y
    }

    fn matrix_inplace(&self, t: Self::T, y: &mut Self::M) {
        self._default_matrix_inplace(t, y);
    }

    fn _default_matrix_inplace(&self, t: Self::T, y: &mut Self::M) {
        let mut v = Self::V::zeros(self.nstates());
        let mut col = Self::V::zeros(self.nout());
        for j in 0..self.nstates() {
            v[j] = Self::T::one();
            self.call_inplace(&v, t, &mut col);
            y.set_column(j, &col);
            v[j] = Self::T::zero();
        }
    }
}

pub trait ConstantOp: Op {
    fn call_inplace(&self, t: Self::T, y: &mut Self::V);
    fn call(&self, t: Self::T) -> Self::V {
        let mut y = Self::V::zeros(self.nout());
        self.call_inplace(t, &mut y);
        y
    }
    fn jac_mul_inplace(&self, y: &mut Self::V) {
        let zeros = Self::V::zeros(self.nout());
        y.copy_from(&zeros);
    }
}

impl<C: Op> Op for &C {
    type T = C::T;
    type V = C::V;
    type M = C::M;
    fn nstates(&self) -> usize {
        C::nstates(*self)
    }
    fn nout(&self) -> usize {
        C::nout(*self)
    }
    fn nparams(&self) -> usize {
        C::nparams(*self)
    }
}

//impl <C: LinearOp> NonLinearOp for C {
//    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
//        C::call_inplace(self, x, t, y)
//    }
//    fn jac_mul_inplace(&self, _x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
//        C::call_inplace(self, v, t, y)
//    }
//    fn jacobian_inplace(&self, _x: &Self::V, t: Self::T, y: &mut Self::M) {
//        C::matrix_inplace(self, t, y)
//    }
//    fn sparsity(&self) -> Option<&<Self::M as Matrix>::Sparsity> {
//        C::sparsity(self)
//    }
//
//}

impl<C: NonLinearOp> NonLinearOp for &C {
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        C::call_inplace(*self, x, t, y)
    }
    fn jac_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        C::jac_mul_inplace(*self, x, t, v, y)
    }
    fn jacobian_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        C::jacobian_inplace(*self, x, t, y)
    }
}

impl<C: LinearOp> LinearOp for &C {
    fn gemv_inplace(&self, x: &Self::V, t: Self::T, beta: Self::T, y: &mut Self::V) {
        C::gemv_inplace(*self, x, t, beta, y)
    }
}
