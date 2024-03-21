use crate::{Matrix, Scalar, Vector};
use num_traits::{One, Zero};
use std::ops::{AddAssign, MulAssign};

pub mod closure;
pub mod constant_closure;
pub mod filter;
pub mod linear_closure;
pub mod linearise;
pub mod ode;
pub mod ode_rhs;
pub mod unit;

pub trait Op {
    type T: Scalar;
    type V: Vector<T = Self::T>;
    type M: Matrix<T = Self::T, V = Self::V>;
    fn nstates(&self) -> usize;
    fn nout(&self) -> usize;
    fn nparams(&self) -> usize;
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
    fn jacobian(&self, x: &Self::V, t: Self::T) -> Self::M {
        let mut v = Self::V::zeros(self.nstates());
        let mut col = Self::V::zeros(self.nout());
        let mut triplets = Vec::with_capacity(self.nstates());
        for j in 0..self.nstates() {
            v[j] = Self::T::one();
            self.jac_mul_inplace(x, t, &v, &mut col);
            for i in 0..self.nout() {
                if col[i] != Self::T::zero() {
                    triplets.push((i, j, col[i]));
                }
            }
            v[j] = Self::T::zero();
        }
        Self::M::try_from_triplets(self.nstates(), self.nout(), triplets).unwrap()
    }
}

pub trait LinearOp: Op {
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V);
    fn call(&self, x: &Self::V, t: Self::T) -> Self::V {
        let mut y = Self::V::zeros(self.nout());
        self.call_inplace(x, t, &mut y);
        y
    }
    fn gemv(&self, x: &Self::V, t: Self::T, alpha: Self::T, beta: Self::T, y: &mut Self::V) {
        let mut beta_y = y.clone();
        beta_y.mul_assign(beta);
        self.call_inplace(x, t, y);
        y.mul_assign(alpha);
        y.add_assign(&beta_y);
    }
    fn jacobian_diagonal(&self, t: Self::T) -> Self::V {
        let mut v = Self::V::zeros(self.nstates());
        let mut col = Self::V::zeros(self.nout());
        let mut diag = Self::V::zeros(self.nstates());
        for j in 0..self.nstates() {
            v[j] = Self::T::one();
            self.call_inplace(&v, t, &mut col);
            diag[j] = col[j];
            v[j] = Self::T::zero();
        }
        diag
    }
    fn jacobian(&self, t: Self::T) -> Self::M {
        let mut v = Self::V::zeros(self.nstates());
        let mut col = Self::V::zeros(self.nout());
        let mut triplets = Vec::with_capacity(self.nstates());
        for j in 0..self.nstates() {
            v[j] = Self::T::one();
            self.call_inplace(&v, t, &mut col);
            for i in 0..self.nout() {
                if col[i] != Self::T::zero() {
                    triplets.push((i, j, col[i]));
                }
            }
            v[j] = Self::T::zero();
        }
        Self::M::try_from_triplets(self.nstates(), self.nout(), triplets).unwrap()
    }
}

impl<C: LinearOp> NonLinearOp for C {
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        C::call_inplace(self, x, t, y)
    }
    fn jac_mul_inplace(&self, _x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        C::call_inplace(self, v, t, y)
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
