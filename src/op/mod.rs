use crate::{
    Matrix, Scalar, Vector,
};

use num_traits::{One, Zero};

pub mod bdf;
pub mod closure;
pub mod constant_closure;
pub mod filter;
pub mod linear_closure;
pub mod linearise;
pub mod matrix;
pub mod ode_rhs;
pub mod sdirk;
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

    /// Compute the Jacobian of the operator and store it in the matrix `y`. 
    /// `y` should have been previously initialised using the output of [`Self::sparsity`].
    fn jacobian_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        let mut v = Self::V::zeros(self.nstates());
        let mut col = Self::V::zeros(self.nout());
        for j in 0..self.nstates() {
            v[j] = Self::T::one();
            self.jac_mul_inplace(x, t, &v, &mut col);
            y.set_column(j, &col);
            v[j] = Self::T::zero();
        }
    }

    /// Return the sparsity pattern of the Jacobian matrix. This should not vary with t or x
    fn sparsity(&self) -> &<Self::M as Matrix>::Sparsity;
}

pub trait LinearOp: Op {
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        let beta = Self::T::zero();
        self.gemv_inplace(x, t, beta, y);
    }

    fn gemv_inplace(&self, x: &Self::V, t: Self::T, beta: Self::T, y: &mut Self::V);

    fn matrix(&self, t: Self::T) -> Self::M {
        let mut y = Self::M::zeros(self.nstates(), self.nout());
        self.matrix_inplace(t, &mut y);
        y
    }
    
    fn matrix_inplace(&self, t: Self::T, y: &mut Self::M) {
        let mut v = Self::V::zeros(self.nstates());
        let mut col = Self::V::zeros(self.nout());
        for j in 0..self.nstates() {
            v[j] = Self::T::one();
            self.call_inplace(&v, t, &mut col);
            y.set_column(j, &col);
            v[j] = Self::T::zero();
        }
    }

    /// Return the sparsity pattern of the Jacobian matrix. This should not vary with t or x
    fn sparsity(&self) -> &<Self::M as Matrix>::Sparsity;
}

impl<C: LinearOp> NonLinearOp for C {
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        C::call_inplace(self, x, t, y)
    }
    fn jac_mul_inplace(&self, _x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        C::call_inplace(self, v, t, y)
    }
    fn sparsity(&self) -> &<Self::M as Matrix>::Sparsity {
        C::sparsity(self)
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
