use crate::{Matrix, Scalar, Vector};
use num_traits::{One, Zero};
use std::ops::{MulAssign, AddAssign};


pub mod closure;
pub mod ode;
pub mod unit;
pub mod filter;
pub mod linearise;
pub mod constant_closure;
pub mod linear_closure;

pub trait Op {
    type T: Scalar;
    type V: Vector<T = Self::T>;
    fn nstates(&self) -> usize;
    fn nout(&self) -> usize;
    fn nparams(&self) -> usize;
}

pub trait NonLinearOp: Op {
    fn call_inplace(&self, x: &Self::V, p: &Self::V, t: Self::T, y: &mut Self::V);
    fn jac_mul_inplace(&self, x: &Self::V, p: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V);
    fn call(&self, x: &Self::V, p: &Self::V, t: Self::T) -> Self::V {
        let mut y = Self::V::zeros(self.nout());
        self.call_inplace(x, p, t, &mut y);
        y
    }
    fn jac_mul(&self, x: &Self::V, p: &Self::V, t: Self::T, v: &Self::V) -> Self::V {
        let mut y = Self::V::zeros(self.nstates());
        self.jac_mul_inplace(x, p, t, v, &mut y);
        y
    }
}

pub trait LinearOp: Op {
    fn call_inplace(&self, x: &Self::V, p: &Self::V, t: Self::T, y: &mut Self::V);
    fn call(&self, x: &Self::V, p: &Self::V, t: Self::T) -> Self::V {
        let mut y = Self::V::zeros(self.nout());
        self.call_inplace(x, p, t, &mut y);
        y
    }
    fn gemv(&self, x: &Self::V, p: &Self::V, t: Self::T, alpha: Self::T, beta: Self::T, y: &mut Self::V) 
    {
        let mut beta_y = y.clone();
        beta_y.mul_assign(beta);
        self.call_inplace(x, p, t, y);
        y.mul_assign(alpha);
        y.add_assign(&beta_y);
    }
    fn jacobian_diagonal(&self, p: &Self::V, t: Self::T) -> Self::V {
        let mut v = Self::V::zeros(self.nstates());
        let mut col = Self::V::zeros(self.nout());
        let mut diag = Self::V::zeros(self.nstates());
        for j in 0..self.nstates() {
            v[j] = Self::T::one();
            self.call_inplace(&v, p, t, &mut col);
            diag[j] = col[j];
            v[j] = Self::T::zero();
        }
        diag
    }
}

impl<C: LinearOp> NonLinearOp for C {
    fn call_inplace(&self, x: &Self::V, p: &Self::V, t: Self::T, y: &mut Self::V) {
        C::call_inplace(self, x, p, t, y)
    }
    fn jac_mul_inplace(&self, _x: &Self::V, p: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        C::call_inplace(self, v, p, t, y)
    }
    
}

pub trait ConstantOp: Op {
    fn call_inplace(&self, p: &Self::V, t: Self::T, y: &mut Self::V);
    fn call(&self, p: &Self::V, t: Self::T) -> Self::V {
        let mut y = Self::V::zeros(self.nout());
        self.call_inplace(p, t, &mut y);
        y
    }
    fn jac_mul_inplace(&self, y: &mut Self::V) {
        let zeros = Self::V::zeros(self.nout());
        y.copy_from(&zeros);
    }
}


pub trait ConstantJacobian: LinearOp
{
    type M: Matrix<T = Self::T, V = Self::V>;
    fn jacobian(&self, p: &Self::V, t: Self::T) -> Self::M {
        let mut v = Self::V::zeros(self.nstates());
        let mut col = Self::V::zeros(self.nout());
        let mut triplets = Vec::with_capacity(self.nstates());
        for j in 0..self.nstates() {
            v[j] = Self::T::one();
            self.call_inplace(&v, p, t, &mut col);
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

pub trait Jacobian: NonLinearOp
{
    type M: Matrix<T = Self::T, V = Self::V>;
    fn jacobian(&self, x: &Self::V, p: &Self::V, t: Self::T) -> Self::M {
        let mut v = Self::V::zeros(self.nstates());
        let mut col = Self::V::zeros(self.nout());
        let mut triplets = Vec::with_capacity(self.nstates());
        for j in 0..self.nstates() {
            v[j] = Self::T::one();
            self.jac_mul_inplace(x, p, t, &v, &mut col);
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


