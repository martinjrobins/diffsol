use crate::{Matrix, Scalar, Vector};
use num_traits::{One, Zero};
use std::ops::{AddAssign, MulAssign};

pub mod closure;
pub mod ode;
pub mod unit;
pub mod filter;

pub trait Callable
{
    type V: Vector<T = Self::T>;
    type T: Scalar;
    fn call(&self, x: &Self::V, p: &Self::V, y: &mut Self::V);
    fn gemv(&self, x: &Self::V, p: &Self::V, alpha: Self::T, beta: Self::T, y: &mut Self::V) {
        let mut beta_y = y.clone();
        beta_y.mul_assign(beta);
        self.call(x, p, y);
        y.mul_assign(alpha);
        y.add_assign(beta_y);
    }
    fn nstates(&self) -> usize;
    fn nout(&self) -> usize;
    fn nparams(&self) -> usize;
    fn jacobian_action(&self, x: &Self::V, p: &Self::V, v: &Self::V, y: &mut Self::V);
}

pub trait Jacobian: Callable
{
    type M: Matrix<T = Self::T, V = Self::V>;
    fn jacobian(&self, x: &Self::V, p: &Self::V) -> Self::M {
        let mut v = Self::V::zeros(self.nstates());
        let mut col = Self::V::zeros(self.nout());
        let mut triplets = Vec::with_capacity(self.nstates());
        for j in 0..self.nstates() {
            v[j] = Self::T::one();
            self.jacobian_action(x, p, &v, &mut col);
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


