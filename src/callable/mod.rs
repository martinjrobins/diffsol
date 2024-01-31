use crate::{VectorRef, Matrix, Vector};
use num_traits::{One, Zero};

pub mod closure;
pub mod ode;
pub mod unit;
pub mod filter;

pub trait Callable<V: Vector> 
{
    fn call(&self, x: &V, p: &V, y: &mut V);
    fn gemv(&self, x: &V, p: &V, alpha: V::T, beta: V::T, y: &mut V) {
        let mut beta_y = y.clone();
        beta_y.mul_assign(beta);
        self.call(x, p, y);
        y.mul_assign(alpha);
        y.add_assign(beta_y);
    }
    fn nstates(&self) -> usize;
    fn nout(&self) -> usize;
    fn nparams(&self) -> usize;
    fn jacobian_action(&self, x: &V, p: &V, v: &V, y: &mut V);
}

pub trait Jacobian<M: Matrix>: Callable<M::V> 
{
    fn jacobian(&self, x: &M::V, p: &M::V) -> M {
        let mut v = M::V::zeros(self.nstates());
        let mut col = M::V::zeros(self.nout());
        let mut triplets = Vec::with_capacity(self.nstates());
        for j in 0..self.nstates() {
            v[j] = M::T::one();
            self.jacobian_action(x, p, &v, &mut col);
            for i in 0..self.nout() {
                if col[i] != M::T::zero() {
                    triplets.push((i, j, col[i]));
                }
            }
            v[j] = M::T::zero();
        }
        M::try_from_triplets(self.nstates(), self.nout(), triplets).unwrap()
    }
}


