use crate::{Scalar, Vector, Matrix};

pub mod closure;
pub mod ode;


pub trait Callable<T: Scalar, V: Vector<T>> {
    fn call(&self, x: &V, p: &V, y: &mut V);
    fn nstates(&self) -> usize;
    fn nparams(&self) -> usize;
    fn jacobian_action(&self, x: &V, p: &V, v: &V, y: &mut V);
    fn jacobian<M: Matrix<T, V>>(&self, x: &V, p: &V) -> M {
        let mut v = V::zeros(x.len());
        let mut col = V::zeros(x.len());
        let mut triplets = Vec::with_capacity(x.len());
        for j in 0..x.len() {
            v[j] = T::one();
            self.jacobian_action(x, p, &v, &mut col);
            for i in 0..x.len() {
                if col[i] != T::zero() {
                    triplets.push((i, j, col[i]));
                }
            }
            v[j] = T::zero();
        }
        M::try_from_triplets(x.len(), x.len(), triplets).unwrap()
    }

}

