use crate::{Scalar, Vector, Matrix};

pub mod closure;
pub mod ode;
pub mod unit;
pub mod filter;

pub trait Callable<T: Scalar, V: Vector<T>> {
    fn call(&self, x: &V, p: &V, y: &mut V);
    fn gemv(&self, x: &V, p: &V, alpha: T, beta: T, y: &mut V) {
        let beta_y = *y * beta;
        self.call(x, p, y);
        *y *= alpha;
        *y += &beta_y;
    }
    fn nstates(&self) -> usize;
    fn nout(&self) -> usize;
    fn nparams(&self) -> usize;
    fn jacobian_action(&self, x: &V, p: &V, v: &V, y: &mut V);
}

pub trait Jacobian<T: Scalar, V: Vector<T>, M: Matrix<T, V>>: Callable<T, V> {
    fn jacobian(&self, x: &V, p: &V) -> M {
        let mut v = V::zeros(self.nstates());
        let mut col = V::zeros(self.nout());
        let mut triplets = Vec::with_capacity(self.nstates());
        for j in 0..self.nstates() {
            v[j] = T::one();
            self.jacobian_action(x, p, &v, &mut col);
            for i in 0..self.nout() {
                if col[i] != T::zero() {
                    triplets.push((i, j, col[i]));
                }
            }
            v[j] = T::zero();
        }
        M::try_from_triplets(self.nstates(), self.nout(), triplets).unwrap()
    }
}

pub trait Diagonal<T: Scalar, V: Vector<T>>: Callable<T, V> {
    fn diagonal(&self, x: &V, p: &V) -> V {
        let mut v = V::zeros(self.nstates());
        let mut col = V::zeros(self.nout());
        let mut diag = V::zeros(self.nstates());
        for j in 0..self.nstates() {
            v[j] = T::one();
            self.jacobian_action(x, p, &v, &mut col);
            diag[j] = col[j];
            v[j] = T::zero();
        }
        diag
    }
}
