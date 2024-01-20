// unit is a callable that returns returns the input vector

use crate::{Scalar, vector::Vector, matrix::Matrix};

use super::{Callable, Jacobian};


pub struct UnitCallable<T: Scalar, V: Vector<T>> {
    n: usize,
    ones: V,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar, V: Vector<T>> Default for UnitCallable<T, V> {
    fn default() -> Self {
        Self::new(1)
    }
}

impl<T: Scalar, V: Vector<T>> UnitCallable<T, V> {
    pub fn new(n: usize) -> Self {
        let mut ones = V::zeros(n);
        ones.add_scalar_mut(T::one());
        Self { n, ones, _phantom: std::marker::PhantomData }
    }
}

impl<T: Scalar, V: Vector<T>> Callable<T, V> for UnitCallable<T, V> {
    fn call(&self, x: &V, _p: &V, y: &mut V) {
        y.copy_from(&x)
    }
    fn jacobian_action(&self, _x: &V, _p: &V, _v: &V, y: &mut V) {
        y.copy_from(&self.ones); 
    }
    fn nstates(&self) -> usize {
        self.n
    }
    fn nout(&self) -> usize {
        self.n
    }
    fn nparams(&self) -> usize {
        0
    }
}

// implement Jacobian
impl<T: Scalar, V: Vector<T>, M: Matrix<T, V>> Jacobian<T, V, M> for UnitCallable<T, V> {}

