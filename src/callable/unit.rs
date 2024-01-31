// unit is a callable that returns returns the input vector

use crate::{VectorRef, Matrix, Vector};
use num_traits::{One, Zero};

use super::{Callable, Jacobian};


pub struct UnitCallable<V: Vector> {
    n: usize,
    ones: V,
    _phantom: std::marker::PhantomData<V::T>,
}

impl<V: Vector> Default for UnitCallable<V> {
    fn default() -> Self {
        Self::new(1)
    }
}

impl<V: Vector> UnitCallable<V> {
    pub fn new(n: usize) -> Self {
        let mut ones = V::zeros(n);
        ones.add_scalar_mut(V::T::one());
        Self { n, ones, _phantom: std::marker::PhantomData }
    }
}

impl<V: Vector> Callable<V> for UnitCallable<V> 
{
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
impl<M: Matrix> Jacobian<M> for UnitCallable<M::V> 
{}

