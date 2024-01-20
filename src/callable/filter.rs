
// a callable that takes another callable and a mask vector
// this callable, when called, will call the other callable with the mask applied

use std::cell::RefCell;

use crate::{Scalar, vector::Vector, matrix::Matrix};

use super::{Callable, Jacobian};


pub struct GatherCallable<'a, T: Scalar, V: Vector<T>, C: Callable<T, V>> {
    callable: &'a C,
    indices: V,
    y_full: RefCell<V>,
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T: Scalar, V: Vector<T>, C: Callable<T, V>> GatherCallable<'a, T, V, C> {
    pub fn new(callable: &'a C, indices: V) -> Self {
        if callable.nstates() != mask.len() {
            panic!("FilterCallable::new() called with callable with different number of states");
        }
        let y_full = RefCell::new(V::zeros(callable.nout()));
        Self { callable, mask, y_full, _phantom: std::marker::PhantomData }
    }
}

impl<'a, T: Scalar, V: Vector<T>, C: Callable<T, V>> Callable<T, V> for GatherCallable<'a, T, V, C> {
    fn call(&self, x: &V, p: &V, y: &mut V) {
        let mut y_full = self.y_full.borrow_mut();
        self.callable.call(&x, p, &mut y_full);
        y.gather_from(&y_full);
    }
    fn jacobian_action(&self, x: &V, p: &V, v: &V, y: &mut V) {
        let mut x_masked = V::zeros(self.callable.nstates());
        for i in 0..self.callable.nstates() {
            x_masked[i] = x[i] * self.mask[i];
        }
        self.callable.jacobian_action(&x_masked, p, v, y)
    }
    fn nstates(&self) -> usize {
        self.callable.nstates()
    }
    fn nout(&self) -> usize {
        self.callable.nout()
    }
    fn nparams(&self) -> usize {
        self.callable.nparams()
    }
}