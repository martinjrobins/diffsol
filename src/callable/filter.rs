
// a callable that takes another callable and a mask vector
// this callable, when called, will call the other callable with the mask applied

use std::cell::RefCell;

use crate::{Matrix, VectorRef, Vector, VectorIndex};

use super::{Callable, Jacobian};

pub struct FilterCallable<'a, V: Vector, C: Callable<V>> 
{
    callable: &'a C,
    indices: <V as Vector>::Index,
    y_full: RefCell<V>,
    x_full: RefCell<V>,
}

impl<'a, V: Vector, C: Callable<V>> FilterCallable<'a, V, C> 
{
    pub fn new(callable: &'a C, x: &V, indices: <V as Vector>::Index) -> Self {
        if callable.nstates() != indices.len() {
            panic!("FilterCallable::new() called with callable with different number of states");
        }
        let y_full = RefCell::new(V::zeros(callable.nout()));
        let x_full = RefCell::new(x.clone());
        Self { callable, indices, y_full, x_full }
    }
}

impl<'a, V: Vector, C: Callable<V>> Callable<V> for FilterCallable<'a, V, C> 
{
    fn call(&self, x: &V, p: &V, y: &mut V) {
        let mut y_full = self.y_full.borrow_mut();
        let mut x_full = self.x_full.borrow_mut();
        x_full.scatter_from(x, &self.indices);
        self.callable.call(&x_full, p, &mut y_full);
        y.gather_from(&y_full, &self.indices);
    }
    fn jacobian_action(&self, x: &V, p: &V, v: &V, y: &mut V) {
        let mut y_full = self.y_full.borrow_mut();
        let mut x_full = self.x_full.borrow_mut();
        x_full.scatter_from(x, &self.indices);
        self.callable.jacobian_action(&x_full, p, v, &mut y_full);
        y.gather_from(&y_full, &self.indices);
    }
    fn nstates(&self) -> usize {
        self.indices.len()
    }
    fn nout(&self) -> usize {
        self.indices.len()
    }
    fn nparams(&self) -> usize {
        self.callable.nparams()
    }
}

impl <'a, C: Callable<M::V> + Jacobian<M>, M: Matrix> Jacobian<M> for FilterCallable<'a, M::V, C> 
{}