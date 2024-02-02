
// a callable that takes another callable and a mask vector
// this callable, when called, will call the other callable with the mask applied

use std::{cell::RefCell, rc::Rc};

use crate::{Vector, VectorIndex};

use super::{Callable, Jacobian};

pub struct FilterCallable<C: Callable> 
{
    callable: Rc<C>,
    indices: <C::V as Vector>::Index,
    y_full: RefCell<C::V>,
    x_full: RefCell<C::V>,
}

impl<C: Callable> FilterCallable<C> 
{
    pub fn new(callable: Rc<C>, x: &C::V, indices: <C::V as Vector>::Index) -> Self {
        if callable.nstates() != indices.len() {
            panic!("FilterCallable::new() called with callable with different number of states");
        }
        let y_full = RefCell::new(C::V::zeros(callable.nout()));
        let x_full = RefCell::new(x.clone());
        Self { callable, indices, y_full, x_full }
    }
}

impl<C: Callable> Callable for FilterCallable<C> 
{
    type V = C::V;
    type T = C::T;
    fn call(&self, x: &Self::V, p: &Self::V, y: &mut Self::V) {
        let mut y_full = self.y_full.borrow_mut();
        let mut x_full = self.x_full.borrow_mut();
        x_full.scatter_from(x, &self.indices);
        self.callable.call(&x_full, p, &mut y_full);
        y.gather_from(&y_full, &self.indices);
    }
    fn jacobian_action(&self, x: &Self::V, p: &Self::V, v: &Self::V, y: &mut Self::V) {
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

impl <C: Callable + Jacobian> Jacobian for FilterCallable<C> 
{
    type M = C::M;
}