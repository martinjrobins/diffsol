
// a callable that takes another callable and a mask vector
// this callable, when called, will call the other callable with the mask applied

use std::{cell::RefCell, rc::Rc};

use crate::{Vector, VectorIndex};

use super::{Jacobian, NonLinearOp, Op};

pub struct FilterCallable<C: NonLinearOp> 
{
    callable: Rc<C>,
    indices: <C::V as Vector>::Index,
    y_full: RefCell<C::V>,
    x_full: RefCell<C::V>,
}

impl<C: NonLinearOp> FilterCallable<C> 
{
    pub fn new(callable: Rc<C>, x: &C::V, indices: <C::V as Vector>::Index) -> Self {
        let y_full = RefCell::new(C::V::zeros(callable.nout()));
        let x_full = RefCell::new(x.clone());
        Self { callable, indices, y_full, x_full }
    }
}

impl<C: NonLinearOp> Op for FilterCallable<C> 
{
    type V = C::V;
    type T = C::T;
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

impl<C: NonLinearOp> NonLinearOp for FilterCallable<C> 
{
    fn call_inplace(&self, x: &Self::V, p: &Self::V, y: &mut Self::V) {
        let mut y_full = self.y_full.borrow_mut();
        let mut x_full = self.x_full.borrow_mut();
        x_full.scatter_from(x, &self.indices);
        self.callable.call_inplace(&x_full, p, &mut y_full);
        y.gather_from(&y_full, &self.indices);
    }
    fn jac_mul_inplace(&self, x: &Self::V, p: &Self::V, v: &Self::V, y: &mut Self::V) {
        let mut y_full = self.y_full.borrow_mut();
        let mut x_full = self.x_full.borrow_mut();
        x_full.scatter_from(x, &self.indices);
        self.callable.jac_mul_inplace(&x_full, p, v, &mut y_full);
        y.gather_from(&y_full, &self.indices);
    }
}

impl <C: Jacobian> Jacobian for FilterCallable<C> 
{
    type M = C::M;
}