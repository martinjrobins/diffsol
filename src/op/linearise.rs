use num_traits::One;
use std::{cell::RefCell, rc::Rc};

use crate::{Vector, Matrix};

use super::{LinearOp, NonLinearOp, Op};

pub struct LinearisedOp<C: NonLinearOp> {
    callable: Rc<C>,
    x: C::V,
    tmp: RefCell<C::V>,
    x_is_set: bool,
}

impl<C: NonLinearOp> LinearisedOp<C> {
    pub fn new(callable: Rc<C>) -> Self {
        let x = C::V::zeros(callable.nstates());
        let tmp = RefCell::new(C::V::zeros(callable.nstates()));
        Self {
            callable,
            x,
            tmp,
            x_is_set: false,
        }
    }

    pub fn set_x(&mut self, x: &C::V) {
        self.x.copy_from(x);
        self.x_is_set = true;
    }

    pub fn unset_x(&mut self) {
        self.x_is_set = false;
    }

    pub fn x_is_set(&self) -> bool {
        self.x_is_set
    }
}

impl<C: NonLinearOp> Op for LinearisedOp<C> {
    type V = C::V;
    type T = C::T;
    type M = C::M;
    fn nstates(&self) -> usize {
        self.callable.nstates()
    }
    fn nout(&self) -> usize {
        self.callable.nout()
    }
    fn nparams(&self) -> usize {
        self.callable.nparams()
    }
    fn sparsity(&self) -> Option<&<Self::M as Matrix>::Sparsity> {
        self.callable.sparsity()
    }
}

impl<C: NonLinearOp> LinearOp for LinearisedOp<C> {
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        self.callable.jac_mul_inplace(&self.x, t, x, y);
    }
    fn gemv_inplace(&self, x: &Self::V, t: Self::T, beta: Self::T, y: &mut Self::V) {
        let mut tmp = self.tmp.borrow_mut();
        tmp.copy_from(y);
        self.callable.jac_mul_inplace(&self.x, t, x, y);
        y.axpy(beta, &tmp, Self::T::one());
    }
}
