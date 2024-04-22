use std::{cell::RefCell, rc::Rc};
use num_traits::One;

use crate::{scale, Vector};

use super::{LinearOp, NonLinearOp, Op};

pub struct LinearisedOp<C: NonLinearOp> {
    callable: Rc<C>,
    x: C::V,
    tmp: RefCell<C::V>,
}

impl<C: NonLinearOp> LinearisedOp<C> {
    pub fn new(callable: Rc<C>, x: &C::V) -> Self {
        Self {
            callable,
            x: x.clone(),
            tmp: RefCell::new(C::V::zeros(callable.nstates())),
        }
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
}

impl<C: NonLinearOp> LinearOp for LinearisedOp<C> {
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        self.callable.jac_mul_inplace(&self.x, t, x, y);
    }
    fn gemv_inplace(&self, x: &Self::V, t: Self::T, beta: Self::T, y: &mut Self::V) {
        let tmp = self.tmp.borrow_mut();
        tmp.copy_from(y);
        self.callable.jac_mul_inplace(&self.x, t, x, y);
        y.axpy(beta, &tmp, Self::T::one());
    }
    fn sparsity(&self) -> &<Self::M as crate::matrix::Matrix>::Sparsity {
        self.callable.sparsity()
    }
}
