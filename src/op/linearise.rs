use std::rc::Rc;

use super::{LinearOp, NonLinearOp, Op};

pub struct LinearisedOp<C: NonLinearOp> 
{
    callable: Rc<C>,
    x: C::V,
}

impl<C: NonLinearOp> LinearisedOp<C> 
{
    pub fn new(callable: Rc<C>, x: &C::V) -> Self {
        Self { callable, x: x.clone() }
    }
}

impl<C: NonLinearOp> Op for LinearisedOp<C> 
{
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

impl<C: NonLinearOp> LinearOp for LinearisedOp<C> 
{
    fn call_inplace(&self, x: &Self::V, p: &Self::V, t: Self::T, y: &mut Self::V) {
        self.callable.jac_mul_inplace(&self.x, p, t, x, y);
    }
    fn jacobian(&self, p: &Self::V, t: Self::T) -> Self::M {
        self.callable.jacobian(&self.x, p, t)
    }
}