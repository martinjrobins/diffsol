use std::rc::Rc;

use super::{ConstantJacobian, Jacobian, LinearOp, NonLinearOp, Op};

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
    fn call_inplace(&self, x: &Self::V, p: &Self::V, y: &mut Self::V) {
        self.callable.jac_mul_inplace(&self.x, p, x, y);
    }

}

impl <C: Jacobian> ConstantJacobian for LinearisedOp<C> 
{
    type M = C::M;
}