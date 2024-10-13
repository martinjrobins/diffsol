use num_traits::Zero;
use std::rc::Rc;

use crate::{Matrix, Vector, ConstantOp, Op};

pub struct ConstantClosure<M, I>
where
    M: Matrix,
    I: Fn(&M::V, M::T) -> M::V,
{
    func: I,
    nstates: usize,
    nout: usize,
    nparams: usize,
    p: Rc<M::V>,
}

impl<M, I> ConstantClosure<M, I>
where
    M: Matrix,
    I: Fn(&M::V, M::T) -> M::V,
{
    pub fn new(func: I, p: Rc<M::V>) -> Self {
        let nparams = p.len();
        let y0 = (func)(p.as_ref(), M::T::zero());
        let nstates = y0.len();
        let nout = nstates;
        Self {
            func,
            nstates,
            nout,
            nparams,
            p,
        }
    }
}

impl<M, I> Op for ConstantClosure<M, I>
where
    M: Matrix,
    I: Fn(&M::V, M::T) -> M::V,
{
    type V = M::V;
    type T = M::T;
    type M = M;
    fn nstates(&self) -> usize {
        self.nstates
    }
    fn nout(&self) -> usize {
        self.nout
    }
    fn nparams(&self) -> usize {
        self.nparams
    }
    fn set_params(&mut self, p: Rc<M::V>) {
        assert_eq!(p.len(), self.nparams);
        self.p = p;
    }
}

impl<M, I> ConstantOp for ConstantClosure<M, I>
where
    M: Matrix,
    I: Fn(&M::V, M::T) -> M::V,
{
    fn call_inplace(&self, t: Self::T, y: &mut Self::V) {
        y.copy_from(&(self.func)(self.p.as_ref(), t));
    }
    fn call(&self, t: Self::T) -> Self::V {
        (self.func)(self.p.as_ref(), t)
    }
}
