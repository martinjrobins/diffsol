use super::{BuilderOp, ParameterisedOp};
use crate::{ConstantOp, Matrix, Op, Vector};

pub struct ConstantClosure<M, I>
where
    M: Matrix,
    I: Fn(&M::V, M::T) -> M::V,
{
    func: I,
    nout: usize,
    nparams: usize,
    _phantom: std::marker::PhantomData<M>,
}

impl<M, I> ConstantClosure<M, I>
where
    M: Matrix,
    I: Fn(&M::V, M::T) -> M::V,
{
    pub fn new(func: I, nout: usize, nparams: usize) -> Self {
        Self {
            func,
            nout,
            nparams,
            _phantom: std::marker::PhantomData,
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
        0
    }
    fn nout(&self) -> usize {
        self.nout
    }
    fn nparams(&self) -> usize {
        self.nparams
    }
}

impl<M, I> BuilderOp for ConstantClosure<M, I>
where
    M: Matrix,
    I: Fn(&M::V, M::T) -> M::V,
{
    fn calculate_sparsity(&mut self, _y0: &Self::V, _t0: Self::T, _p: &Self::V) {
        // do nothing
    }
    fn set_nstates(&mut self, _nstates: usize) {
        // do nothing
    }
    fn set_nout(&mut self, nout: usize) {
        self.nout = nout;
    }
    fn set_nparams(&mut self, nparams: usize) {
        self.nparams = nparams;
    }
}

impl<'a, M, I> ConstantOp for ParameterisedOp<'a, ConstantClosure<M, I>>
where
    M: Matrix,
    I: Fn(&M::V, M::T) -> M::V,
{
    fn call_inplace(&self, t: Self::T, y: &mut Self::V) {
        y.copy_from(&(self.op.func)(self.p, t));
    }
    fn call(&self, t: Self::T) -> Self::V {
        (self.op.func)(self.p, t)
    }
}
