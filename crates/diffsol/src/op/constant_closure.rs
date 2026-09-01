use super::{BuilderOp, ParameterisedOp};
use crate::{ConstantOp, Matrix, Op, Vector};

pub struct ConstantClosure<M, I>
where
    M: Matrix,
    I: Fn(&[M::T], M::T, &mut [M::T]),
{
    func: I,
    nout: usize,
    nparams: usize,
    ctx: M::C,
}

impl<M, I> ConstantClosure<M, I>
where
    M: Matrix,
    I: Fn(&[M::T], M::T, &mut [M::T]),
{
    pub fn new(func: I, nout: usize, nparams: usize, ctx: M::C) -> Self {
        Self {
            func,
            nout,
            nparams,
            ctx,
        }
    }
}

impl<M, I> Op for ConstantClosure<M, I>
where
    M: Matrix,
    I: Fn(&[M::T], M::T, &mut [M::T]),
{
    type V = M::V;
    type T = M::T;
    type M = M;
    type C = M::C;
    fn nstates(&self) -> usize {
        0
    }
    fn nout(&self) -> usize {
        self.nout
    }
    fn nparams(&self) -> usize {
        self.nparams
    }
    fn context(&self) -> &Self::C {
        &self.ctx
    }
}

impl<M, I> BuilderOp for ConstantClosure<M, I>
where
    M: Matrix,
    I: Fn(&[M::T], M::T, &mut [M::T]),
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

impl<M, I> ConstantOp for ParameterisedOp<'_, ConstantClosure<M, I>>
where
    M: Matrix,
    I: Fn(&[M::T], M::T, &mut [M::T]),
{
    fn call_inplace(&self, t: Self::T, y: &mut Self::V) {
        y.for_each_batch([self.p], |y, [p], _| (self.op.func)(p, t, y));
    }
}
