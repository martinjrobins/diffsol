use crate::{ConstantOp, ConstantOpSensAdjoint, Matrix, Op, Vector};

use super::{BuilderOp, ParameterisedOp};

pub struct ConstantClosureWithAdjoint<M, I, J>
where
    M: Matrix,
    I: Fn(&[M::T], M::T, &mut [M::T]),
    J: Fn(&[M::T], M::T, &[M::T], &mut [M::T]),
{
    func: I,
    func_sens_adjoint: J,
    nout: usize,
    nparams: usize,
    ctx: M::C,
}

impl<M, I, J> ConstantClosureWithAdjoint<M, I, J>
where
    M: Matrix,
    I: Fn(&[M::T], M::T, &mut [M::T]),
    J: Fn(&[M::T], M::T, &[M::T], &mut [M::T]),
{
    pub fn new(func: I, func_sens_adjoint: J, nout: usize, nparams: usize, ctx: M::C) -> Self {
        Self {
            func,
            func_sens_adjoint,
            nout,
            nparams,
            ctx,
        }
    }
}

impl<M, I, J> BuilderOp for ConstantClosureWithAdjoint<M, I, J>
where
    M: Matrix,
    I: Fn(&[M::T], M::T, &mut [M::T]),
    J: Fn(&[M::T], M::T, &[M::T], &mut [M::T]),
{
    fn calculate_sparsity(&mut self, _y0: &Self::V, _t0: Self::T, _p: &Self::V) {
        // Do nothing
    }
    fn set_nstates(&mut self, _nstates: usize) {
        // Do nothing
    }
    fn set_nout(&mut self, nout: usize) {
        self.nout = nout;
    }
    fn set_nparams(&mut self, nparams: usize) {
        self.nparams = nparams;
    }
}

impl<M, I, J> Op for ConstantClosureWithAdjoint<M, I, J>
where
    M: Matrix,
    I: Fn(&[M::T], M::T, &mut [M::T]),
    J: Fn(&[M::T], M::T, &[M::T], &mut [M::T]),
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

impl<M, I, J> ConstantOp for ParameterisedOp<'_, ConstantClosureWithAdjoint<M, I, J>>
where
    M: Matrix,
    I: Fn(&[M::T], M::T, &mut [M::T]),
    J: Fn(&[M::T], M::T, &[M::T], &mut [M::T]),
{
    fn call_inplace(&self, t: Self::T, y: &mut Self::V) {
        y.for_each_batch([self.p], |y, [p], _| (self.op.func)(p, t, y));
    }
}

impl<M, I, J> ConstantOpSensAdjoint for ParameterisedOp<'_, ConstantClosureWithAdjoint<M, I, J>>
where
    M: Matrix,
    I: Fn(&[M::T], M::T, &mut [M::T]),
    J: Fn(&[M::T], M::T, &[M::T], &mut [M::T]),
{
    fn sens_transpose_mul_inplace(&self, t: Self::T, v: &Self::V, y: &mut Self::V) {
        y.for_each_batch([self.p, v], |y, [p, v], _| {
            (self.op.func_sens_adjoint)(p, t, v, y)
        });
    }
}
