use crate::{ConstantOp, ConstantOpSensAdjoint, Matrix, Op};

use super::{BuilderOp, ParameterisedOp};

pub struct ConstantClosureWithAdjoint<M, I, J>
where
    M: Matrix,
    I: Fn(&M::V, M::T, &mut M::V),
    J: Fn(&M::V, M::T, &M::V, &mut M::V),
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
    I: Fn(&M::V, M::T, &mut M::V),
    J: Fn(&M::V, M::T, &M::V, &mut M::V),
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
    I: Fn(&M::V, M::T, &mut M::V),
    J: Fn(&M::V, M::T, &M::V, &mut M::V),
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
    I: Fn(&M::V, M::T, &mut M::V),
    J: Fn(&M::V, M::T, &M::V, &mut M::V),
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
    I: Fn(&M::V, M::T, &mut M::V),
    J: Fn(&M::V, M::T, &M::V, &mut M::V),
{
    fn call_inplace(&self, t: Self::T, y: &mut Self::V) {
        (self.op.func)(self.p, t, y)
    }
}

impl<M, I, J> ConstantOpSensAdjoint for ParameterisedOp<'_, ConstantClosureWithAdjoint<M, I, J>>
where
    M: Matrix,
    I: Fn(&M::V, M::T, &mut M::V),
    J: Fn(&M::V, M::T, &M::V, &mut M::V),
{
    fn sens_transpose_mul_inplace(&self, t: Self::T, v: &Self::V, y: &mut Self::V) {
        (self.op.func_sens_adjoint)(self.p, t, v, y);
    }
}
