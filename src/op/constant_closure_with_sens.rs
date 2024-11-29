use crate::{ConstantOp, ConstantOpSens, Matrix, Op, Vector};

use super::{BuilderOp, ParameterisedOp};

pub struct ConstantClosureWithSens<M, I, J>
where
    M: Matrix,
    I: Fn(&M::V, M::T) -> M::V,
    J: Fn(&M::V, M::T, &M::V, &mut M::V),
{
    func: I,
    func_sens: J,
    nout: usize,
    nparams: usize,
    _phantom: std::marker::PhantomData<M>,
}

impl<M, I, J> ConstantClosureWithSens<M, I, J>
where
    M: Matrix,
    I: Fn(&M::V, M::T) -> M::V,
    J: Fn(&M::V, M::T, &M::V, &mut M::V),
{
    pub fn new(func: I, func_sens: J, nout: usize, nparams: usize) -> Self {
        Self {
            func,
            func_sens,
            nout,
            nparams,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<M, I, J> Op for ConstantClosureWithSens<M, I, J>
where
    M: Matrix,
    I: Fn(&M::V, M::T) -> M::V,
    J: Fn(&M::V, M::T, &M::V, &mut M::V),
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

impl<M, I, J> BuilderOp for ConstantClosureWithSens<M, I, J>
where
    M: Matrix,
    I: Fn(&M::V, M::T) -> M::V,
    J: Fn(&M::V, M::T, &M::V, &mut M::V),
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

impl<'a, M, I, J> ConstantOp for ParameterisedOp<'a, ConstantClosureWithSens<M, I, J>>
where
    M: Matrix,
    I: Fn(&M::V, M::T) -> M::V,
    J: Fn(&M::V, M::T, &M::V, &mut M::V),
{
    fn call_inplace(&self, t: Self::T, y: &mut Self::V) {
        y.copy_from(&(self.op.func)(self.p, t));
    }
    fn call(&self, t: Self::T) -> Self::V {
        (self.op.func)(self.p, t)
    }
}

impl<'a, M, I, J> ConstantOpSens for ParameterisedOp<'a, ConstantClosureWithSens<M, I, J>>
where
    M: Matrix,
    I: Fn(&M::V, M::T) -> M::V,
    J: Fn(&M::V, M::T, &M::V, &mut M::V),
{
    fn sens_mul_inplace(&self, t: Self::T, v: &Self::V, y: &mut Self::V) {
        (self.op.func_sens)(self.p, t, v, y);
    }
}
