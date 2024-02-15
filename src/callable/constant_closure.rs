use crate::Matrix;

use super::{ConstantOp, Op};

pub struct ConstantClosure<M, F> 
where
    M: Matrix,
    F: Fn(&M::V, M::T, &mut M::V)
{
    func: F,
    nstates: usize,
    nout: usize,
    nparams: usize,
    _phantom: std::marker::PhantomData<M>,
}

impl<M, F> ConstantClosure<M, F> 
where
    M: Matrix,
    F: Fn(&M::V, M::T, &mut M::V)
{
    pub fn new(func: F, nstates: usize, nout: usize, nparams: usize) -> Self {
        Self { func, nstates, nout, nparams, _phantom: std::marker::PhantomData }
    }
}

impl<M, F> Op for ConstantClosure<M, F>
where
    M: Matrix,
    F: Fn(&M::V, M::T, &mut M::V)
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
}


impl<M, F> ConstantOp for ConstantClosure<M, F>
where
    M: Matrix,
    F: Fn(&M::V, M::T, &mut M::V)
{
    fn call_inplace(&self, p: &M::V, t: M::T, y: &mut M::V) {
        (self.func)(p, t, y)
    }
}

