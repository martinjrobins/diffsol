use crate::{matrix::MatrixCommon, Matrix};

use super::{LinearOp, Op};

pub struct LinearClosure<M, F> 
where
    M: Matrix,
    F: Fn(&<M as MatrixCommon>::V, &<M as MatrixCommon>::V, <M as MatrixCommon>::T, &mut <M as MatrixCommon>::V)
{
    func: F,
    nstates: usize,
    nout: usize,
    nparams: usize,
    _phantom: std::marker::PhantomData<M>,
}

impl<M, F> LinearClosure<M, F> 
where
    M: Matrix,
    F: Fn(&<M as MatrixCommon>::V, &<M as MatrixCommon>::V, <M as MatrixCommon>::T, &mut <M as MatrixCommon>::V)
{
    pub fn new(func: F, nstates: usize, nout: usize, nparams: usize) -> Self {
        Self { func, nstates, nout, nparams, _phantom: std::marker::PhantomData }
    }
}

impl<M, F> Op for LinearClosure<M, F>
where
    M: Matrix,
    F: Fn(&<M as MatrixCommon>::V, &<M as MatrixCommon>::V, <M as MatrixCommon>::T, &mut <M as MatrixCommon>::V)
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


impl<M, F> LinearOp for LinearClosure<M, F>
where
    M: Matrix,
    F: Fn(&<M as MatrixCommon>::V, &<M as MatrixCommon>::V, <M as MatrixCommon>::T, &mut <M as MatrixCommon>::V)
{
    fn call_inplace(&self, x: &M::V, p: &M::V, t: M::T, y: &mut M::V) {
        (self.func)(x, p, t, y)
    }
}
