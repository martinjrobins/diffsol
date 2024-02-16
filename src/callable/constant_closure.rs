use crate::{Matrix, Vector};

use super::{ConstantOp, Op};

pub struct ConstantClosure<M> 
where
    M: Matrix,
{
    func: Box<dyn Fn(&M::V, M::T) -> M::V>,
    nstates: usize,
    nout: usize,
    nparams: usize,
    _phantom: std::marker::PhantomData<M>,
}

impl<M> ConstantClosure<M> 
where
    M: Matrix,
{
    pub fn new(func: impl Fn(&M::V, M::T) -> M::V + 'static, nstates: usize, nout: usize, nparams: usize) -> Self {
        Self { func: Box::new(func), nstates, nout, nparams, _phantom: std::marker::PhantomData }
    }
}

impl<M,> Op for ConstantClosure<M>
where
    M: Matrix,
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


impl<M> ConstantOp for ConstantClosure<M>
where
    M: Matrix,
{
    fn call_inplace(&self, p: &M::V, t: M::T, y: &mut M::V) {
        y.copy_from(&(self.func)(p, t))
    }
}

