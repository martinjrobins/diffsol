use std::rc::Rc;

use crate::{Matrix, Vector};

use super::{ConstantOp, Op};

type ConstFn<V, T> = dyn Fn(&V, T) -> V;

pub struct ConstantClosure<M>
where
    M: Matrix,
{
    func: Box<ConstFn<M::V, M::T>>,
    nstates: usize,
    nout: usize,
    nparams: usize,
    p: Rc<M::V>,
}

impl<M> ConstantClosure<M>
where
    M: Matrix,
{
    pub fn new(
        func: impl Fn(&M::V, M::T) -> M::V + 'static,
        nstates: usize,
        nout: usize,
        p: Rc<M::V>,
    ) -> Self {
        let nparams = p.len();
        Self {
            func: Box::new(func),
            nstates,
            nout,
            nparams,
            p,
        }
    }
}

impl<M> Op for ConstantClosure<M>
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
    fn call_inplace(&self, t: M::T, y: &mut M::V) {
        y.copy_from(&(self.func)(self.p.as_ref(), t))
    }
}
