use crate::{matrix::MatrixCommon, IndexType, Matrix};

use super::{ConstantOp, Op};

type ClosureFn<M, D> = dyn Fn(&<M as MatrixCommon>::V, &mut <M as MatrixCommon>::V, &D);

pub struct ConstantClosure<M: Matrix, D> 
{
    func: Box<ClosureFn<M, D>>,
    data: D,
    nstates: IndexType,
    _phantom: std::marker::PhantomData<M>,
}

impl<M: Matrix, D> ConstantClosure<M, D> 
{
    pub fn new(func: impl Fn(&M::V, &mut M::V, &D) + 'static, data: D, nstates: IndexType) -> Self {
        Self { func: Box::new(func), data, nstates, _phantom: std::marker::PhantomData }
    }
}

impl<M: Matrix, D> Op for ConstantClosure<M, D>
{
    type V = M::V;
    type T = M::T;
    fn nstates(&self) -> usize {
        self.nstates
    }
    fn nout(&self) -> usize {
        self.nstates
    }
    fn nparams(&self) -> usize {
        0
    }
}


impl<M: Matrix, D> ConstantOp for ConstantClosure<M, D>
{
    fn call_inplace(&self, p: &M::V, y: &mut M::V) {
        (self.func)(p, y, &self.data)
    }
}

