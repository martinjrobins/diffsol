use crate::{matrix::MatrixCommon, IndexType, Matrix};

use super::{ConstantJacobian, LinearOp, Op};

type ClosureFn<M, D> = dyn Fn(&<M as MatrixCommon>::V, &<M as MatrixCommon>::V, <M as MatrixCommon>::T, &mut <M as MatrixCommon>::V, &D);

pub struct LinearClosure<M: Matrix, D> 
{
    func: Box<ClosureFn<M, D>>,
    data: D,
    nstates: IndexType,
    _phantom: std::marker::PhantomData<M>,
}

impl<M: Matrix, D> LinearClosure<M, D> 
{
    pub fn new(func: impl Fn(&M::V, &M::V, M::T, &mut M::V, &D) + 'static, data: D, nstates: IndexType) -> Self {
        Self { func: Box::new(func), data, nstates, _phantom: std::marker::PhantomData }
    }
}

impl<M: Matrix, D> Op for LinearClosure<M, D>
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


impl<M: Matrix, D> LinearOp for LinearClosure<M, D>
{
    fn call_inplace(&self, x: &M::V, p: &M::V, t: M::T, y: &mut M::V) {
        (self.func)(x, p, t, y, &self.data)
    }
}

// implement ConstantJacobian
impl<M: Matrix, D> ConstantJacobian for LinearClosure<M, D>
{
    type M = M;
}
