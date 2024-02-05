use crate::{matrix::MatrixCommon, IndexType, Matrix};

use super::{ConstantJacobian, LinearOp, Op};

type ClosureFn<M, D> = dyn Fn(&<M as MatrixCommon>::V, &<M as MatrixCommon>::V, &mut <M as MatrixCommon>::V, &D);
type JacobianActionFn<M, D> = dyn Fn(&<M as MatrixCommon>::V, &<M as MatrixCommon>::V, &mut <M as MatrixCommon>::V, &D);

pub struct LinearClosure<M: Matrix, D> 
{
    func: Box<ClosureFn<M, D>>,
    jacobian_action: Box<JacobianActionFn<M, D>>,
    data: D,
    nstates: IndexType,
    _phantom: std::marker::PhantomData<M>,
}

impl<M: Matrix, D> LinearClosure<M, D> 
{
    pub fn new(func: impl Fn(&M::V, &M::V, &mut M::V, &D) + 'static, jacobian_action: impl Fn(&M::V, &M::V, &mut M::V, &D) + 'static, data: D, nstates: IndexType) -> Self {
        Self { func: Box::new(func), jacobian_action: Box::new(jacobian_action), data, nstates, _phantom: std::marker::PhantomData }
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
    fn call_inplace(&self, x: &M::V, p: &M::V, y: &mut M::V) {
        (self.func)(x, p, y, &self.data)
    }
    fn jac_mul_inplace(&self, p: &M::V, v: &M::V, y: &mut M::V) {
        (self.jacobian_action)(p, v, y, &self.data)
    }
}

// implement ConstantJacobian
impl<M: Matrix, D> ConstantJacobian for LinearClosure<M, D>
{
    type M = M;
}
