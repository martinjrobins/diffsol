use crate::{matrix::MatrixCommon, IndexType, Matrix};

use super::{Jacobian, NonLinearOp, Op};

type ClosureFn<M, D> = dyn Fn(&<M as MatrixCommon>::V, &<M as MatrixCommon>::V, <M as MatrixCommon>::T, &mut <M as MatrixCommon>::V, &D);
type JacobianActionFn<M, D> = dyn Fn(&<M as MatrixCommon>::V, &<M as MatrixCommon>::V, <M as MatrixCommon>::T, &<M as MatrixCommon>::V, &mut <M as MatrixCommon>::V, &D);

pub struct Closure<M: Matrix, D> 
{
    func: Box<ClosureFn<M, D>>,
    jacobian_action: Box<JacobianActionFn<M, D>>,
    data: D,
    nstates: IndexType,
    _phantom: std::marker::PhantomData<M>,
}

impl<M: Matrix, D> Closure<M, D> 
{
    pub fn new(func: impl Fn(&M::V, &M::V, M::T, &mut M::V, &D) + 'static, jacobian_action: impl Fn(&M::V, &M::V, M::T, &M::V, &mut M::V, &D) + 'static, data: D, nstates: IndexType) -> Self {
        Self { func: Box::new(func), jacobian_action: Box::new(jacobian_action), data, nstates, _phantom: std::marker::PhantomData }
    }
}

impl<M: Matrix, D> Op for Closure<M, D>
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


impl<M: Matrix, D> NonLinearOp for Closure<M, D>
{
    fn call_inplace(&self, x: &M::V, p: &M::V, t: M::T, y: &mut M::V) {
        (self.func)(x, p, t, y, &self.data)
    }
    fn jac_mul_inplace(&self, x: &M::V, p: &M::V, t: M::T, v: &M::V, y: &mut M::V) {
        (self.jacobian_action)(x, p, t, v, y, &self.data)
    }
}

// implement Jacobian
impl<M: Matrix, D> Jacobian for Closure<M, D>
{
    type M = M;
}
