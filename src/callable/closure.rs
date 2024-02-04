use crate::{matrix::MatrixCommon, IndexType, Matrix};

use super::{Callable, Jacobian};

type ClosureFn<M, D> = dyn Fn(&<M as MatrixCommon>::V, &<M as MatrixCommon>::V, &mut <M as MatrixCommon>::V, &D);
type JacobianActionFn<M, D> = dyn Fn(&<M as MatrixCommon>::V, &<M as MatrixCommon>::V, &<M as MatrixCommon>::V, &mut <M as MatrixCommon>::V, &D);

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
    pub fn new(func: impl Fn(&M::V, &M::V, &mut M::V, &D) + 'static, jacobian_action: impl Fn(&M::V, &M::V, &M::V, &mut M::V, &D) + 'static, data: D, nstates: IndexType) -> Self {
        Self { func: Box::new(func), jacobian_action: Box::new(jacobian_action), data, nstates, _phantom: std::marker::PhantomData }
    }
}


impl<M: Matrix, D> Callable for Closure<M, D>
{
    type V = M::V;
    type T = M::T;
    fn call(&self, x: &M::V, p: &M::V, y: &mut M::V) {
        (self.func)(x, p, y, &self.data)
    }
    fn jacobian_action(&self, x: &M::V, p: &M::V, v: &M::V, y: &mut M::V) {
        (self.jacobian_action)(x, p, v, y, &self.data)
    }
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

// implement Jacobian
impl<M: Matrix, D> Jacobian for Closure<M, D>
{
    type M = M;
}
