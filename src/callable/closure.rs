use crate::Matrix;

use super::{NonLinearOp, Op};

pub struct Closure<M, F, G> 
where 
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V)
{
    func: F,
    jacobian_action: G,
    nstates: usize,
    nout: usize,
    nparams: usize,
    _phantom: std::marker::PhantomData<M>,
}

impl<M, F, G> Closure<M, F, G> 
where 
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V)

{
    pub fn new(func: F, jacobian_action: G, nstates: usize, nout: usize, nparams: usize) -> Self {
        Self { func, jacobian_action: jacobian_action, nstates, nout, nparams, _phantom: std::marker::PhantomData }
    }
}

impl<M, F, G> Op for Closure<M, F, G>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V)
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


impl<M, F, G> NonLinearOp for Closure<M, F, G>
where 
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V)
{
    fn call_inplace(&self, x: &M::V, p: &M::V, t: M::T, y: &mut M::V) {
        (self.func)(x, p, t, y)
    }
    fn jac_mul_inplace(&self, x: &M::V, p: &M::V, t: M::T, v: &M::V, y: &mut M::V) {
        (self.jacobian_action)(x, p, t, v, y)
    }
}

