use crate::{IndexType, Matrix};

use super::{Callable, Jacobian};

pub struct Closure<M, F, G, D> 
    where F: Fn(&M::V, &M::V, &mut M::V, &D),
          G: Fn(&M::V, &M::V, &M::V, &mut M::V, &D),
          M: Matrix,
{
    func: F,
    jacobian_action: G,
    data: D,
    nstates: IndexType,
    _phantom: std::marker::PhantomData<M>,
}

impl<M, F, G, D> Closure<M, F, G, D> 
    where F: Fn(&M::V, &M::V, &mut M::V, &D),
          G: Fn(&M::V, &M::V, &M::V, &mut M::V, &D),
          M: Matrix,
{
    pub fn new(func: F, jacobian_action: G, data: D, nstates: IndexType) -> Self {
        Self { func, jacobian_action, data, nstates, _phantom: std::marker::PhantomData }
    }
}

impl<F, G, D, M> Callable for Closure<M, F, G, D>
    where F: Fn(&M::V, &M::V, &mut M::V, &D),
          G: Fn(&M::V, &M::V, &M::V, &mut M::V, &D),
          M: Matrix,
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
impl<F, G, D, M> Jacobian for Closure<M, F, G, D>
    where F: Fn(&M::V, &M::V, &mut M::V, &D),
          G: Fn(&M::V, &M::V, &M::V, &mut M::V, &D),
          M: Matrix,
{
    type M = M;
}
