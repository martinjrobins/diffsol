use crate::{IndexType, Matrix, Vector};

use super::{Callable, Jacobian};

pub struct Closure<F, G, D> {
    func: F,
    jacobian_action: G,
    data: D,
    nstates: IndexType,
}

impl<F, G, D> Closure<F, G, D> {
    pub fn new(func: F, jacobian_action: G, data: D, nstates: IndexType) -> Self {
        Self { func, jacobian_action, data, nstates }
    }
}

impl<F, G, D, V> Callable<V> for Closure<F, G, D>
    where F: Fn(&V, &V, &mut V, &D),
          G: Fn(&V, &V, &V, &mut V, &D),
          V: Vector,
{
    fn call(&self, x: &V, p: &V, y: &mut V) {
        (self.func)(x, p, y, &self.data)
    }
    fn jacobian_action(&self, x: &V, p: &V, v: &V, y: &mut V) {
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
impl<F, G, D, M> Jacobian<M> for Closure<F, G, D>
    where F: Fn(&M::V, &M::V, &mut M::V, &D),
          G: Fn(&M::V, &M::V, &M::V, &mut M::V, &D),
          M: Matrix,
{}
