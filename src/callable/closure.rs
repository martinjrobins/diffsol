use crate::{Scalar, Vector, IndexType};

use super::Callable;

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

impl<F, G, D, T, V> Callable<T, V> for Closure<F, G, D>
    where F: Fn(&V, &mut V, &D),
          G: Fn(&V, &V, &mut V, &D),
          T: Scalar,
          V: Vector<T>,
{
    fn call(&self, x: &V, y: &mut V) {
        (self.func)(x, y, &self.data)
    }
    fn jacobian_action(&self, x: &V, v: &V, y: &mut V) {
        (self.jacobian_action)(x, v, y, &self.data)
    }
    fn nstates(&self) -> usize {
        self.nstates
    }
    fn nparams(&self) -> usize {
        0
    }
}