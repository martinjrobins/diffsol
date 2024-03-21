// unit is a callable that returns returns the input vector

use crate::{Matrix, Vector};
use num_traits::One;

use super::{LinearOp, Op};

pub struct UnitCallable<M: Matrix> {
    n: usize,
    _phantom: std::marker::PhantomData<M>,
}

impl<M: Matrix> Default for UnitCallable<M> {
    fn default() -> Self {
        Self::new(1)
    }
}

impl<M: Matrix> UnitCallable<M> {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<M: Matrix> Op for UnitCallable<M> {
    type T = M::T;
    type V = M::V;
    type M = M;
    fn nstates(&self) -> usize {
        self.n
    }
    fn nout(&self) -> usize {
        self.n
    }
    fn nparams(&self) -> usize {
        0
    }
}

impl<M: Matrix> LinearOp for UnitCallable<M> {
    fn call_inplace(&self, x: &M::V, _t: M::T, y: &mut M::V) {
        y.copy_from(x)
    }
    fn jacobian(&self, _t: Self::T) -> Self::M {
        let mut jac = M::V::zeros(self.n);
        jac.add_scalar_mut(M::T::one());
        M::from_diagonal(&jac)
    }
}
