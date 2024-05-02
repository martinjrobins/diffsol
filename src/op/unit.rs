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
    fn gemv_inplace(&self, x: &Self::V, _t: Self::T, beta: Self::T, y: &mut Self::V) {
        y.axpy(Self::T::one(), x, beta);
    }
}
