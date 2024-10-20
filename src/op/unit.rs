// unit is a callable that returns returns the input vector

use crate::{
    LinearOp, LinearOpSens, LinearOpTranspose, Matrix, NonLinearOp, NonLinearOpAdjoint,
    NonLinearOpJacobian, NonLinearOpSens, NonLinearOpSensAdjoint, Op, Vector,
};
use num_traits::{One, Zero};

/// A dummy operator that returns the input vector. Can be used either as a [NonLinearOp] or [LinearOp].
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

impl<M: Matrix> NonLinearOp for UnitCallable<M> {
    fn call_inplace(&self, x: &Self::V, _t: Self::T, y: &mut Self::V) {
        y.copy_from(x);
    }
}

impl<M: Matrix> NonLinearOpJacobian for UnitCallable<M> {
    fn jac_mul_inplace(&self, _x: &Self::V, _t: Self::T, v: &Self::V, y: &mut Self::V) {
        y.copy_from(v);
    }
}

impl<M: Matrix> NonLinearOpAdjoint for UnitCallable<M> {
    fn jac_transpose_mul_inplace(&self, _x: &Self::V, _t: Self::T, v: &Self::V, y: &mut Self::V) {
        y.copy_from(v);
    }
}

impl<M: Matrix> NonLinearOpSens for UnitCallable<M> {
    fn sens_mul_inplace(&self, _x: &Self::V, _t: Self::T, _v: &Self::V, y: &mut Self::V) {
        y.fill(Self::T::zero());
    }
}

impl<M: Matrix> NonLinearOpSensAdjoint for UnitCallable<M> {
    fn sens_transpose_mul_inplace(&self, _x: &Self::V, _t: Self::T, _v: &Self::V, y: &mut Self::V) {
        y.fill(Self::T::zero());
    }
}

impl<M: Matrix> LinearOpSens for UnitCallable<M> {
    fn sens_mul_inplace(&self, _x: &Self::V, _t: Self::T, _v: &Self::V, y: &mut Self::V) {
        y.fill(Self::T::zero());
    }
}

impl<M: Matrix> LinearOpTranspose for UnitCallable<M> {
    fn gemv_transpose_inplace(&self, x: &Self::V, _t: Self::T, beta: Self::T, y: &mut Self::V) {
        y.axpy(Self::T::one(), x, beta);
    }
}
