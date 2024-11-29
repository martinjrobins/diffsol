// unit is a callable that returns returns the input vector

use crate::{
    LinearOp, LinearOpSens, LinearOpTranspose, Matrix, NonLinearOp, NonLinearOpAdjoint,
    NonLinearOpJacobian, NonLinearOpSens, NonLinearOpSensAdjoint, Op, Vector,
};
use num_traits::{One, Zero};

use super::{BuilderOp, ParameterisedOp};

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

impl<M: Matrix> BuilderOp for UnitCallable<M> {
    fn calculate_sparsity(&mut self, _y0: &Self::V, _t0: Self::T, _p: &Self::V) {
        // Do nothing
    }
    fn set_nout(&mut self, nout: usize) {
        self.n = nout;
    }
    fn set_nparams(&mut self, _nparams: usize) {
        // Do nothing
    }
    fn set_nstates(&mut self, nstates: usize) {
        self.n = nstates;
    }
}

impl<'a, M: Matrix> LinearOp for ParameterisedOp<'a, UnitCallable<M>> {
    fn gemv_inplace(&self, x: &Self::V, _t: Self::T, beta: Self::T, y: &mut Self::V) {
        y.axpy(Self::T::one(), x, beta);
    }
}

impl<'a, M: Matrix> NonLinearOp for ParameterisedOp<'a, UnitCallable<M>> {
    fn call_inplace(&self, x: &Self::V, _t: Self::T, y: &mut Self::V) {
        y.copy_from(x);
    }
}

impl<'a, M: Matrix> NonLinearOpJacobian for ParameterisedOp<'a, UnitCallable<M>> {
    fn jac_mul_inplace(&self, _x: &Self::V, _t: Self::T, v: &Self::V, y: &mut Self::V) {
        y.copy_from(v);
    }
}

impl<'a, M: Matrix> NonLinearOpAdjoint for ParameterisedOp<'a, UnitCallable<M>> {
    fn jac_transpose_mul_inplace(&self, _x: &Self::V, _t: Self::T, v: &Self::V, y: &mut Self::V) {
        y.copy_from(v);
    }
}

impl<'a, M: Matrix> NonLinearOpSens for ParameterisedOp<'a, UnitCallable<M>> {
    fn sens_mul_inplace(&self, _x: &Self::V, _t: Self::T, _v: &Self::V, y: &mut Self::V) {
        y.fill(Self::T::zero());
    }
}

impl<'a, M: Matrix> NonLinearOpSensAdjoint for ParameterisedOp<'a, UnitCallable<M>> {
    fn sens_transpose_mul_inplace(&self, _x: &Self::V, _t: Self::T, _v: &Self::V, y: &mut Self::V) {
        y.fill(Self::T::zero());
    }
}

impl<'a, M: Matrix> LinearOpSens for ParameterisedOp<'a, UnitCallable<M>> {
    fn sens_mul_inplace(&self, _x: &Self::V, _t: Self::T, _v: &Self::V, y: &mut Self::V) {
        y.fill(Self::T::zero());
    }
}

impl<'a, M: Matrix> LinearOpTranspose for ParameterisedOp<'a, UnitCallable<M>> {
    fn gemv_transpose_inplace(&self, x: &Self::V, _t: Self::T, beta: Self::T, y: &mut Self::V) {
        y.axpy(Self::T::one(), x, beta);
    }
}
