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
    ctx: M::C,
}

impl<M: Matrix> Default for UnitCallable<M> {
    fn default() -> Self {
        Self::new(1, M::C::default())
    }
}

impl<M: Matrix> UnitCallable<M> {
    pub fn new(n: usize, ctx: M::C) -> Self {
        Self { n, ctx }
    }
}

impl<M: Matrix> Op for UnitCallable<M> {
    type T = M::T;
    type V = M::V;
    type M = M;
    type C = M::C;
    fn nstates(&self) -> usize {
        self.n
    }
    fn nout(&self) -> usize {
        self.n
    }
    fn nparams(&self) -> usize {
        0
    }
    fn context(&self) -> &Self::C {
        &self.ctx
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

impl<M: Matrix> LinearOp for ParameterisedOp<'_, UnitCallable<M>> {
    fn gemv_inplace(&self, x: &Self::V, _t: Self::T, beta: Self::T, y: &mut Self::V) {
        y.axpy(Self::T::one(), x, beta);
    }
}

impl<M: Matrix> NonLinearOp for ParameterisedOp<'_, UnitCallable<M>> {
    fn call_inplace(&self, x: &Self::V, _t: Self::T, y: &mut Self::V) {
        y.copy_from(x);
    }
}

impl<M: Matrix> NonLinearOpJacobian for ParameterisedOp<'_, UnitCallable<M>> {
    fn jac_mul_inplace(&self, _x: &Self::V, _t: Self::T, v: &Self::V, y: &mut Self::V) {
        y.copy_from(v);
    }
}

impl<M: Matrix> NonLinearOpAdjoint for ParameterisedOp<'_, UnitCallable<M>> {
    fn jac_transpose_mul_inplace(&self, _x: &Self::V, _t: Self::T, v: &Self::V, y: &mut Self::V) {
        y.copy_from(v);
    }
}

impl<M: Matrix> NonLinearOpSens for ParameterisedOp<'_, UnitCallable<M>> {
    fn sens_mul_inplace(&self, _x: &Self::V, _t: Self::T, _v: &Self::V, y: &mut Self::V) {
        y.fill(Self::T::zero());
    }
}

impl<M: Matrix> NonLinearOpSensAdjoint for ParameterisedOp<'_, UnitCallable<M>> {
    fn sens_transpose_mul_inplace(&self, _x: &Self::V, _t: Self::T, _v: &Self::V, y: &mut Self::V) {
        y.fill(Self::T::zero());
    }
}

impl<M: Matrix> LinearOpSens for ParameterisedOp<'_, UnitCallable<M>> {
    fn sens_mul_inplace(&self, _x: &Self::V, _t: Self::T, _v: &Self::V, y: &mut Self::V) {
        y.fill(Self::T::zero());
    }
}

impl<M: Matrix> LinearOpTranspose for ParameterisedOp<'_, UnitCallable<M>> {
    fn gemv_transpose_inplace(&self, x: &Self::V, _t: Self::T, beta: Self::T, y: &mut Self::V) {
        y.axpy(Self::T::one(), x, beta);
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
