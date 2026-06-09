use std::marker::PhantomData;

use num_traits::Zero;

use crate::{ConstantAutodiff, ConstantOp, ConstantOpSensAdjoint, Matrix, Op, Vector};

use super::{BuilderOp, ParameterisedOp};

/// An [`Op`] wrapping a [`ConstantAutodiff`] implementation.
///
/// Extracts slices from vector types and passes them to the trait's slice-based
/// functions.
pub struct ConstantClosureAutodiff<M: Matrix, T: ConstantAutodiff<M>> {
    nout: usize,
    nparams: usize,
    ctx: M::C,
    _phantom: PhantomData<(M, T)>,
}

impl<M: Matrix, T: ConstantAutodiff<M>> ConstantClosureAutodiff<M, T> {
    pub fn new(nout: usize, nparams: usize, ctx: M::C) -> Self {
        Self {
            nout,
            nparams,
            ctx,
            _phantom: PhantomData,
        }
    }
}

impl<M: Matrix, T: ConstantAutodiff<M>> BuilderOp for ConstantClosureAutodiff<M, T> {
    fn calculate_sparsity(&mut self, _y0: &Self::V, _t0: Self::T, _p: &Self::V) {}
    fn set_nstates(&mut self, _nstates: usize) {}
    fn set_nout(&mut self, nout: usize) {
        self.nout = nout;
    }
    fn set_nparams(&mut self, nparams: usize) {
        self.nparams = nparams;
    }
}

impl<M: Matrix, T: ConstantAutodiff<M>> Op for ConstantClosureAutodiff<M, T> {
    type V = M::V;
    type T = M::T;
    type M = M;
    type C = M::C;
    fn nstates(&self) -> usize {
        0
    }
    fn nout(&self) -> usize {
        self.nout
    }
    fn nparams(&self) -> usize {
        self.nparams
    }
    fn context(&self) -> &Self::C {
        &self.ctx
    }
}

fn to_vec<V: Vector>(v: &V, n: usize) -> Vec<V::T> {
    (0..n).map(|i| v.get_index(i)).collect()
}

impl<M: Matrix, T: ConstantAutodiff<M, T = M::T>> ConstantOp
    for ParameterisedOp<'_, ConstantClosureAutodiff<M, T>>
{
    fn call_inplace(&self, _t: Self::T, y: &mut Self::V) {
        let p_slice = to_vec(self.p, self.op.nparams);
        let mut y_slice = vec![M::T::zero(); self.op.nout];
        T::init_inplace(&p_slice, &mut y_slice);
        y.set_index(0, y_slice[0]);
    }
}

impl<M: Matrix, T: ConstantAutodiff<M, T = M::T>> ConstantOpSensAdjoint
    for ParameterisedOp<'_, ConstantClosureAutodiff<M, T>>
{
    fn sens_transpose_mul_inplace(&self, _t: Self::T, v: &Self::V, y: &mut Self::V) {
        let p_slice = to_vec(self.p, self.op.nparams);
        let mut dp = vec![M::T::zero(); self.op.nparams];
        let mut y_slice = vec![M::T::zero(); self.op.nout];
        let mut dy = vec![v.get_index(0); self.op.nout];
        T::init_sens_vjp_inplace(&p_slice, &mut dp, &mut y_slice, &mut dy);
        for i in 0..self.op.nparams {
            y.set_index(i, -dp[i]);
        }
    }
}
