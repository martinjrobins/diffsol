use std::{cell::RefCell, marker::PhantomData};

use num_traits::Zero;

use crate::{
    Matrix, NonLinearAutodiff, NonLinearOp, NonLinearOpAdjoint, NonLinearOpJacobian,
    NonLinearOpSensAdjoint, Op, Vector,
};

use super::{BuilderOp, OpStatistics, ParameterisedOp};

/// An [`Op`] wrapping a [`NonLinearAutodiff`] implementation.
///
/// Extracts slices from vector types and passes them to the trait's slice-based
/// functions, avoiding Enzyme contact with nalgebra internals.
pub struct ClosureAutodiff<M: Matrix, T: NonLinearAutodiff<M>> {
    nstates: usize,
    nout: usize,
    nparams: usize,
    statistics: RefCell<OpStatistics>,
    ctx: M::C,
    _phantom: PhantomData<(M, T)>,
}

impl<M: Matrix, T: NonLinearAutodiff<M>> ClosureAutodiff<M, T> {
    pub fn new(nstates: usize, nout: usize, nparams: usize, ctx: M::C) -> Self {
        Self {
            nstates,
            nout,
            nparams,
            statistics: RefCell::new(OpStatistics::default()),
            ctx,
            _phantom: PhantomData,
        }
    }
}

impl<M: Matrix, T: NonLinearAutodiff<M>> Op for ClosureAutodiff<M, T> {
    type V = M::V;
    type T = M::T;
    type M = M;
    type C = M::C;
    fn nstates(&self) -> usize {
        self.nstates
    }
    fn nout(&self) -> usize {
        self.nout
    }
    fn nparams(&self) -> usize {
        self.nparams
    }
    fn statistics(&self) -> OpStatistics {
        self.statistics.borrow().clone()
    }
    fn context(&self) -> &Self::C {
        &self.ctx
    }
}

impl<M: Matrix, T: NonLinearAutodiff<M>> BuilderOp for ClosureAutodiff<M, T> {
    fn calculate_sparsity(&mut self, _y0: &Self::V, _t0: Self::T, _p: &Self::V) {}
    fn set_nstates(&mut self, nstates: usize) {
        self.nstates = nstates;
    }
    fn set_nout(&mut self, nout: usize) {
        self.nout = nout;
    }
    fn set_nparams(&mut self, nparams: usize) {
        self.nparams = nparams;
    }
}

fn to_vec<V: Vector>(v: &V, n: usize) -> Vec<V::T> {
    (0..n).map(|i| v.get_index(i)).collect()
}

fn write_back<V: Vector>(v: &mut V, data: &[V::T]) {
    for (i, &val) in data.iter().enumerate() {
        v.set_index(i, val);
    }
}

impl<M: Matrix, T: NonLinearAutodiff<M, T = M::T>> NonLinearOp
    for ParameterisedOp<'_, ClosureAutodiff<M, T>>
{
    fn call_inplace(&self, x: &M::V, _t: M::T, y: &mut M::V) {
        self.op.statistics.borrow_mut().increment_call();
        let x_slice = to_vec(x, self.op.nstates);
        let p_slice = to_vec(self.p, self.op.nparams);
        let mut y_slice = vec![M::T::zero(); self.op.nstates];
        T::rhs_inplace(&x_slice, &p_slice, &mut y_slice);
        write_back(y, &y_slice);
    }
}

impl<M: Matrix, T: NonLinearAutodiff<M, T = M::T>> NonLinearOpJacobian
    for ParameterisedOp<'_, ClosureAutodiff<M, T>>
{
    fn jac_mul_inplace(&self, x: &M::V, _t: M::T, v: &M::V, y: &mut M::V) {
        self.op.statistics.borrow_mut().increment_jac_mul();
        let x_slice = to_vec(x, self.op.nstates);
        let mut dx = to_vec(v, self.op.nstates);
        let p_slice = to_vec(self.p, self.op.nparams);
        let mut y_slice = vec![M::T::zero(); self.op.nstates];
        let mut dy = vec![M::T::zero(); self.op.nstates];
        T::rhs_jvp_inplace(&x_slice, &mut dx, &p_slice, &mut y_slice, &mut dy);
        write_back(y, &dy);
    }
    fn jacobian_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        self.op.statistics.borrow_mut().increment_matrix();
        self._default_jacobian_inplace(x, t, y);
    }
    fn jacobian_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        None
    }
}

impl<M: Matrix, T: NonLinearAutodiff<M, T = M::T>> NonLinearOpAdjoint
    for ParameterisedOp<'_, ClosureAutodiff<M, T>>
{
    fn jac_transpose_mul_inplace(&self, x: &Self::V, _t: Self::T, v: &Self::V, y: &mut Self::V) {
        self.op.statistics.borrow_mut().increment_jac_adj_mul();
        let x_slice = to_vec(x, self.op.nstates);
        let mut dx = vec![M::T::zero(); self.op.nstates];
        let p_slice = to_vec(self.p, self.op.nparams);
        let mut y_slice = vec![M::T::zero(); self.op.nstates];
        let mut dy = to_vec(v, self.op.nstates);
        T::rhs_vjp_inplace(&x_slice, &mut dx, &p_slice, &mut y_slice, &mut dy);
        for i in 0..self.op.nstates {
            y.set_index(i, -dx[i]);
        }
    }
    fn adjoint_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        self._default_adjoint_inplace(x, t, y);
    }
    fn adjoint_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        None
    }
}

impl<M: Matrix, T: NonLinearAutodiff<M, T = M::T>> NonLinearOpSensAdjoint
    for ParameterisedOp<'_, ClosureAutodiff<M, T>>
{
    fn sens_transpose_mul_inplace(
        &self,
        x: &Self::V,
        _t: Self::T,
        v: &Self::V,
        y: &mut Self::V,
    ) {
        let x_slice = to_vec(x, self.op.nstates);
        let p_slice = to_vec(self.p, self.op.nparams);
        let mut dp = vec![M::T::zero(); self.op.nparams];
        let mut y_slice = vec![M::T::zero(); self.op.nstates];
        let mut dy = to_vec(v, self.op.nstates);
        T::rhs_sens_vjp_inplace(&x_slice, &p_slice, &mut dp, &mut y_slice, &mut dy);
        for i in 0..self.op.nparams {
            y.set_index(i, -dp[i]);
        }
    }
    fn sens_adjoint_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        self._default_sens_adjoint_inplace(x, t, y);
    }
    fn sens_adjoint_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        None
    }
}
