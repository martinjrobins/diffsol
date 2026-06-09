use std::marker::PhantomData;

use crate::{Matrix, Op, Vector};

use super::BuilderOp;

/// An [`Op`] wrapping a user-provided initial-condition closure with `std::autodiff` support.
pub struct ConstantClosureAutodiff<M: Matrix, F> {
    func: F,
    nout: usize,
    nparams: usize,
    ctx: M::C,
    _phantom: PhantomData<M>,
}

impl<M: Matrix, F> ConstantClosureAutodiff<M, F> {
    pub fn new(func: F, nout: usize, nparams: usize, ctx: M::C) -> Self {
        Self {
            func,
            nout,
            nparams,
            ctx,
            _phantom: PhantomData,
        }
    }
}

impl<M: Matrix, F> BuilderOp for ConstantClosureAutodiff<M, F> {
    fn calculate_sparsity(&mut self, _y0: &Self::V, _t0: Self::T, _p: &Self::V) {}
    fn set_nstates(&mut self, _nstates: usize) {}
    fn set_nout(&mut self, nout: usize) {
        self.nout = nout;
    }
    fn set_nparams(&mut self, nparams: usize) {
        self.nparams = nparams;
    }
}

impl<M: Matrix, F> Op for ConstantClosureAutodiff<M, F> {
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

#[cfg(feature = "autodiff")]
mod autodiff_impl {
    use super::*;
    use crate::{ConstantOp, ConstantOpSensAdjoint, ParameterisedOp, VectorIndex};
    use num_traits::Zero;
    use std::autodiff::autodiff_reverse;

    impl<M: Matrix, F: Fn(&[M::T], &mut [M::T])> ConstantClosureAutodiff<M, F> {
        #[autodiff_reverse(call_sens_vjp, Const, Duplicated, Duplicated)]
        pub fn call_func(&self, p: &[M::T], y: &mut [M::T]) {
            (self.func)(p, y)
        }
    }

    impl<M: Matrix, F: Fn(&[M::T], &mut [M::T])> ConstantOp
        for ParameterisedOp<'_, ConstantClosureAutodiff<M, F>>
    {
        fn call_inplace(&self, _t: Self::T, y: &mut Self::V) {
            let p_slice = to_vec(self.p, self.p.len());
            let mut y_slice = vec![M::T::zero(); self.op.nout];
            self.op.call_func(&p_slice, &mut y_slice);
            y.set_index(0, y_slice[0]);
        }
    }

    impl<M: Matrix, F: Fn(&[M::T], &mut [M::T])> ConstantOpSensAdjoint
        for ParameterisedOp<'_, ConstantClosureAutodiff<M, F>>
    {
        fn sens_transpose_mul_inplace(&self, _t: Self::T, v: &Self::V, y: &mut Self::V) {
            let p_slice = to_vec(self.p, self.p.len());
            let mut dp = vec![M::T::zero(); self.p.len()];
            let mut y_slice = vec![M::T::zero(); self.op.nout];
            let mut dy = vec![v.get_index(0); self.op.nout];
            self.op
                .call_sens_vjp(&p_slice, &mut dp, &mut y_slice, &mut dy);
            for i in 0..self.p.len() {
                y.set_index(i, -dp[i]);
            }
        }
    }
}
