use std::{cell::RefCell, marker::PhantomData};

use crate::{
    Matrix, NonLinearOp, NonLinearOpAdjoint, NonLinearOpJacobian, NonLinearOpSensAdjoint, Op,
    Scale, Vector,
};
use num_traits::{One, Zero};
use std::ops::MulAssign;

use super::{BuilderOp, OpStatistics, ParameterisedOp};

/// An [`Op`] wrapping a user-provided closure with `std::autodiff` support.
///
/// When the `autodiff` feature is enabled, [`call_func`](Self::call_func) is annotated
/// with `#[autodiff_forward]` and `#[autodiff_reverse]` to automatically generate
/// `call_jvp`, `call_vjp`, and `call_sens_vjp`.
pub struct ClosureAutodiff<M: Matrix, F> {
    func: F,
    nstates: usize,
    nout: usize,
    nparams: usize,
    statistics: RefCell<OpStatistics>,
    tmp_nstates: RefCell<M::V>,
    tmp_nstates2: RefCell<M::V>,
    ctx: M::C,
    _phantom: PhantomData<M>,
}

impl<M: Matrix, F> ClosureAutodiff<M, F> {
    pub fn new(func: F, nstates: usize, nout: usize, nparams: usize, ctx: M::C) -> Self {
        Self {
            func,
            nstates,
            nout,
            nparams,
            tmp_nstates: RefCell::new(M::V::zeros(nstates, ctx.clone())),
            tmp_nstates2: RefCell::new(M::V::zeros(nstates, ctx.clone())),
            statistics: RefCell::new(OpStatistics::default()),
            ctx,
            _phantom: PhantomData,
        }
    }
}

impl<M: Matrix, F> Op for ClosureAutodiff<M, F> {
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

impl<M: Matrix, F> BuilderOp for ClosureAutodiff<M, F> {
    fn calculate_sparsity(&mut self, _y0: &Self::V, _t0: Self::T, _p: &Self::V) {}
    fn set_nstates(&mut self, nstates: usize) {
        self.nstates = nstates;
        self.tmp_nstates = RefCell::new(M::V::zeros(nstates, self.ctx.clone()));
        self.tmp_nstates2 = RefCell::new(M::V::zeros(nstates, self.ctx.clone()));
    }
    fn set_nout(&mut self, nout: usize) {
        self.nout = nout;
    }
    fn set_nparams(&mut self, nparams: usize) {
        self.nparams = nparams;
    }
}

#[cfg(feature = "autodiff")]
mod autodiff_impl {
    use super::*;
    use crate::Vector;
    use std::autodiff::{autodiff_forward, autodiff_reverse};

    impl<M: Matrix, F: Fn(&M::V, &M::V, &mut M::V)> ClosureAutodiff<M, F> {
        #[autodiff_forward(call_jvp, Const, Dual, Const, Dual)]
        #[autodiff_reverse(call_vjp, Const, Duplicated, Const, Duplicated)]
        #[autodiff_reverse(call_sens_vjp, Const, Const, Duplicated, Duplicated)]
        pub fn call_func(&self, x: &M::V, p: &M::V, y: &mut M::V) {
            (self.func)(x, p, y)
        }
    }

    impl<M: Matrix, F: Fn(&M::V, &M::V, &mut M::V)> NonLinearOp
        for ParameterisedOp<'_, ClosureAutodiff<M, F>>
    {
        fn call_inplace(&self, x: &M::V, _t: M::T, y: &mut M::V) {
            self.op.statistics.borrow_mut().increment_call();
            self.op.call_func(x, self.p, y);
        }
    }

    impl<M: Matrix, F: Fn(&M::V, &M::V, &mut M::V)> NonLinearOpJacobian
        for ParameterisedOp<'_, ClosureAutodiff<M, F>>
    {
        fn jac_mul_inplace(&self, x: &M::V, _t: M::T, v: &M::V, y: &mut M::V) {
            self.op.statistics.borrow_mut().increment_jac_mul();
            let mut tmp_nstates = self.op.tmp_nstates.borrow_mut();
            self.op.call_jvp(x, v, self.p, &mut tmp_nstates, y);
        }
        fn jacobian_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
            self.op.statistics.borrow_mut().increment_matrix();
            self._default_jacobian_inplace(x, t, y);
        }
        fn jacobian_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
            None
        }
    }

    impl<M: Matrix, F: Fn(&M::V, &M::V, &mut M::V)> NonLinearOpAdjoint
        for ParameterisedOp<'_, ClosureAutodiff<M, F>>
    {
        fn jac_transpose_mul_inplace(
            &self,
            x: &Self::V,
            _t: Self::T,
            v: &Self::V,
            y: &mut Self::V,
        ) {
            self.op.statistics.borrow_mut().increment_jac_adj_mul();
            let mut tmp_nstates = self.op.tmp_nstates.borrow_mut();
            let mut tmp_nstates2 = self.op.tmp_nstates2.borrow_mut();
            tmp_nstates.copy_from(v);
            y.fill(M::T::zero());
            self.op
                .call_vjp(x, y, self.p, &mut tmp_nstates2, &mut tmp_nstates);
            y.mul_assign(Scale(-M::T::one()));
        }
        fn adjoint_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
            self._default_adjoint_inplace(x, t, y);
        }
        fn adjoint_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
            None
        }
    }

    impl<M: Matrix, F: Fn(&M::V, &M::V, &mut M::V)> NonLinearOpSensAdjoint
        for ParameterisedOp<'_, ClosureAutodiff<M, F>>
    {
        fn sens_transpose_mul_inplace(
            &self,
            x: &Self::V,
            _t: Self::T,
            v: &Self::V,
            y: &mut Self::V,
        ) {
            let mut tmp_nstates = self.op.tmp_nstates.borrow_mut();
            let mut tmp_nstates2 = self.op.tmp_nstates2.borrow_mut();
            tmp_nstates.copy_from(&v);
            y.fill(M::T::zero());
            self.op
                .call_sens_vjp(x, self.p, y, &mut tmp_nstates2, &mut tmp_nstates);
            y.mul_assign(Scale(-M::T::one()));
        }
        fn sens_adjoint_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
            self._default_sens_adjoint_inplace(x, t, y);
        }
        fn sens_adjoint_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
            None
        }
    }
}
