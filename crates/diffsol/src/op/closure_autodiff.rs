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

    impl<M: Matrix, F: Fn(&M::V, &M::V, M::T, &mut M::V)> ClosureAutodiff<M, F> {
        #[autodiff_forward(call_jvp, Const, Dual, Const, Const, Dual)]
        #[autodiff_reverse(call_vjp, Const, Duplicated, Const, Const, Duplicated)]
        #[autodiff_reverse(call_sens_vjp, Const, Const, Duplicated, Const, Duplicated)]
        pub fn call_func(&self, x: &M::V, p: &M::V, t: M::T, y: &mut M::V) {
            (self.func)(x, p, t, y)
        }
    }

    impl<M: Matrix, F: Fn(&M::V, &M::V, M::T, &mut M::V)> NonLinearOp
        for ParameterisedOp<'_, ClosureAutodiff<M, F>>
    {
        fn call_inplace(&self, x: &M::V, t: M::T, y: &mut M::V) {
            self.op.statistics.borrow_mut().increment_call();
            self.op.call_func(x, self.p, t, y);
        }
    }

    impl<M: Matrix, F: Fn(&M::V, &M::V, M::T, &mut M::V)> NonLinearOpJacobian
        for ParameterisedOp<'_, ClosureAutodiff<M, F>>
    {
        fn jac_mul_inplace(&self, x: &M::V, t: M::T, v: &M::V, y: &mut M::V) {
            self.op.statistics.borrow_mut().increment_jac_mul();
            let mut tmp_nstates = self.op.tmp_nstates.borrow_mut();
            self.op.call_jvp(x, v, self.p, t, &mut tmp_nstates, y);
        }
        fn jacobian_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
            self.op.statistics.borrow_mut().increment_matrix();
            self._default_jacobian_inplace(x, t, y);
        }
        fn jacobian_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
            None
        }
    }

    impl<M: Matrix, F: Fn(&M::V, &M::V, M::T, &mut M::V)> NonLinearOpAdjoint
        for ParameterisedOp<'_, ClosureAutodiff<M, F>>
    {
        fn jac_transpose_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
            self.op.statistics.borrow_mut().increment_jac_adj_mul();
            let mut tmp_nstates = self.op.tmp_nstates.borrow_mut();
            let mut tmp_nstates2 = self.op.tmp_nstates2.borrow_mut();
            tmp_nstates.copy_from(v);
            y.fill(M::T::zero());
            self.op
                .call_vjp(x, y, self.p, t, &mut tmp_nstates2, &mut tmp_nstates);
            y.mul_assign(Scale(-M::T::one()));
        }
        fn adjoint_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
            self._default_adjoint_inplace(x, t, y);
        }
        fn adjoint_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
            None
        }
    }

    impl<M: Matrix, F: Fn(&M::V, &M::V, M::T, &mut M::V)> NonLinearOpSensAdjoint
        for ParameterisedOp<'_, ClosureAutodiff<M, F>>
    {
        fn sens_transpose_mul_inplace(
            &self,
            x: &Self::V,
            t: Self::T,
            v: &Self::V,
            y: &mut Self::V,
        ) {
            let mut tmp_nstates = self.op.tmp_nstates.borrow_mut();
            let mut tmp_nstates2 = self.op.tmp_nstates2.borrow_mut();
            tmp_nstates.copy_from(&v);
            y.fill(M::T::zero());
            self.op
                .call_sens_vjp(x, self.p, y, t, &mut tmp_nstates2, &mut tmp_nstates);
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

#[cfg(test)]
mod tests {
    use crate::{
        context::nalgebra::NalgebraContext, NalgebraMat, NalgebraVec, NonLinearOp,
        NonLinearOpAdjoint, NonLinearOpJacobian, NonLinearOpSensAdjoint, ParameterisedOp, Vector,
    };

    use super::ClosureAutodiff;

    type M = NalgebraMat<f64>;
    type V = NalgebraVec<f64>;

    fn nonlinear(x: &V, p: &V, t: f64, y: &mut V) {
        y[0] = p[0] * x[0] * x[0] + 2.0 * x[1] + t;
        y[1] = 3.0 * x[0] + p[1] * x[1] - t;
    }

    #[test]
    fn autodiff_closure_generates_state_and_parameter_gradients() {
        let ctx = NalgebraContext::default();
        let op = ClosureAutodiff::<M, _>::new(nonlinear, 2, 2, 2, ctx);
        let p = V::from_vec(vec![4.0, 5.0], ctx);
        let pop = ParameterisedOp::new(&op, &p);
        let x = V::from_vec(vec![2.0, 3.0], ctx);

        let mut value = V::zeros(2, ctx);
        pop.call_inplace(&x, 0.5, &mut value);
        value.assert_eq_st(&V::from_vec(vec![22.5, 20.5], ctx), 1e-12);

        let direction = V::from_vec(vec![7.0, 11.0], ctx);
        let mut jvp = V::zeros(2, ctx);
        pop.jac_mul_inplace(&x, 0.5, &direction, &mut jvp);
        jvp.assert_eq_st(&V::from_vec(vec![134.0, 76.0], ctx), 1e-12);

        let mut state_vjp = V::zeros(2, ctx);
        pop.jac_transpose_mul_inplace(&x, 0.5, &direction, &mut state_vjp);
        state_vjp.assert_eq_st(&V::from_vec(vec![-145.0, -69.0], ctx), 1e-12);

        let mut parameter_vjp = V::zeros(2, ctx);
        pop.sens_transpose_mul_inplace(&x, 0.5, &direction, &mut parameter_vjp);
        parameter_vjp.assert_eq_st(&V::from_vec(vec![-28.0, -33.0], ctx), 1e-12);
    }
}
