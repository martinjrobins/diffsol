use std::marker::PhantomData;

use crate::{Matrix, Op, Scale, Vector};
use num_traits::One;
use std::cell::RefCell;
use std::ops::MulAssign;

use super::BuilderOp;

/// An [`Op`] wrapping a user-provided initial-condition closure with `std::autodiff` support.
///
/// When the `autodiff` feature is enabled, [`call_func`](Self::call_func) is annotated
/// with `#[autodiff_reverse]` to automatically generate `call_sens_vjp`.
pub struct ConstantClosureAutodiff<M: Matrix, F> {
    func: F,
    nout: usize,
    nparams: usize,
    tmp_nstates: RefCell<M::V>,
    tmp_nstates2: RefCell<M::V>,
    ctx: M::C,
    _phantom: PhantomData<M>,
}

impl<M: Matrix, F> ConstantClosureAutodiff<M, F> {
    pub fn new(func: F, nout: usize, nparams: usize, ctx: M::C) -> Self {
        Self {
            func,
            nout,
            nparams,
            tmp_nstates: RefCell::new(M::V::zeros(0, ctx.clone())),
            tmp_nstates2: RefCell::new(M::V::zeros(0, ctx.clone())),
            ctx,
            _phantom: PhantomData,
        }
    }
}

impl<M: Matrix, F> BuilderOp for ConstantClosureAutodiff<M, F> {
    fn calculate_sparsity(&mut self, _y0: &Self::V, _t0: Self::T, _p: &Self::V) {}
    fn set_nstates(&mut self, nstates: usize) {
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

#[cfg(feature = "autodiff")]
mod autodiff_impl {
    use super::*;
    use crate::{ConstantOp, ConstantOpSensAdjoint, ParameterisedOp, Vector};
    use std::autodiff::autodiff_reverse;

    impl<M: Matrix, F: Fn(&M::V, &mut M::V)> ConstantClosureAutodiff<M, F> {
        #[autodiff_reverse(call_sens_vjp, Const, Duplicated, Duplicated)]
        pub fn call_func(&self, p: &M::V, y: &mut M::V) {
            (self.func)(p, y)
        }
    }

    impl<M: Matrix, F: Fn(&M::V, &mut M::V)> ConstantOp
        for ParameterisedOp<'_, ConstantClosureAutodiff<M, F>>
    {
        fn call_inplace(&self, _t: Self::T, y: &mut Self::V) {
            self.op.call_func(self.p, y);
        }
    }

    impl<M: Matrix, F: Fn(&M::V, &mut M::V)> ConstantOpSensAdjoint
        for ParameterisedOp<'_, ConstantClosureAutodiff<M, F>>
    {
        fn sens_transpose_mul_inplace(&self, _t: Self::T, v: &Self::V, y: &mut Self::V) {
            let mut tmp_nstates = self.op.tmp_nstates.borrow_mut();
            let mut tmp_nstates2 = self.op.tmp_nstates2.borrow_mut();
            tmp_nstates.copy_from(v);
            self.op
                .call_sens_vjp(self.p, y, &mut tmp_nstates2, &mut tmp_nstates);
            y.mul_assign(Scale(-M::T::one()));
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        context::nalgebra::NalgebraContext, BuilderOp, ConstantOp, ConstantOpSensAdjoint,
        NalgebraMat, NalgebraVec, ParameterisedOp, Vector,
    };

    use super::ConstantClosureAutodiff;

    type M = NalgebraMat<f64>;
    type V = NalgebraVec<f64>;

    fn initial_condition(p: &V, y: &mut V) {
        y[0] = p[0] * p[0] + 2.0 * p[1];
        y[1] = 3.0 * p[0] + p[1];
    }

    #[test]
    fn autodiff_constant_closure_generates_parameter_gradient() {
        let ctx = NalgebraContext::default();
        let mut op = ConstantClosureAutodiff::<M, _>::new(initial_condition, 2, 2, ctx);
        op.set_nstates(2);
        let p = V::from_vec(vec![2.0, 5.0], ctx);
        let pop = ParameterisedOp::new(&op, &p);

        let mut value = V::zeros(2, ctx);
        pop.call_inplace(0.0, &mut value);
        value.assert_eq_st(&V::from_vec(vec![14.0, 11.0], ctx), 1e-12);

        let seed = V::from_vec(vec![7.0, 11.0], ctx);
        let mut parameter_vjp = V::zeros(2, ctx);
        pop.sens_transpose_mul_inplace(0.0, &seed, &mut parameter_vjp);
        parameter_vjp.assert_eq_st(&V::from_vec(vec![-61.0, -25.0], ctx), 1e-12);
    }
}
