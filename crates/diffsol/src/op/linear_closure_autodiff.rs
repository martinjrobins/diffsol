use std::{cell::RefCell, marker::PhantomData};

use crate::{LinearOp, LinearOpTranspose, Matrix, Op, Vector};
use num_traits::Zero;

use super::{BuilderOp, OpStatistics, ParameterisedOp};

/// A linear [`Op`] wrapping a user-provided closure with `std::autodiff` support.
///
/// The closure has the same GEMV contract as [`LinearClosure`](super::LinearClosure):
/// `y = M * x + beta * y`. Reverse autodiff generates the transpose product.
pub struct LinearClosureAutodiff<M: Matrix, F> {
    func: F,
    nstates: usize,
    nout: usize,
    nparams: usize,
    statistics: RefCell<OpStatistics>,
    tmp_input: RefCell<M::V>,
    tmp_output: RefCell<M::V>,
    tmp_input_adjoint: RefCell<M::V>,
    tmp_output_adjoint: RefCell<M::V>,
    ctx: M::C,
    _phantom: PhantomData<M>,
}

impl<M: Matrix, F> LinearClosureAutodiff<M, F> {
    pub fn new(func: F, nstates: usize, nout: usize, nparams: usize, ctx: M::C) -> Self {
        Self {
            func,
            nstates,
            nout,
            nparams,
            statistics: RefCell::new(OpStatistics::default()),
            tmp_input: RefCell::new(M::V::zeros(nstates, ctx.clone())),
            tmp_output: RefCell::new(M::V::zeros(nout, ctx.clone())),
            tmp_input_adjoint: RefCell::new(M::V::zeros(nstates, ctx.clone())),
            tmp_output_adjoint: RefCell::new(M::V::zeros(nout, ctx.clone())),
            ctx,
            _phantom: PhantomData,
        }
    }
}

impl<M: Matrix, F> Op for LinearClosureAutodiff<M, F> {
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

impl<M: Matrix, F> BuilderOp for LinearClosureAutodiff<M, F> {
    fn calculate_sparsity(&mut self, _y0: &Self::V, _t0: Self::T, _p: &Self::V) {}

    fn set_nstates(&mut self, nstates: usize) {
        self.nstates = nstates;
        self.tmp_input = RefCell::new(M::V::zeros(nstates, self.ctx.clone()));
        self.tmp_input_adjoint = RefCell::new(M::V::zeros(nstates, self.ctx.clone()));
    }

    fn set_nout(&mut self, nout: usize) {
        self.nout = nout;
        self.tmp_output = RefCell::new(M::V::zeros(nout, self.ctx.clone()));
        self.tmp_output_adjoint = RefCell::new(M::V::zeros(nout, self.ctx.clone()));
    }

    fn set_nparams(&mut self, nparams: usize) {
        self.nparams = nparams;
    }
}

#[cfg(feature = "autodiff")]
mod autodiff_impl {
    use super::*;
    use std::autodiff::autodiff_reverse;

    impl<M: Matrix, F: Fn(&[M::T], &[M::T], M::T, M::T, &mut [M::T])> LinearClosureAutodiff<M, F> {
        #[autodiff_reverse(call_vjp, Const, Duplicated, Const, Const, Const, Duplicated)]
        pub fn call_func(&self, x: &[M::T], p: &[M::T], t: M::T, beta: M::T, y: &mut [M::T]) {
            (self.func)(x, p, t, beta, y)
        }
    }

    impl<M: Matrix, F: Fn(&[M::T], &[M::T], M::T, M::T, &mut [M::T])> LinearOp
        for ParameterisedOp<'_, LinearClosureAutodiff<M, F>>
    {
        fn gemv_inplace(&self, x: &M::V, t: M::T, beta: M::T, y: &mut M::V) {
            self.op.statistics.borrow_mut().increment_call();
            y.for_each_batch([x, self.p], |y, [x, p], _| {
                self.op.call_func(x, p, t, beta, y)
            });
        }
    }

    impl<M: Matrix, F: Fn(&[M::T], &[M::T], M::T, M::T, &mut [M::T])> LinearOpTranspose
        for ParameterisedOp<'_, LinearClosureAutodiff<M, F>>
    {
        fn gemv_transpose_inplace(&self, x: &M::V, t: M::T, beta: M::T, y: &mut M::V) {
            let tmp_input = self.op.tmp_input.borrow();
            let mut tmp_output = self.op.tmp_output.borrow_mut();
            let mut tmp_input_adjoint = self.op.tmp_input_adjoint.borrow_mut();
            let mut tmp_output_adjoint = self.op.tmp_output_adjoint.borrow_mut();
            <M::V as Vector>::for_each_batch_mut(
                [
                    y,
                    &mut tmp_input_adjoint,
                    &mut tmp_output,
                    &mut tmp_output_adjoint,
                ],
                [x, &tmp_input, self.p],
                |[y, tmp_input_adjoint, tmp_output, tmp_output_adjoint], [x, tmp_input, p], _| {
                    tmp_output.fill(M::T::zero());
                    tmp_input_adjoint.fill(M::T::zero());
                    tmp_output_adjoint.copy_from_slice(x);
                    self.op.call_vjp(
                        tmp_input,
                        tmp_input_adjoint,
                        p,
                        t,
                        M::T::zero(),
                        tmp_output,
                        tmp_output_adjoint,
                    );
                    for (y, adj) in y.iter_mut().zip(tmp_input_adjoint.iter()) {
                        *y = *adj + beta * *y;
                    }
                },
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        context::nalgebra::NalgebraContext, LinearOp, LinearOpTranspose, NalgebraMat, NalgebraVec,
        ParameterisedOp, Vector,
    };

    use super::LinearClosureAutodiff;

    type M = NalgebraMat<f64>;
    type V = NalgebraVec<f64>;

    fn mass(x: &[f64], p: &[f64], _t: f64, beta: f64, y: &mut [f64]) {
        let out = [p[0] * x[0] + 2.0 * x[1], 3.0 * x[0] + p[1] * x[1]];
        for (y, out) in y.iter_mut().zip(out.iter()) {
            *y = *out + beta * *y;
        }
    }

    #[test]
    fn autodiff_linear_closure_applies_mass_and_transpose() {
        let ctx = NalgebraContext::default();
        let op = LinearClosureAutodiff::<M, _>::new(mass, 2, 2, 2, ctx);
        let p = V::from_vec(vec![4.0, 5.0], ctx);
        let pop = ParameterisedOp::new(&op, &p);
        let x = V::from_vec(vec![5.0, 7.0], ctx);

        let mut y = V::from_vec(vec![11.0, 13.0], ctx);
        pop.gemv_inplace(&x, 0.0, 0.5, &mut y);
        y.assert_eq_st(&V::from_vec(vec![39.5, 56.5], ctx), 1e-12);

        let mut y_transpose = V::from_vec(vec![11.0, 13.0], ctx);
        pop.gemv_transpose_inplace(&x, 0.0, 0.5, &mut y_transpose);
        y_transpose.assert_eq_st(&V::from_vec(vec![46.5, 51.5], ctx), 1e-12);
    }
}
