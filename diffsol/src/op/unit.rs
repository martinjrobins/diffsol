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

#[cfg(test)]
mod tests {
    use crate::{
        context::nalgebra::NalgebraContext, matrix::dense_nalgebra_serial::NalgebraMat, Context,
        LinearOp, LinearOpSens, LinearOpTranspose, NonLinearOp, NonLinearOpAdjoint,
        NonLinearOpJacobian, NonLinearOpSens, NonLinearOpSensAdjoint, Op, Vector,
    };

    use super::{BuilderOp, ParameterisedOp, UnitCallable};

    type M = NalgebraMat<f64>;

    fn assert_raw_and_parameterised_unit(op: &UnitCallable<M>) {
        let ctx = *op.context();
        let x = ctx.vector_from_vec(vec![1.0, -2.0, 3.0]);
        let v = ctx.vector_from_vec(vec![0.5, -1.5, 2.0]);
        let mut y: crate::NalgebraVec<f64> = ctx.vector_from_vec(vec![9.0, 8.0, 7.0]);

        NonLinearOp::call_inplace(op, &x, 0.0, &mut y);
        y.assert_eq_st(&x, 1e-12);

        y = ctx.vector_from_vec(vec![9.0, 8.0, 7.0]);
        op.gemv_inplace(&x, 0.0, 2.0, &mut y);
        y.assert_eq_st(&ctx.vector_from_vec(vec![19.0, 14.0, 17.0]), 1e-12);

        y = ctx.vector_from_vec(vec![9.0, 8.0, 7.0]);
        op.gemv_transpose_inplace(&x, 0.0, 2.0, &mut y);
        y.assert_eq_st(&ctx.vector_from_vec(vec![19.0, 14.0, 17.0]), 1e-12);

        y = ctx.vector_zeros(3);
        op.jac_mul_inplace(&x, 0.0, &v, &mut y);
        y.assert_eq_st(&v, 1e-12);

        y = ctx.vector_zeros(3);
        op.jac_transpose_mul_inplace(&x, 0.0, &v, &mut y);
        y.assert_eq_st(&v, 1e-12);

        y = ctx.vector_from_vec(vec![1.0, 1.0, 1.0]);
        <UnitCallable<M> as NonLinearOpSens>::sens_mul_inplace(op, &x, 0.0, &v, &mut y);
        y.assert_eq_st(&ctx.vector_zeros(3), 1e-12);

        y = ctx.vector_from_vec(vec![1.0, 1.0, 1.0]);
        <UnitCallable<M> as NonLinearOpSensAdjoint>::sens_transpose_mul_inplace(
            op, &x, 0.0, &v, &mut y,
        );
        y.assert_eq_st(&ctx.vector_zeros(3), 1e-12);

        y = ctx.vector_from_vec(vec![1.0, 1.0, 1.0]);
        <UnitCallable<M> as LinearOpSens>::sens_mul_inplace(op, &x, 0.0, &v, &mut y);
        y.assert_eq_st(&ctx.vector_zeros(3), 1e-12);

        let p = ctx.vector_zeros(0);
        let pop = ParameterisedOp::new(op, &p);

        y = ctx.vector_zeros(3);
        NonLinearOp::call_inplace(&pop, &x, 0.0, &mut y);
        y.assert_eq_st(&x, 1e-12);

        y = ctx.vector_from_vec(vec![1.0, 2.0, 3.0]);
        pop.gemv_inplace(&x, 0.0, -1.0, &mut y);
        y.assert_eq_st(&ctx.vector_from_vec(vec![0.0, -4.0, 0.0]), 1e-12);

        y = ctx.vector_zeros(3);
        pop.jac_mul_inplace(&x, 0.0, &v, &mut y);
        y.assert_eq_st(&v, 1e-12);

        y = ctx.vector_zeros(3);
        pop.jac_transpose_mul_inplace(&x, 0.0, &v, &mut y);
        y.assert_eq_st(&v, 1e-12);

        y = ctx.vector_from_vec(vec![1.0, 1.0, 1.0]);
        <ParameterisedOp<'_, UnitCallable<M>> as NonLinearOpSens>::sens_mul_inplace(
            &pop, &x, 0.0, &v, &mut y,
        );
        y.assert_eq_st(&ctx.vector_zeros(3), 1e-12);

        y = ctx.vector_from_vec(vec![1.0, 1.0, 1.0]);
        <ParameterisedOp<'_, UnitCallable<M>> as NonLinearOpSensAdjoint>::sens_transpose_mul_inplace(
            &pop, &x, 0.0, &v, &mut y,
        );
        y.assert_eq_st(&ctx.vector_zeros(3), 1e-12);

        y = ctx.vector_from_vec(vec![1.0, 1.0, 1.0]);
        <ParameterisedOp<'_, UnitCallable<M>> as LinearOpSens>::sens_mul_inplace(
            &pop, &x, 0.0, &v, &mut y,
        );
        y.assert_eq_st(&ctx.vector_zeros(3), 1e-12);

        y = ctx.vector_from_vec(vec![1.0, 2.0, 3.0]);
        pop.gemv_transpose_inplace(&x, 0.0, 1.0, &mut y);
        y.assert_eq_st(&ctx.vector_from_vec(vec![2.0, 0.0, 6.0]), 1e-12);
    }

    #[test]
    fn unit_callable_behaves_as_identity_and_zero_sens_operator() {
        let ctx = NalgebraContext;
        let mut op = UnitCallable::<M>::default();
        assert_eq!(op.nstates(), 1);
        assert_eq!(op.nout(), 1);
        assert_eq!(op.nparams(), 0);

        op.set_nout(3);
        op.set_nstates(3);
        op.set_nparams(99);
        op.calculate_sparsity(&ctx.vector_zeros(3), 0.0, &ctx.vector_zeros(0));

        assert_eq!(op.nstates(), 3);
        assert_eq!(op.nout(), 3);
        assert_eq!(op.nparams(), 0);

        assert_raw_and_parameterised_unit(&op);
    }

    #[test]
    fn unit_callable_new_uses_supplied_context() {
        let op = UnitCallable::<M>::new(3, NalgebraContext);
        assert_eq!(op.nstates(), 3);
        assert_eq!(op.nout(), 3);
        assert_eq!(op.nparams(), 0);
        assert_raw_and_parameterised_unit(&op);
    }
}
