use std::cell::RefCell;

use crate::{
    find_jacobian_non_zeros, jacobian::JacobianColoring, Matrix, MatrixSparsity, NonLinearOp,
    NonLinearOpJacobian, Op, Vector,
};

use super::{BuilderOp, OpStatistics, ParameterisedOp};

pub struct Closure<M, F, G>
where
    M: Matrix,
    F: Fn(&[M::T], &[M::T], M::T, &mut [M::T]),
    G: Fn(&[M::T], &[M::T], M::T, &[M::T], &mut [M::T]),
{
    func: F,
    jacobian_action: G,
    nstates: usize,
    nout: usize,
    nparams: usize,
    coloring: Option<JacobianColoring<M>>,
    sparsity: Option<M::Sparsity>,
    statistics: RefCell<OpStatistics>,
    ctx: M::C,
}

impl<M, F, G> Closure<M, F, G>
where
    M: Matrix,
    F: Fn(&[M::T], &[M::T], M::T, &mut [M::T]),
    G: Fn(&[M::T], &[M::T], M::T, &[M::T], &mut [M::T]),
{
    pub fn new(
        func: F,
        jacobian_action: G,
        nstates: usize,
        nout: usize,
        nparams: usize,
        ctx: M::C,
    ) -> Self {
        Self {
            func,
            jacobian_action,
            nstates,
            nparams,
            nout,
            statistics: RefCell::new(OpStatistics::default()),
            coloring: None,
            sparsity: None,
            ctx,
        }
    }
    pub fn calculate_sparsity(&mut self, y0: &M::V, t0: M::T, p: &M::V) {
        let param_op = ParameterisedOp { op: self, p };
        let non_zeros = find_jacobian_non_zeros(&param_op, y0, t0);
        self.sparsity = Some(
            MatrixSparsity::try_from_indices(self.nout(), self.nstates(), non_zeros.clone())
                .expect("invalid sparsity pattern"),
        );
        self.coloring = Some(JacobianColoring::new(
            self.sparsity.as_ref().unwrap(),
            &non_zeros,
            self.ctx.clone(),
        ));
    }
}

impl<M, F, G> BuilderOp for Closure<M, F, G>
where
    M: Matrix,
    F: Fn(&[M::T], &[M::T], M::T, &mut [M::T]),
    G: Fn(&[M::T], &[M::T], M::T, &[M::T], &mut [M::T]),
{
    fn calculate_sparsity(&mut self, y0: &M::V, t0: M::T, p: &M::V) {
        self.calculate_sparsity(y0, t0, p);
    }

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

impl<M, F, G> Op for Closure<M, F, G>
where
    M: Matrix,
    F: Fn(&[M::T], &[M::T], M::T, &mut [M::T]),
    G: Fn(&[M::T], &[M::T], M::T, &[M::T], &mut [M::T]),
{
    type V = M::V;
    type T = M::T;
    type M = M;
    type C = M::C;

    fn context(&self) -> &Self::C {
        &self.ctx
    }

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
}

impl<M, F, G> NonLinearOp for ParameterisedOp<'_, Closure<M, F, G>>
where
    M: Matrix,
    F: Fn(&[M::T], &[M::T], M::T, &mut [M::T]),
    G: Fn(&[M::T], &[M::T], M::T, &[M::T], &mut [M::T]),
{
    fn call_inplace(&self, x: &M::V, t: M::T, y: &mut M::V) {
        self.op.statistics.borrow_mut().increment_call();
        y.for_each_batch([x, self.p], |y, [x, p], _| (self.op.func)(x, p, t, y));
    }
}

impl<M, F, G> NonLinearOpJacobian for ParameterisedOp<'_, Closure<M, F, G>>
where
    M: Matrix,
    F: Fn(&[M::T], &[M::T], M::T, &mut [M::T]),
    G: Fn(&[M::T], &[M::T], M::T, &[M::T], &mut [M::T]),
{
    fn jac_mul_inplace(&self, x: &M::V, t: M::T, v: &M::V, y: &mut M::V) {
        self.op.statistics.borrow_mut().increment_jac_mul();
        y.for_each_batch([x, self.p, v], |y, [x, p, v], _| {
            (self.op.jacobian_action)(x, p, t, v, y)
        });
    }
    fn jacobian_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        self.op.statistics.borrow_mut().increment_matrix();
        if let Some(coloring) = self.op.coloring.as_ref() {
            coloring.jacobian_inplace(self, x, t, y);
        } else {
            self._default_jacobian_inplace(x, t, y);
        }
    }
    fn jacobian_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.op.sparsity.clone()
    }
}
