use std::cell::RefCell;

use crate::{
    jacobian::{
        find_adjoint_non_zeros, find_jacobian_non_zeros, find_sens_adjoint_non_zeros,
        JacobianColoring,
    },
    Matrix, MatrixSparsity, NonLinearOp, NonLinearOpAdjoint, NonLinearOpJacobian,
    NonLinearOpSensAdjoint, Op, Vector,
};

use super::{BuilderOp, OpStatistics, ParameterisedOp};

#[derive(Clone)]
pub struct ClosureWithAdjoint<M, F, G, H, I>
where
    M: Matrix,
    F: Fn(&[M::T], &[M::T], M::T, &mut [M::T]),
    G: Fn(&[M::T], &[M::T], M::T, &[M::T], &mut [M::T]),
    H: Fn(&[M::T], &[M::T], M::T, &[M::T], &mut [M::T]),
    I: Fn(&[M::T], &[M::T], M::T, &[M::T], &mut [M::T]),
{
    func: F,
    jacobian_action: G,
    jacobian_adjoint_action: H,
    sens_adjoint_action: I,
    nstates: usize,
    nout: usize,
    nparams: usize,
    coloring: Option<JacobianColoring<M>>,
    sparsity: Option<M::Sparsity>,
    sparsity_adjoint: Option<M::Sparsity>,
    coloring_adjoint: Option<JacobianColoring<M>>,
    sens_sparsity: Option<M::Sparsity>,
    coloring_sens_adjoint: Option<JacobianColoring<M>>,
    statistics: RefCell<OpStatistics>,
    ctx: M::C,
}

impl<M, F, G, H, I> ClosureWithAdjoint<M, F, G, H, I>
where
    M: Matrix,
    F: Fn(&[M::T], &[M::T], M::T, &mut [M::T]),
    G: Fn(&[M::T], &[M::T], M::T, &[M::T], &mut [M::T]),
    H: Fn(&[M::T], &[M::T], M::T, &[M::T], &mut [M::T]),
    I: Fn(&[M::T], &[M::T], M::T, &[M::T], &mut [M::T]),
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        func: F,
        jacobian_action: G,
        jacobian_adjoint_action: H,
        sens_adjoint_action: I,
        nstates: usize,
        nout: usize,
        nparams: usize,
        ctx: M::C,
    ) -> Self {
        Self {
            func,
            jacobian_action,
            jacobian_adjoint_action,
            sens_adjoint_action,
            nstates,
            nout,
            nparams,
            statistics: RefCell::new(OpStatistics::default()),
            coloring: None,
            sparsity: None,
            sparsity_adjoint: None,
            coloring_adjoint: None,
            sens_sparsity: None,
            coloring_sens_adjoint: None,
            ctx,
        }
    }

    pub fn calculate_jacobian_sparsity(&mut self, y0: &M::V, t0: M::T, p: &M::V) {
        let op = ParameterisedOp { op: self, p };
        let non_zeros = find_jacobian_non_zeros(&op, y0, t0);
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

    pub fn calculate_adjoint_sparsity(&mut self, y0: &M::V, t0: M::T, p: &M::V) {
        let op = ParameterisedOp { op: self, p };
        let non_zeros = find_adjoint_non_zeros(&op, y0, t0);
        self.sparsity_adjoint = Some(
            MatrixSparsity::try_from_indices(self.nstates, self.nout, non_zeros.clone())
                .expect("invalid sparsity pattern"),
        );
        self.coloring_adjoint = Some(JacobianColoring::new(
            self.sparsity_adjoint.as_ref().unwrap(),
            &non_zeros,
            self.ctx.clone(),
        ));
    }

    pub fn calculate_sens_adjoint_sparsity(&mut self, y0: &M::V, t0: M::T, p: &M::V) {
        let op = ParameterisedOp { op: self, p };
        let non_zeros = find_sens_adjoint_non_zeros(&op, y0, t0);
        let nparams = p.len();
        // `-g_p^T` maps outputs to parameters: the same as `nstates` for the rhs, but not for an
        // out operator
        self.sens_sparsity = Some(
            MatrixSparsity::try_from_indices(nparams, self.nout(), non_zeros.clone())
                .expect("invalid sparsity pattern"),
        );
        self.coloring_sens_adjoint = Some(JacobianColoring::new(
            self.sens_sparsity.as_ref().unwrap(),
            &non_zeros,
            self.ctx.clone(),
        ));
    }
}

impl<M, F, G, H, I> Op for ClosureWithAdjoint<M, F, G, H, I>
where
    M: Matrix,
    F: Fn(&[M::T], &[M::T], M::T, &mut [M::T]),
    G: Fn(&[M::T], &[M::T], M::T, &[M::T], &mut [M::T]),
    H: Fn(&[M::T], &[M::T], M::T, &[M::T], &mut [M::T]),
    I: Fn(&[M::T], &[M::T], M::T, &[M::T], &mut [M::T]),
{
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

impl<M, F, G, H, I> BuilderOp for ClosureWithAdjoint<M, F, G, H, I>
where
    M: Matrix,
    F: Fn(&[M::T], &[M::T], M::T, &mut [M::T]),
    G: Fn(&[M::T], &[M::T], M::T, &[M::T], &mut [M::T]),
    H: Fn(&[M::T], &[M::T], M::T, &[M::T], &mut [M::T]),
    I: Fn(&[M::T], &[M::T], M::T, &[M::T], &mut [M::T]),
{
    fn calculate_sparsity(&mut self, y0: &Self::V, t0: Self::T, p: &Self::V) {
        self.calculate_jacobian_sparsity(y0, t0, p);
        self.calculate_adjoint_sparsity(y0, t0, p);
        self.calculate_sens_adjoint_sparsity(y0, t0, p);
    }

    fn calculate_augmented_sparsity(&mut self, y0: &Self::V, t0: Self::T, p: &Self::V) {
        self.calculate_adjoint_sparsity(y0, t0, p);
        self.calculate_sens_adjoint_sparsity(y0, t0, p);
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

impl<M, F, G, H, I> NonLinearOp for ParameterisedOp<'_, ClosureWithAdjoint<M, F, G, H, I>>
where
    M: Matrix,
    F: Fn(&[M::T], &[M::T], M::T, &mut [M::T]),
    G: Fn(&[M::T], &[M::T], M::T, &[M::T], &mut [M::T]),
    H: Fn(&[M::T], &[M::T], M::T, &[M::T], &mut [M::T]),
    I: Fn(&[M::T], &[M::T], M::T, &[M::T], &mut [M::T]),
{
    fn call_inplace(&self, x: &M::V, t: M::T, y: &mut M::V) {
        self.op.statistics.borrow_mut().increment_call();
        y.for_each_batch([x, self.p], |y, [x, p], _| (self.op.func)(x, p, t, y));
    }
}

impl<M, F, G, H, I> NonLinearOpJacobian for ParameterisedOp<'_, ClosureWithAdjoint<M, F, G, H, I>>
where
    M: Matrix,
    F: Fn(&[M::T], &[M::T], M::T, &mut [M::T]),
    G: Fn(&[M::T], &[M::T], M::T, &[M::T], &mut [M::T]),
    H: Fn(&[M::T], &[M::T], M::T, &[M::T], &mut [M::T]),
    I: Fn(&[M::T], &[M::T], M::T, &[M::T], &mut [M::T]),
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

impl<M, F, G, H, I> NonLinearOpAdjoint for ParameterisedOp<'_, ClosureWithAdjoint<M, F, G, H, I>>
where
    M: Matrix,
    F: Fn(&[M::T], &[M::T], M::T, &mut [M::T]),
    G: Fn(&[M::T], &[M::T], M::T, &[M::T], &mut [M::T]),
    H: Fn(&[M::T], &[M::T], M::T, &[M::T], &mut [M::T]),
    I: Fn(&[M::T], &[M::T], M::T, &[M::T], &mut [M::T]),
{
    fn jac_transpose_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        self.op.statistics.borrow_mut().increment_jac_adj_mul();
        y.for_each_batch([x, self.p, v], |y, [x, p, v], _| {
            (self.op.jacobian_adjoint_action)(x, p, t, v, y)
        });
    }

    fn adjoint_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        if let Some(coloring) = self.op.coloring_adjoint.as_ref() {
            coloring.adjoint_inplace(self, x, t, y);
        } else {
            self._default_adjoint_inplace(x, t, y);
        }
    }
    fn adjoint_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.op.sparsity_adjoint.clone()
    }
}

impl<M, F, G, H, I> NonLinearOpSensAdjoint
    for ParameterisedOp<'_, ClosureWithAdjoint<M, F, G, H, I>>
where
    M: Matrix,
    F: Fn(&[M::T], &[M::T], M::T, &mut [M::T]),
    G: Fn(&[M::T], &[M::T], M::T, &[M::T], &mut [M::T]),
    H: Fn(&[M::T], &[M::T], M::T, &[M::T], &mut [M::T]),
    I: Fn(&[M::T], &[M::T], M::T, &[M::T], &mut [M::T]),
{
    fn sens_transpose_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        y.for_each_batch([x, self.p, v], |y, [x, p, v], _| {
            (self.op.sens_adjoint_action)(x, p, t, v, y)
        });
    }
    fn sens_adjoint_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        if let Some(coloring) = self.op.coloring_sens_adjoint.as_ref() {
            coloring.sens_adjoint_inplace(self, x, t, y);
        } else {
            self._default_sens_adjoint_inplace(x, t, y);
        }
    }
    fn sens_adjoint_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.op.sens_sparsity.clone()
    }
}
