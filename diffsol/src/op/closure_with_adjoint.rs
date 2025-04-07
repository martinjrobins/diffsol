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
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    H: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    I: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
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
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    H: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    I: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
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
        self.sens_sparsity = Some(
            MatrixSparsity::try_from_indices(self.nstates, nparams, non_zeros.clone())
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
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    H: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    I: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
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
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    H: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    I: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
{
    fn calculate_sparsity(&mut self, y0: &Self::V, t0: Self::T, p: &Self::V) {
        self.calculate_jacobian_sparsity(y0, t0, p);
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
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    H: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    I: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
{
    fn call_inplace(&self, x: &M::V, t: M::T, y: &mut M::V) {
        self.op.statistics.borrow_mut().increment_call();
        (self.op.func)(x, self.p, t, y)
    }
}

impl<M, F, G, H, I> NonLinearOpJacobian for ParameterisedOp<'_, ClosureWithAdjoint<M, F, G, H, I>>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    H: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    I: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
{
    fn jac_mul_inplace(&self, x: &M::V, t: M::T, v: &M::V, y: &mut M::V) {
        self.op.statistics.borrow_mut().increment_jac_mul();
        (self.op.jacobian_action)(x, self.p, t, v, y)
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
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    H: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    I: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
{
    fn jac_transpose_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        self.op.statistics.borrow_mut().increment_jac_adj_mul();
        (self.op.jacobian_adjoint_action)(x, self.p, t, v, y);
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
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    H: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    I: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
{
    fn sens_transpose_mul_inplace(&self, _x: &Self::V, _t: Self::T, _v: &Self::V, y: &mut Self::V) {
        (self.op.sens_adjoint_action)(_x, self.p, _t, _v, y);
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
