use std::cell::RefCell;

use crate::{
    jacobian::{find_jacobian_non_zeros, find_sens_non_zeros, JacobianColoring},
    Matrix, MatrixSparsity, NonLinearOp, NonLinearOpJacobian, NonLinearOpSens, Op, Vector,
};

use super::{BuilderOp, OpStatistics, ParameterisedOp};

pub struct ClosureWithSens<M, F, G, H>
where
    M: Matrix,
{
    func: F,
    jacobian_action: G,
    sens_action: H,
    nstates: usize,
    nparams: usize,
    nout: usize,
    coloring: Option<JacobianColoring<M>>,
    sens_coloring: Option<JacobianColoring<M>>,
    sparsity: Option<M::Sparsity>,
    sens_sparsity: Option<M::Sparsity>,
    statistics: RefCell<OpStatistics>,
    ctx: M::C,
}

impl<M, F, G, H> ClosureWithSens<M, F, G, H>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    H: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
{
    pub fn new(
        func: F,
        jacobian_action: G,
        sens_action: H,
        nstates: usize,
        nparams: usize,
        nout: usize,
        ctx: M::C,
    ) -> Self {
        Self {
            func,
            jacobian_action,
            sens_action,
            nstates,
            nout,
            nparams,
            statistics: RefCell::new(OpStatistics::default()),
            coloring: None,
            sparsity: None,
            sens_coloring: None,
            sens_sparsity: None,
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
    pub fn calculate_sens_sparsity(&mut self, y0: &M::V, t0: M::T, p: &M::V) {
        let op = ParameterisedOp { op: self, p };
        let non_zeros = find_sens_non_zeros(&op, y0, t0);
        let nparams = p.len();
        self.sens_sparsity = Some(
            MatrixSparsity::try_from_indices(self.nout(), nparams, non_zeros.clone())
                .expect("invalid sparsity pattern"),
        );
        self.sens_coloring = Some(JacobianColoring::new(
            self.sens_sparsity.as_ref().unwrap(),
            &non_zeros,
            self.ctx.clone(),
        ));
    }
}

impl<M, F, G, H> BuilderOp for ClosureWithSens<M, F, G, H>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    H: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
{
    fn set_nstates(&mut self, nstates: usize) {
        self.nstates = nstates;
    }
    fn set_nout(&mut self, nout: usize) {
        self.nout = nout;
    }
    fn set_nparams(&mut self, nparams: usize) {
        self.nparams = nparams;
    }

    fn calculate_sparsity(&mut self, y0: &Self::V, t0: Self::T, p: &Self::V) {
        self.calculate_jacobian_sparsity(y0, t0, p);
        self.calculate_sens_sparsity(y0, t0, p);
    }
}

impl<M, F, G, H> Op for ClosureWithSens<M, F, G, H>
where
    M: Matrix,
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

impl<M, F, G, H> NonLinearOp for ParameterisedOp<'_, ClosureWithSens<M, F, G, H>>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    H: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
{
    fn call_inplace(&self, x: &M::V, t: M::T, y: &mut M::V) {
        self.op.statistics.borrow_mut().increment_call();
        (self.op.func)(x, self.p, t, y)
    }
}

impl<M, F, G, H> NonLinearOpJacobian for ParameterisedOp<'_, ClosureWithSens<M, F, G, H>>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    H: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
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

impl<M, F, G, H> NonLinearOpSens for ParameterisedOp<'_, ClosureWithSens<M, F, G, H>>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    H: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
{
    fn sens_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        (self.op.sens_action)(x, self.p, t, v, y);
    }

    fn sens_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        if let Some(coloring) = self.op.sens_coloring.as_ref() {
            coloring.jacobian_inplace(self, x, t, y);
        } else {
            self._default_sens_inplace(x, t, y);
        }
    }
    fn sens_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.op.sens_sparsity.clone()
    }
}
