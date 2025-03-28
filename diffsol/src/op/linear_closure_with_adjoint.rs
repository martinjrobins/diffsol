use std::cell::RefCell;

use crate::{
    find_matrix_non_zeros, find_transpose_non_zeros, jacobian::JacobianColoring,
    matrix::sparsity::MatrixSparsity, LinearOp, LinearOpTranspose, Matrix, Op,
};

use super::{BuilderOp, OpStatistics, ParameterisedOp};

pub struct LinearClosureWithAdjoint<M, F, G>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
{
    func: F,
    func_adjoint: G,
    nstates: usize,
    nout: usize,
    nparams: usize,
    coloring: Option<JacobianColoring<M>>,
    sparsity: Option<M::Sparsity>,
    coloring_adjoint: Option<JacobianColoring<M>>,
    sparsity_adjoint: Option<M::Sparsity>,
    statistics: RefCell<OpStatistics>,
    ctx: M::C,
}

impl<M, F, G> LinearClosureWithAdjoint<M, F, G>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
{
    pub fn new(
        func: F,
        func_adjoint: G,
        nstates: usize,
        nout: usize,
        nparams: usize,
        ctx: M::C,
    ) -> Self {
        Self {
            func,
            func_adjoint,
            nstates,
            statistics: RefCell::new(OpStatistics::default()),
            nout,
            nparams,
            coloring: None,
            sparsity: None,
            coloring_adjoint: None,
            sparsity_adjoint: None,
            ctx,
        }
    }

    pub fn calculate_sparsity(&mut self, t0: M::T, p: &M::V) {
        let op = ParameterisedOp { op: self, p };
        let non_zeros = find_matrix_non_zeros(&op, t0);
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
    pub fn calculate_adjoint_sparsity(&mut self, t0: M::T, p: &M::V) {
        let op = ParameterisedOp { op: self, p };
        let non_zeros = find_transpose_non_zeros(&op, t0);
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
}

impl<M, F, G> Op for LinearClosureWithAdjoint<M, F, G>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
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
    fn context(&self) -> &Self::C {
        &self.ctx
    }

    fn statistics(&self) -> OpStatistics {
        self.statistics.borrow().clone()
    }
}

impl<M, F, G> BuilderOp for LinearClosureWithAdjoint<M, F, G>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
{
    fn calculate_sparsity(&mut self, _y0: &Self::V, t0: Self::T, p: &Self::V) {
        self.calculate_sparsity(t0, p);
        self.calculate_adjoint_sparsity(t0, p);
    }
    fn set_nout(&mut self, nout: usize) {
        self.nout = nout;
    }
    fn set_nparams(&mut self, nparams: usize) {
        self.nparams = nparams;
    }
    fn set_nstates(&mut self, nstates: usize) {
        self.nstates = nstates;
    }
}

impl<M, F, G> LinearOp for ParameterisedOp<'_, LinearClosureWithAdjoint<M, F, G>>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
{
    fn gemv_inplace(&self, x: &M::V, t: M::T, beta: M::T, y: &mut M::V) {
        self.op.statistics.borrow_mut().increment_call();
        (self.op.func)(x, self.p, t, beta, y)
    }

    fn matrix_inplace(&self, t: Self::T, y: &mut Self::M) {
        self.op.statistics.borrow_mut().increment_matrix();
        if let Some(coloring) = &self.op.coloring {
            coloring.matrix_inplace(self, t, y);
        } else {
            self._default_matrix_inplace(t, y);
        }
    }
    fn sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.op.sparsity.clone()
    }
}

impl<M, F, G> LinearOpTranspose for ParameterisedOp<'_, LinearClosureWithAdjoint<M, F, G>>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
{
    fn gemv_transpose_inplace(&self, x: &Self::V, t: Self::T, beta: Self::T, y: &mut Self::V) {
        (self.op.func_adjoint)(x, self.p, t, beta, y)
    }
    fn transpose_inplace(&self, t: Self::T, y: &mut Self::M) {
        if let Some(coloring) = &self.op.coloring_adjoint {
            coloring.matrix_inplace(self, t, y);
        } else {
            self._default_transpose_inplace(t, y);
        }
    }

    fn transpose_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.op.sparsity_adjoint.clone()
    }
}
