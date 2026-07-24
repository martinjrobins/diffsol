use std::cell::RefCell;

use crate::{
    find_matrix_non_zeros, jacobian::JacobianColoring, matrix::sparsity::MatrixSparsity, LinearOp,
    Matrix, Op,
};

use super::{BuilderOp, OpStatistics, ParameterisedOp};

pub struct LinearClosure<M, F>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
{
    func: F,
    nstates: usize,
    nout: usize,
    nparams: usize,
    coloring: Option<JacobianColoring<M>>,
    sparsity: Option<M::Sparsity>,
    statistics: RefCell<OpStatistics>,
    ctx: M::C,
}

impl<M, F> LinearClosure<M, F>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
{
    pub fn new(func: F, nstates: usize, nout: usize, nparams: usize, ctx: M::C) -> Self {
        Self {
            func,
            nstates,
            statistics: RefCell::new(OpStatistics::default()),
            nout,
            nparams,
            coloring: None,
            sparsity: None,
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
}

impl<M, F> Op for LinearClosure<M, F>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
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

impl<M, F> BuilderOp for LinearClosure<M, F>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
{
    fn calculate_sparsity(&mut self, _y0: &Self::V, t0: Self::T, p: &Self::V) {
        self.calculate_sparsity(t0, p);
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

impl<M, F> LinearOp for ParameterisedOp<'_, LinearClosure<M, F>>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
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
