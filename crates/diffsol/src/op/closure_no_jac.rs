use std::cell::RefCell;

use crate::{Matrix, NonLinearOp, Op};

use super::{BuilderOp, OpStatistics, ParameterisedOp};

pub struct ClosureNoJac<M, F>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
{
    func: F,
    nstates: usize,
    nout: usize,
    nparams: usize,
    statistics: RefCell<OpStatistics>,
    ctx: M::C,
}

impl<M, F> ClosureNoJac<M, F>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
{
    pub fn new(func: F, nstates: usize, nout: usize, nparams: usize, ctx: M::C) -> Self {
        Self {
            func,
            nstates,
            nparams,
            nout,
            statistics: RefCell::new(OpStatistics::default()),
            ctx,
        }
    }
}

impl<M, F> BuilderOp for ClosureNoJac<M, F>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
{
    fn calculate_sparsity(&mut self, _y0: &Self::V, _t0: Self::T, _p: &Self::V) {
        // Do nothing
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

impl<M, F> Op for ClosureNoJac<M, F>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
{
    type V = M::V;
    type T = M::T;
    type M = M;
    type C = M::C;
    fn nstates(&self) -> usize {
        self.nstates
    }
    fn context(&self) -> &Self::C {
        &self.ctx
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

impl<M, F> NonLinearOp for ParameterisedOp<'_, ClosureNoJac<M, F>>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
{
    fn call_inplace(&self, x: &M::V, t: M::T, y: &mut M::V) {
        self.op.statistics.borrow_mut().increment_call();
        (self.op.func)(x, self.p, t, y)
    }
}
