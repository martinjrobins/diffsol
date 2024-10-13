use std::{cell::RefCell, rc::Rc};

use crate::{Matrix, Vector, NonLinearOp, Op};

use super::OpStatistics;

pub struct ClosureNoJac<M, F>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
{
    func: F,
    nstates: usize,
    nout: usize,
    nparams: usize,
    p: Rc<M::V>,
    statistics: RefCell<OpStatistics>,
}

impl<M, F> ClosureNoJac<M, F>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
{
    pub fn new(func: F, nstates: usize, nout: usize, p: Rc<M::V>) -> Self {
        let nparams = p.len();
        Self {
            func,
            nstates,
            nout,
            nparams,
            p,
            statistics: RefCell::new(OpStatistics::default()),
        }
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
    fn nstates(&self) -> usize {
        self.nstates
    }
    fn nout(&self) -> usize {
        self.nout
    }
    fn nparams(&self) -> usize {
        self.nparams
    }
    fn set_params(&mut self, p: Rc<M::V>) {
        assert_eq!(p.len(), self.nparams);
        self.p = p;
    }
    fn statistics(&self) -> OpStatistics {
        self.statistics.borrow().clone()
    }
}

impl<M, F> NonLinearOp for ClosureNoJac<M, F>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
{
    fn call_inplace(&self, x: &M::V, t: M::T, y: &mut M::V) {
        self.statistics.borrow_mut().increment_call();
        (self.func)(x, self.p.as_ref(), t, y)
    }
}
