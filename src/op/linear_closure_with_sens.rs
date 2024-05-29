use std::{cell::RefCell, rc::Rc};

use crate::{
    jacobian::{find_non_zeros_linear, JacobianColoring}, matrix::sparsity::MatrixSparsity, Matrix, Vector
};

use super::{LinearOp, Op, OpStatistics};

pub struct LinearClosureWithSens<M, F, H>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
    H: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
{
    func: F,
    func_sens: H,
    nstates: usize,
    nout: usize,
    nparams: usize,
    p: Rc<M::V>,
    coloring: Option<JacobianColoring<M>>,
    sparsity: Option<M::Sparsity>,
    statistics: RefCell<OpStatistics>,
}

impl<M, F, H> LinearClosureWithSens<M, F, H>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
    H: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
{
    pub fn new(func: F, func_sens: H, nstates: usize, nout: usize, p: Rc<M::V>) -> Self {
        let nparams = p.len();
        Self {
            func,
            func_sens,
            nstates,
            statistics: RefCell::new(OpStatistics::default()),
            nout,
            nparams,
            p,
            coloring: None,
            sparsity: None,
        }
    }

    pub fn calculate_sparsity(&mut self, t0: M::T) {
        let non_zeros = find_non_zeros_linear(self, t0);
        self.sparsity = Some(
            MatrixSparsity::try_from_indices(self.nout(), self.nstates(), non_zeros.clone())
                .expect("invalid sparsity pattern"),
        );
        self.coloring = Some(JacobianColoring::new_from_non_zeros(self, non_zeros));
    }
}

impl<M, F, H> Op for LinearClosureWithSens<M, F, H>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
    H: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
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
    fn sparsity(&self) -> Option<<Self::M as Matrix>::SparsityRef<'_>> {
        self.sparsity.as_ref().map(|s| s.as_ref())
    }
    fn statistics(&self) -> OpStatistics {
        self.statistics.borrow().clone()
    }
}

impl<M, F, H> LinearOp for LinearClosureWithSens<M, F, H>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
    H: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
{
    fn gemv_inplace(&self, x: &M::V, t: M::T, beta: M::T, y: &mut M::V) {
        self.statistics.borrow_mut().increment_call();
        (self.func)(x, self.p.as_ref(), t, beta, y)
    }
    fn matrix_inplace(&self, t: Self::T, y: &mut Self::M) {
        self.statistics.borrow_mut().increment_matrix();
        if let Some(coloring) = &self.coloring {
            coloring.matrix_inplace(self, t, y);
        } else {
            self._default_matrix_inplace(t, y);
        }
    }
    fn sens_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        (self.func_sens)(self.p.as_ref(), x, t, v, y)
    }
    fn has_sens(&self) -> bool {
        true
    }
}
