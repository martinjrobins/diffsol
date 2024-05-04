use std::{cell::RefCell, rc::Rc};

use crate::{
    jacobian::{find_non_zeros_linear, JacobianColoring},
    matrix::{MatrixCommon, MatrixSparsity},
    Matrix, Vector,
};

use super::{LinearOp, Op, OpStatistics};

pub struct LinearClosure<M, F>
where
    M: Matrix,
    F: Fn(
        &<M as MatrixCommon>::V,
        &<M as MatrixCommon>::V,
        <M as MatrixCommon>::T,
        <M as MatrixCommon>::T,
        &mut <M as MatrixCommon>::V,
    ),
{
    func: F,
    nstates: usize,
    nout: usize,
    nparams: usize,
    p: Rc<M::V>,
    coloring: Option<JacobianColoring<M>>,
    sparsity: Option<M::Sparsity>,
    statistics: RefCell<OpStatistics>,
}

impl<M, F> LinearClosure<M, F>
where
    M: Matrix,
    F: Fn(
        &<M as MatrixCommon>::V,
        &<M as MatrixCommon>::V,
        <M as MatrixCommon>::T,
        <M as MatrixCommon>::T,
        &mut <M as MatrixCommon>::V,
    ),
{
    pub fn new(func: F, nstates: usize, nout: usize, p: Rc<M::V>) -> Self {
        let nparams = p.len();
        Self {
            func,
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

    pub fn set_params(&mut self, p: Rc<M::V>) {
        assert_eq!(p.len(), self.nparams);
        self.p = p;
    }
}

impl<M, F> Op for LinearClosure<M, F>
where
    M: Matrix,
    F: Fn(
        &<M as MatrixCommon>::V,
        &<M as MatrixCommon>::V,
        <M as MatrixCommon>::T,
        <M as MatrixCommon>::T,
        &mut <M as MatrixCommon>::V,
    ),
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
    fn sparsity(&self) -> Option<&<Self::M as Matrix>::Sparsity> {
        self.sparsity.as_ref()
    }
    fn statistics(&self) -> OpStatistics {
        self.statistics.borrow().clone()
    }
}

impl<M, F> LinearOp for LinearClosure<M, F>
where
    M: Matrix,
    F: Fn(
        &<M as MatrixCommon>::V,
        &<M as MatrixCommon>::V,
        <M as MatrixCommon>::T,
        <M as MatrixCommon>::T,
        &mut <M as MatrixCommon>::V,
    ),
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
}
