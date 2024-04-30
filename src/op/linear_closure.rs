use std::rc::Rc;

use crate::{jacobian::{find_non_zeros_linear, JacobianColoring}, matrix::{MatrixCommon, MatrixSparsity}, Matrix, Vector};

use super::{LinearOp, Op};

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
            nout,
            nparams,
            p,
            coloring: None,
            sparsity: None,
        }
    }

    pub fn calculate_sparsity(&mut self, t0: M::T) {
        let non_zeros = find_non_zeros_linear(self, t0);
        self.sparsity = Some(MatrixSparsity::try_from_indices(self.nout(), self.nstates(), non_zeros).expect("invalid sparsity pattern"));
        self.coloring = Some(JacobianColoring::new(self));
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
        (self.func)(x, self.p.as_ref(), t, beta, y)
    }
}
