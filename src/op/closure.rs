use std::{cell::RefCell, rc::Rc};

use crate::{
    jacobian::{find_non_zeros_nonlinear, JacobianColoring},
    matrix::MatrixSparsity,
    Matrix, Vector,
};

use super::{NonLinearOp, Op, OpStatistics, VView, VViewMut};

pub struct Closure<M, F, G>
where
    M: Matrix,
    F: Fn(VView<'_, M>, &M::V, M::T, VViewMut<'_, M>),
    G: Fn(VView<'_, M>, &M::V, M::T, VView<'_, M>, VViewMut<'_, M>),
{
    func: F,
    jacobian_action: G,
    nstates: usize,
    nout: usize,
    nparams: usize,
    p: Rc<M::V>,
    coloring: Option<JacobianColoring<M>>,
    sparsity: Option<M::Sparsity>,
    statistics: RefCell<OpStatistics>,
}

impl<M, F, G> Closure<M, F, G>
where
    M: Matrix,
    F: Fn(VView<'_, M>, &M::V, M::T, VViewMut<'_, M>),
    G: Fn(VView<'_, M>, &M::V, M::T, VView<'_, M>, VViewMut<'_, M>),
{
    pub fn new(func: F, jacobian_action: G, nstates: usize, nout: usize, p: Rc<M::V>) -> Self {
        let nparams = p.len();
        Self {
            func,
            jacobian_action,
            nstates,
            nout,
            nparams,
            p,
            statistics: RefCell::new(OpStatistics::default()),
            coloring: None,
            sparsity: None,
        }
    }

    pub fn calculate_sparsity(&mut self, y0: &M::V, t0: M::T) {
        let non_zeros = find_non_zeros_nonlinear(self, y0, t0);
        self.sparsity = Some(
            MatrixSparsity::try_from_indices(self.nout(), self.nstates(), non_zeros.clone())
                .expect("invalid sparsity pattern"),
        );
        self.coloring = Some(JacobianColoring::new_from_non_zeros(self, non_zeros));
    }
}

impl<M, F, G> Op for Closure<M, F, G>
where
    M: Matrix,
    F: Fn(<M::V as Vector>::View<'_>, &M::V, M::T, <M::V as Vector>::ViewMut<'_>),
    G: Fn(<M::V as Vector>::View<'_>, &M::V, M::T, <M::V as Vector>::View<'_>, <M::V as Vector>::ViewMut<'_>),
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
    fn sparsity(&self) -> Option<&<Self::M as Matrix>::Sparsity> {
        self.sparsity.as_ref()
    }
    fn statistics(&self) -> OpStatistics {
        self.statistics.borrow().clone()
    }
}

impl<M, F, G> NonLinearOp for Closure<M, F, G>
where
    M: Matrix,
    F: Fn(VView<'_, M>, &M::V, M::T, VViewMut<'_, M>),
    G: Fn(VView<'_, M>, &M::V, M::T, VView<'_, M>, VViewMut<'_, M>),
{
    fn call_inplace(&self, x: <M::V as Vector>::View<'_>, t: M::T, y: <M::V as Vector>::ViewMut<'_>) {
        self.statistics.borrow_mut().increment_call();
        (self.func)(x, self.p.as_ref(), t, y)
    }
    fn jac_mul_inplace(&self, x: &M::V, t: M::T, v: &M::V, y: &mut M::V) {
        self.statistics.borrow_mut().increment_jac_mul();
        (self.jacobian_action)(x, self.p.as_ref(), t, v, y)
    }
    fn jacobian_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        self.statistics.borrow_mut().increment_matrix();
        if let Some(coloring) = self.coloring.as_ref() {
            coloring.jacobian_inplace(self, x, t, y);
        } else {
            self._default_jacobian_inplace(x, t, y);
        }
    }
}
