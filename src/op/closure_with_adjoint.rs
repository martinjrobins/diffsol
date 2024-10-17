use std::{cell::RefCell, rc::Rc};

use crate::{
    jacobian::{
        find_adjoint_non_zeros, find_jacobian_non_zeros, find_sens_adjoint_non_zeros,
        JacobianColoring,
    },
    Matrix, MatrixSparsity, NonLinearOp, NonLinearOpAdjoint, NonLinearOpJacobian,
    NonLinearOpSensAdjoint, Op, Vector,
};

use super::OpStatistics;

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
    p: Rc<M::V>,
    coloring: Option<JacobianColoring<M>>,
    sparsity: Option<M::Sparsity>,
    sparsity_adjoint: Option<M::Sparsity>,
    coloring_adjoint: Option<JacobianColoring<M>>,
    sens_sparsity: Option<M::Sparsity>,
    coloring_sens_adjoint: Option<JacobianColoring<M>>,
    statistics: RefCell<OpStatistics>,
}

impl<M, F, G, H, I> ClosureWithAdjoint<M, F, G, H, I>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    H: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    I: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
{
    pub fn new(
        func: F,
        jacobian_action: G,
        jacobian_adjoint_action: H,
        sens_adjoint_action: I,
        nstates: usize,
        nout: usize,
        p: Rc<M::V>,
    ) -> Self {
        let nparams = p.len();
        Self {
            func,
            jacobian_action,
            jacobian_adjoint_action,
            sens_adjoint_action,
            nstates,
            nout,
            nparams,
            p,
            statistics: RefCell::new(OpStatistics::default()),
            coloring: None,
            sparsity: None,
            sparsity_adjoint: None,
            coloring_adjoint: None,
            sens_sparsity: None,
            coloring_sens_adjoint: None,
        }
    }

    pub fn calculate_jacobian_sparsity(&mut self, y0: &M::V, t0: M::T) {
        let non_zeros = find_jacobian_non_zeros(self, y0, t0);
        self.sparsity = Some(
            MatrixSparsity::try_from_indices(self.nout(), self.nstates(), non_zeros.clone())
                .expect("invalid sparsity pattern"),
        );
        self.coloring = Some(JacobianColoring::new_from_non_zeros(self, non_zeros));
    }

    pub fn calculate_adjoint_sparsity(&mut self, y0: &M::V, t0: M::T) {
        let non_zeros = find_adjoint_non_zeros(self, y0, t0);
        self.sparsity_adjoint = Some(
            MatrixSparsity::try_from_indices(self.nstates, self.nout, non_zeros.clone())
                .expect("invalid sparsity pattern"),
        );
        self.coloring_adjoint = Some(JacobianColoring::new_from_non_zeros(self, non_zeros));
    }

    pub fn calculate_sens_adjoint_sparsity(&mut self, y0: &M::V, t0: M::T) {
        let non_zeros = find_sens_adjoint_non_zeros(self, y0, t0);
        self.sens_sparsity = Some(
            MatrixSparsity::try_from_indices(self.nstates, self.nparams, non_zeros.clone())
                .expect("invalid sparsity pattern"),
        );
        self.coloring_sens_adjoint = Some(JacobianColoring::new_from_non_zeros(self, non_zeros));
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
    fn sparsity_adjoint(&self) -> Option<<Self::M as Matrix>::SparsityRef<'_>> {
        self.sparsity_adjoint.as_ref().map(|s| s.as_ref())
    }
    fn sparsity_sens_adjoint(&self) -> Option<<Self::M as Matrix>::SparsityRef<'_>> {
        self.sens_sparsity.as_ref().map(|s| s.as_ref())
    }
    fn statistics(&self) -> OpStatistics {
        self.statistics.borrow().clone()
    }
}

impl<M, F, G, H, I> NonLinearOp for ClosureWithAdjoint<M, F, G, H, I>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    H: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    I: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
{
    fn call_inplace(&self, x: &M::V, t: M::T, y: &mut M::V) {
        self.statistics.borrow_mut().increment_call();
        (self.func)(x, self.p.as_ref(), t, y)
    }
}

impl<M, F, G, H, I> NonLinearOpJacobian for ClosureWithAdjoint<M, F, G, H, I>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    H: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    I: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
{
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

impl<M, F, G, H, I> NonLinearOpAdjoint for ClosureWithAdjoint<M, F, G, H, I>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    H: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    I: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
{
    fn jac_transpose_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        self.statistics.borrow_mut().increment_jac_adj_mul();
        (self.jacobian_adjoint_action)(x, self.p.as_ref(), t, v, y);
    }

    fn adjoint_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        if let Some(coloring) = self.coloring_adjoint.as_ref() {
            coloring.adjoint_inplace(self, x, t, y);
        } else {
            self._default_adjoint_inplace(x, t, y);
        }
    }
}

impl<M, F, G, H, I> NonLinearOpSensAdjoint for ClosureWithAdjoint<M, F, G, H, I>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    H: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
    I: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
{
    fn sens_transpose_mul_inplace(&self, _x: &Self::V, _t: Self::T, _v: &Self::V, y: &mut Self::V) {
        (self.sens_adjoint_action)(_x, self.p.as_ref(), _t, _v, y);
    }
    fn sens_adjoint_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        if let Some(coloring) = self.coloring_sens_adjoint.as_ref() {
            coloring.sens_adjoint_inplace(self, x, t, y);
        } else {
            self._default_sens_adjoint_inplace(x, t, y);
        }
    }
}
