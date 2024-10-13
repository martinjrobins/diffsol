use std::{cell::RefCell, rc::Rc};

use crate::{
    find_matrix_non_zeros, find_transpose_non_zeros, jacobian::JacobianColoring, matrix::sparsity::MatrixSparsity, Matrix,
    Vector, LinearOp, LinearOpMatrix, LinearOpTranspose, Op,
};

use super::OpStatistics;

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
    p: Rc<M::V>,
    coloring: Option<JacobianColoring<M>>,
    sparsity: Option<M::Sparsity>,
    coloring_adjoint: Option<JacobianColoring<M>>,
    sparsity_adjoint: Option<M::Sparsity>,
    statistics: RefCell<OpStatistics>,
}

impl<M, F, G> LinearClosureWithAdjoint<M, F, G>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
{
    pub fn new(func: F, func_adjoint: G, nstates: usize, nout: usize, p: Rc<M::V>) -> Self {
        let nparams = p.len();
        Self {
            func,
            func_adjoint,
            nstates,
            statistics: RefCell::new(OpStatistics::default()),
            nout,
            nparams,
            p,
            coloring: None,
            sparsity: None,
            coloring_adjoint: None,
            sparsity_adjoint: None,
        }
    }

    pub fn calculate_sparsity(&mut self, t0: M::T) {
        let non_zeros = find_matrix_non_zeros(self, t0);
        self.sparsity = Some(
            MatrixSparsity::try_from_indices(self.nout(), self.nstates(), non_zeros.clone())
                .expect("invalid sparsity pattern"),
        );
        self.coloring = Some(JacobianColoring::new_from_non_zeros(self, non_zeros));
    }
    pub fn calculate_adjoint_sparsity(&mut self, t0: M::T) {
        let non_zeros = find_transpose_non_zeros(self, t0);
        self.sparsity_adjoint = Some(
            MatrixSparsity::try_from_indices(self.nstates, self.nout, non_zeros.clone())
                .expect("invalid sparsity pattern"),
        );
        self.coloring_adjoint = Some(JacobianColoring::new_from_non_zeros(self, non_zeros));
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
    fn statistics(&self) -> OpStatistics {
        self.statistics.borrow().clone()
    }
}

impl<M, F, G> LinearOp for LinearClosureWithAdjoint<M, F, G>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
{
    fn gemv_inplace(&self, x: &M::V, t: M::T, beta: M::T, y: &mut M::V) {
        self.statistics.borrow_mut().increment_call();
        (self.func)(x, self.p.as_ref(), t, beta, y)
    }
    fn gemv_transpose_inplace(&self, x: &Self::V, t: Self::T, beta: Self::T, y: &mut Self::V) {
        (self.func_adjoint)(x, self.p.as_ref(), t, beta, y)
    }
}

impl<M, F, G> LinearOpMatrix for LinearClosureWithAdjoint<M, F, G>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
{
    fn matrix_inplace(&self, t: Self::T, y: &mut Self::M) {
        self.statistics.borrow_mut().increment_matrix();
        if let Some(coloring) = &self.coloring {
            coloring.matrix_inplace(self, t, y);
        } else {
            self._default_matrix_inplace(t, y);
        }
    }
}

impl<M, F, G> LinearOpTranspose for LinearClosureWithAdjoint<M, F, G>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, M::T, &mut M::V),
{
    fn transpose_inplace(&self, t: Self::T, y: &mut Self::M) {
        if let Some(coloring) = &self.coloring_adjoint {
            coloring.matrix_inplace(self, t, y);
        } else {
            self._default_transpose_inplace(t, y);
        }
    }
}
