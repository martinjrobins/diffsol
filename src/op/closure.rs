use std::rc::Rc;

use crate::{jacobian::{find_non_zeros_nonlinear, JacobianColoring}, matrix::MatrixSparsity, Matrix, Vector};

use super::{NonLinearOp, Op};

pub struct Closure<M, F, G>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
{
    func: F,
    jacobian_action: G,
    nstates: usize,
    nout: usize,
    nparams: usize,
    p: Rc<M::V>,
    coloring: Option<JacobianColoring<M>>,
    sparsity: Option<M::Sparsity>,
}

impl<M, F, G> Closure<M, F, G>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
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
            coloring: None,
            sparsity: None,
        }
    }

    pub fn calculate_sparsity(&mut self, y0: &M::V, t0: M::T) {
        let non_zeros = find_non_zeros_nonlinear(self, y0, t0);
        self.sparsity = Some(MatrixSparsity::try_from_indices(self.nout(), self.nstates(), non_zeros).expect("invalid sparsity pattern"));
        self.coloring = Some(JacobianColoring::new(self));
    }
}

impl<M, F, G> Op for Closure<M, F, G>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
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

impl<M, F, G> NonLinearOp for Closure<M, F, G>
where
    M: Matrix,
    F: Fn(&M::V, &M::V, M::T, &mut M::V),
    G: Fn(&M::V, &M::V, M::T, &M::V, &mut M::V),
{
    fn call_inplace(&self, x: &M::V, t: M::T, y: &mut M::V) {
        (self.func)(x, self.p.as_ref(), t, y)
    }
    fn jac_mul_inplace(&self, x: &M::V, t: M::T, v: &M::V, y: &mut M::V) {
        (self.jacobian_action)(x, self.p.as_ref(), t, v, y)
    }
}
