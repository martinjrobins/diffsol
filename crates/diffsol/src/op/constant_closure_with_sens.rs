use crate::{
    jacobian::{find_constant_sens_non_zeros, JacobianColoring},
    matrix::sparsity::MatrixSparsity,
    ConstantOp, ConstantOpSens, Matrix, Op, Vector,
};

use super::{BuilderOp, ParameterisedOp};

pub struct ConstantClosureWithSens<M, I, J>
where
    M: Matrix,
    I: Fn(&M::V, M::T, &mut M::V),
    J: Fn(&M::V, M::T, &M::V, &mut M::V),
{
    func: I,
    func_sens: J,
    nout: usize,
    nparams: usize,
    sens_sparsity: Option<M::Sparsity>,
    sens_coloring: Option<JacobianColoring<M>>,
    ctx: M::C,
}

impl<M, I, J> ConstantClosureWithSens<M, I, J>
where
    M: Matrix,
    I: Fn(&M::V, M::T, &mut M::V),
    J: Fn(&M::V, M::T, &M::V, &mut M::V),
{
    pub fn new(func: I, func_sens: J, nout: usize, nparams: usize, ctx: M::C) -> Self {
        Self {
            func,
            func_sens,
            nout,
            nparams,
            sens_sparsity: None,
            sens_coloring: None,
            ctx,
        }
    }

    /// Find the sparsity of the parameter Jacobian dy0/dp, and a coloring to compute it with.
    pub fn calculate_sens_sparsity(&mut self, t0: M::T, p: &M::V) {
        let op = ParameterisedOp { op: self, p };
        let non_zeros = find_constant_sens_non_zeros(&op, t0);
        self.sens_sparsity = Some(
            MatrixSparsity::try_from_indices(self.nout(), p.len(), non_zeros.clone())
                .expect("invalid sparsity pattern"),
        );
        self.sens_coloring = Some(JacobianColoring::new(
            self.sens_sparsity.as_ref().unwrap(),
            &non_zeros,
            self.ctx.clone(),
        ));
    }
}

impl<M, I, J> Op for ConstantClosureWithSens<M, I, J>
where
    M: Matrix,
    I: Fn(&M::V, M::T, &mut M::V),
    J: Fn(&M::V, M::T, &M::V, &mut M::V),
{
    type V = M::V;
    type T = M::T;
    type M = M;
    type C = M::C;
    fn nstates(&self) -> usize {
        0
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
}

impl<M, I, J> BuilderOp for ConstantClosureWithSens<M, I, J>
where
    M: Matrix,
    I: Fn(&M::V, M::T, &mut M::V),
    J: Fn(&M::V, M::T, &M::V, &mut M::V),
{
    fn calculate_sparsity(&mut self, _y0: &Self::V, _t0: Self::T, _p: &Self::V) {
        // do nothing
    }
    fn calculate_augmented_sparsity(&mut self, _y0: &Self::V, t0: Self::T, p: &Self::V) {
        ConstantClosureWithSens::calculate_sens_sparsity(self, t0, p);
    }
    fn set_nstates(&mut self, _nstates: usize) {
        // do nothing
    }
    fn set_nout(&mut self, nout: usize) {
        self.nout = nout;
    }
    fn set_nparams(&mut self, nparams: usize) {
        self.nparams = nparams;
    }
}

impl<M, I, J> ConstantOp for ParameterisedOp<'_, ConstantClosureWithSens<M, I, J>>
where
    M: Matrix,
    I: Fn(&M::V, M::T, &mut M::V),
    J: Fn(&M::V, M::T, &M::V, &mut M::V),
{
    fn call_inplace(&self, t: Self::T, y: &mut Self::V) {
        (self.op.func)(self.p, t, y)
    }
}

impl<M, I, J> ConstantOpSens for ParameterisedOp<'_, ConstantClosureWithSens<M, I, J>>
where
    M: Matrix,
    I: Fn(&M::V, M::T, &mut M::V),
    J: Fn(&M::V, M::T, &M::V, &mut M::V),
{
    fn sens_mul_inplace(&self, t: Self::T, v: &Self::V, y: &mut Self::V) {
        (self.op.func_sens)(self.p, t, v, y);
    }

    fn sens_inplace(&self, t: Self::T, y: &mut Self::M) {
        if let Some(coloring) = self.op.sens_coloring.as_ref() {
            coloring.constant_sens_inplace(self, t, y);
        } else {
            self._default_sens_inplace(t, y);
        }
    }

    fn sens_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.op.sens_sparsity.clone()
    }
}
