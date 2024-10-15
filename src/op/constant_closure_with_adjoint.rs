use num_traits::Zero;
use std::rc::Rc;

use crate::{ConstantOp, ConstantOpSensAdjoint, Matrix, Op, Vector};

pub struct ConstantClosureWithAdjoint<M, I, J>
where
    M: Matrix,
    I: Fn(&M::V, M::T) -> M::V,
    J: Fn(&M::V, M::T, &M::V, &mut M::V),
{
    func: I,
    func_sens_adjoint: J,
    nstates: usize,
    nout: usize,
    nparams: usize,
    p: Rc<M::V>,
}

impl<M, I, J> ConstantClosureWithAdjoint<M, I, J>
where
    M: Matrix,
    I: Fn(&M::V, M::T) -> M::V,
    J: Fn(&M::V, M::T, &M::V, &mut M::V),
{
    pub fn new(func: I, func_sens_adjoint: J, p: Rc<M::V>) -> Self {
        let nparams = p.len();
        let y0 = (func)(p.as_ref(), M::T::zero());
        let nstates = y0.len();
        let nout = nstates;
        Self {
            func,
            func_sens_adjoint,
            nstates,
            nout,
            nparams,
            p,
        }
    }
}

impl<M, I, J> Op for ConstantClosureWithAdjoint<M, I, J>
where
    M: Matrix,
    I: Fn(&M::V, M::T) -> M::V,
    J: Fn(&M::V, M::T, &M::V, &mut M::V),
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
}

impl<M, I, J> ConstantOp for ConstantClosureWithAdjoint<M, I, J>
where
    M: Matrix,
    I: Fn(&M::V, M::T) -> M::V,
    J: Fn(&M::V, M::T, &M::V, &mut M::V),
{
    fn call_inplace(&self, t: Self::T, y: &mut Self::V) {
        y.copy_from(&(self.func)(self.p.as_ref(), t));
    }
    fn call(&self, t: Self::T) -> Self::V {
        (self.func)(self.p.as_ref(), t)
    }
}

impl<M, I, J> ConstantOpSensAdjoint for ConstantClosureWithAdjoint<M, I, J>
where
    M: Matrix,
    I: Fn(&M::V, M::T) -> M::V,
    J: Fn(&M::V, M::T, &M::V, &mut M::V),
{
    fn sens_mul_transpose_inplace(&self, t: Self::T, v: &Self::V, y: &mut Self::V) {
        (self.func_sens_adjoint)(self.p.as_ref(), t, v, y);
    }
}
