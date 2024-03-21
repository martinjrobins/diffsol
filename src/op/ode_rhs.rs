use std::rc::Rc;

use crate::OdeEquations;

use super::{NonLinearOp, Op};

pub struct OdeRhs<Eqn: OdeEquations> {
    eqn: Rc<Eqn>,
}

impl<Eqn: OdeEquations> OdeRhs<Eqn> {
    pub fn new(eqn: Rc<Eqn>) -> Self {
        Self { eqn }
    }
}

impl<Eqn: OdeEquations> Op for OdeRhs<Eqn> {
    type V = Eqn::V;
    type T = Eqn::T;
    type M = Eqn::M;
    fn nstates(&self) -> usize {
        self.eqn.nstates()
    }
    fn nout(&self) -> usize {
        self.eqn.nstates()
    }
    fn nparams(&self) -> usize {
        self.eqn.nparams()
    }
}

impl<Eqn: OdeEquations> NonLinearOp for OdeRhs<Eqn> {
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        self.eqn.rhs_inplace(t, x, y);
    }
    fn jac_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        self.eqn.rhs_jac_inplace(t, x, v, y);
    }
}
