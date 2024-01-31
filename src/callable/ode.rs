use crate::{VectorRef, IndexType, Matrix, Vector};
use num_traits::{One, Zero};
use std::ops::SubAssign;

use super::{Callable, Jacobian};

// callable to solve for F(y) = M (y' + psi) - c * f(y) = 0 
pub struct BdfCallable<'a, M: Matrix, CRhs: Callable<M::V>, CMass: Callable<M::V>> 
{
    rhs: &'a CRhs,
    mass: &'a CMass,
    psi_neg_y0: M::V,
    c: M::T,
    rhs_jac: M,
    jac: M,
    mass_jac: M,
    rhs_jacobian_is_stale: bool,
    jacobian_is_stale: bool,
    mass_jacobian_is_stale: bool,
}

impl<'a, M: Matrix, CRhs: Callable<M::V>, CMass: Callable<M::V>> BdfCallable<'a, M, CRhs, CMass> 
{
    pub fn new(rhs: &'a CRhs, mass: &'a CMass) -> Self {
        let n = rhs.nstates();
        let c = M::T::zero();
        let psi_neg_y0 = <M::V as Vector>::zeros(n);
        Self { rhs, mass, psi_neg_y0, c, rhs_jacobian_is_stale: true, jacobian_is_stale: true, mass_jacobian_is_stale: true, rhs_jac: M::zeros(n, n), jac: M::zeros(n, n), mass_jac: M::zeros(n, n) }
    }
    pub fn set_c(&mut self, h: M::T, alpha: &[M::T], order: IndexType) {
        self.c = h * alpha[usize::from(order)];
        self.jacobian_is_stale = true;
    }
    pub fn set_psi_and_y0(&mut self, diff: &M, gamma: &[M::T], alpha: &[M::T], order: usize, y0: &M::V) {
        // update psi term as defined in second equation on page 9 of [1]
        self.psi_neg_y0 = diff.column(0) * gamma[0];
        for i in 1..=order {
            self.psi_neg_y0 += diff.column(i) * gamma[i]
        }
        self.psi_neg_y0 *= alpha[order];

        // now negate y0
        self.psi_neg_y0.sub_assign(y0);
    }
    pub fn set_rhs_jacobian_is_stale(&mut self) {
        self.rhs_jacobian_is_stale = true;
        self.jacobian_is_stale = true;
    }
}


// callable to solve for F(y) = M (y' + psi) - f(y) = 0 
impl<'a, M: Matrix, CRhs: Callable<M::V>, CMass: Callable<M::V>> Callable<M::V> for BdfCallable<'a, M, CRhs, CMass> 
where 
    for <'b> &'b M::V: VectorRef<M::V>,
{

    // F(y) = M (y - y0 + psi) - c * f(y) = 0
    fn call(&self, x: &M::V, p: &M::V, y: &mut M::V) {
        self.rhs.call(x, p, y);
        let tmp = x - &self.psi_neg_y0;
        self.mass.gemv(&tmp, p, M::T::one(), -self.c, y);
}
    fn nstates(&self) -> usize {
        self.rhs.nstates()
    }
    fn nparams(&self) -> usize {
        self.rhs.nparams()
    }
    fn nout(&self) -> usize {
        self.rhs.nout()
    }
    fn jacobian_action(&self, x: &M::V, p: &M::V, v: &M::V, y: &mut M::V) {
        self.rhs.jacobian_action(x, p, v, y);
        self.mass.gemv(v, p,  M::T::one(), -self.c, y);
    }
    
}

impl<M: Matrix, CRhs: Callable<M::V> + Jacobian<M>, CMass: Callable<M::V> + Jacobian<M>> Jacobian<M> for BdfCallable<'_, M, CRhs, CMass> 
where 
    for <'b> &'b M::V: VectorRef<M::V>,
{
    fn jacobian(&self, x: &M::V, p: &M::V) -> M {
        if self.mass_jacobian_is_stale {
            self.mass_jac = self.mass.jacobian(x, p);
        }
        if self.rhs_jacobian_is_stale {
            self.rhs_jac = self.rhs.jacobian(x, p);
            self.rhs_jacobian_is_stale = false;
        }
        if self.jacobian_is_stale {
            self.jac = self.mass_jac - self.rhs_jac * self.c; 
            self.jacobian_is_stale = false;
        }
        self.jac
    }
}

