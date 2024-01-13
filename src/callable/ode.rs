use crate::{Scalar, Vector, Matrix};

use super::Callable;

// callable to solve for F(y) = M (y' + psi) - c * f(y) = 0 
pub struct BdfCallable<'a, T: Scalar, V: Vector<T>, M: Matrix<T, V>> {
    rhs: &'a dyn Callable<T, V>,
    mass: &'a dyn Callable<T, V>,
    psi_neg_y0: V,
    c: T,
    rhs_jac: M,
    jac: M,
    rhs_jacobian_is_stale: bool,
    jacobian_is_stale: bool,
}

impl<'a, T: Scalar, V: Vector<T>, M: Matrix<T, V>> BdfCallable<'a, T, V, M> {
    pub fn new<C: Callable<T, V>>(rhs: &'a impl Callable<T, V>, mass: &'a C) -> Self {
        let n = rhs.nstates();
        let c = T::zero();
        let psi_neg_y0 = V::zeros(n);
        Self { rhs, mass, psi_neg_y0, c, rhs_jacobian_is_stale: true, jacobian_is_stale: true, rhs_jac: M::zeros(n, n), jac: M::zeros(n, n) }
    }
    pub fn set_c(&mut self, h: T, alpha: &[T], order: u32) {
        self.c = h * alpha[order];
        self.jacobian_is_stale = true;
    }
    pub fn set_psi_and_y0<M2: Matrix<T, V>>(&mut self, diff: &M2, gamma: &[T], alpha: &[T], order: usize, y0: &V) {
        // update psi term as defined in second equation on page 9 of [1]
        self.psi_neg_y0 = diff.row(0) * gamma[0];
        for i in 1..=order {
            self.psi_neg_y0 += diff.row(i) * gamma[i]
        }
        self.psi_neg_y0 *= alpha[order];

        // now negate y0
        self.psi_neg_y0 -= y0;
    }
    pub fn set_rhs_jacobian_is_stale(&mut self) {
        self.rhs_jacobian_is_stale = true;
        self.jacobian_is_stale = true;
    }
}


// callable to solve for F(y) = M (y' + psi) - f(y) = 0 
impl<'a, RightHandSide: Callable<T, V>, T: Scalar, V: Vector<T>, M: Matrix<T, V>> Callable<T, V> for BdfCallable<'a, T, V, M> {

    // F(y) = M (y - y0 + psi) - c * f(y) = 0
    fn call(&self, x: &V, p: &V, y: &mut V) {
        self.rhs.call(x, p, y);
        self.mass.gemv(x - &self.psi_neg_y0, p, T::one(), T::from(-self.c), y);
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
    fn jacobian_action(&self, x: &V, p: &V, v: &V, y: &mut V) {
        self.rhs.jacobian_action(x, p, v, y);
        self.mass.gemv(v, p,  T::one(), -self.c, y);
    }
    fn jacobian<M2: Matrix<T, V>>(&self, p: &V) -> M2 {
        if self.rhs_jacobian_is_stale {
            self.rhs_jac = self.rhs.jacobian(p);
            self.rhs_jacobian_is_stale = false;
        }
        if self.jacobian_is_stale {
            self.jac = self.mass - self.c * &self.rhs_jac;
            self.jacobian_is_stale = false;
        }
        self.jac
    }
}
