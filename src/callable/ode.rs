use nalgebra::Unit;

use crate::{Scalar, Vector, Matrix, IndexType};

use super::{Callable, Jacobian, unit::UnitCallable};

// callable to solve for F(y) = M (y' + psi) - c * f(y) = 0 
pub struct BdfCallable<'a, T: Scalar, V: Vector<T>, M: Matrix<T, V>, CRhs: Callable<T, V>, CMass: Callable<T, V>> {
    rhs: &'a CRhs,
    mass: &'a CMass,
    psi_neg_y0: V,
    c: T,
    rhs_jac: M,
    jac: M,
    mass_jac: M,
    rhs_jacobian_is_stale: bool,
    jacobian_is_stale: bool,
    mass_jacobian_is_stale: bool,
}

impl<'a, T: Scalar, V: Vector<T>, M: Matrix<T, V>, CRhs: Callable<T, V>, CMass: Callable<T, V>> BdfCallable<'a, T, V, M, CRhs, CMass> {
    pub fn new(rhs: &'a CRhs, mass: &'a CMass) -> Self {
        let n = rhs.nstates();
        let c = T::zero();
        let psi_neg_y0 = V::zeros(n);
        Self { rhs, mass, psi_neg_y0, c, rhs_jacobian_is_stale: true, jacobian_is_stale: true, mass_jacobian_is_stale: true, rhs_jac: M::zeros(n, n), jac: M::zeros(n, n), mass_jac: M::zeros(n, n) }
    }
    pub fn set_c(&mut self, h: T, alpha: &[T], order: IndexType) {
        self.c = h * alpha[usize::from(order)];
        self.jacobian_is_stale = true;
    }
    pub fn set_psi_and_y0(&mut self, diff: &M, gamma: &[T], alpha: &[T], order: usize, y0: &V) {
        // update psi term as defined in second equation on page 9 of [1]
        self.psi_neg_y0 = diff.column(0) * gamma[0];
        for i in 1..=order {
            self.psi_neg_y0 += diff.column(i) * gamma[i]
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
impl<'a, T: Scalar, V: Vector<T>, M: Matrix<T, V>, CRhs: Callable<T, V>, CMass: Callable<T, V>> Callable<T, V> for BdfCallable<'a, T, V, M, CRhs, CMass> {

    // F(y) = M (y - y0 + psi) - c * f(y) = 0
    fn call(&self, x: &V, p: &V, y: &mut V) {
        self.rhs.call(x, p, y);
        let tmp = *x - self.psi_neg_y0;
        self.mass.gemv(&tmp, p, T::one(), -self.c, y);
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
    
}

impl<T: Scalar, V: Vector<T>, M: Matrix<T, V>, CRhs: Callable<T, V> + Jacobian<T, V, M>, CMass: Callable<T, V> + Jacobian<T, V, M>> Jacobian<T, V, M> for BdfCallable<'_, T, V, M, CRhs, CMass> {
    fn jacobian(&self, x: &V, p: &V) -> M {
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

