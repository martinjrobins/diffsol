use crate::{Scalar, Vector, Matrix, IndexType};

use super::Callable;

// callable to solve for F(y) = M (y' + psi) - c * f(y) = 0 
pub struct BdfCallable<RightHandSide: Callable<T, V>, T: Scalar, V: Vector<T>, M: Matrix<T, V>> {
    rhs: RightHandSide,
    mass: M,
    psi: T,
    c: T,
    phantom: std::marker::PhantomData<V>,
}

impl<RightHandSide: Callable<T, V>, T: Scalar, V: Vector<T>, M: Matrix<T, V>> BdfCallable<RightHandSide, T, V, M> {
    pub fn new(rhs: RightHandSide, mass: M, psi: T, c: T) -> Self {
        Self { rhs, mass, psi, c, phantom: std::marker::PhantomData }
    }
}


// callable to solve for F(y) = M (y' + psi) - f(y) = 0 
impl<RightHandSide: Callable<T, V>, T: Scalar, V: Vector<T>, M: Matrix<T, V>> Callable<T, V> for BdfCallable<RightHandSide, T, V, M> {
    // F(y) = M (y' + psi) - c * f(y) = 0
    fn call(&self, x: &V, p: &V, y: &mut V) {
        self.rhs.call(x, p, &mut y);
        y.gemv(-1.0, &self.mass, x + self.psi, self.c);
    }
    fn nstates(&self) -> usize {
        self.rhs.nstates()
    }
    fn nparams(&self) -> usize {
        self.rhs.nparams()
    }
    // J_F = M - c * J_f
    fn jacobian_action(&self, x: &V, p: &V, v: &V, y: &mut V) {
        self.rhs.jacobian_action(x, p, v, y);
        y.gemv(-self.c, &self.mass, v, 1.0);
    }
}
