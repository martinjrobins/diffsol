use crate::{Scalar, Vector, Matrix, NonLinearSolver, callable::{Callable, ode::BdfCallable}, nonlinear_solver};

pub mod bdf;

trait OdeSolverMethod<T: Scalar, V: Vector<T>, M: Matrix<T, V>, C: Callable<T, V>> {
    fn new(callable: C, y0: V, p: V, t0: T, h0: T, mass: M) -> Self;
    fn step(&mut self, t: T) -> T;
    fn interpolate(&self, t: T) -> V;
}

pub struct OdeSolver<Method: OdeSolverMethod<T, V, M, C>, T: Scalar, V: Vector<T>, M: Matrix<T, V>, C: Callable<T, V>> {
    method: Method,
    t: T,
}

impl <Method: OdeSolverMethod<T, V, M, C>, T: Scalar, V: Vector<T>, M: Matrix<T, V>, C: Callable<T, V>> OdeSolver<Method, T, V, M, C> {
    pub fn new(rhs: C1, alg: C2, y0: V, p: V, t0: T, h0: T, mass: M) -> Self {
        let y0 = self.calculate_initial_y0(callable, y0, p, t0, h0, mass)?;
        let method = Method::new(callable, y0, p, t0, h0, mass);
        Self { method, t: t0 }
    }
    
    fn calculate_initial_y0(callable: C, y0: V, p: V, t0: T, mass: M) -> Result<V> {
        let mut mask = mass.diagonal();
        mask.map(|x| if x == T::zero() { T::one() } else { T::zero() });
        let newton = Newton::new(callable, mask);
        newton.solve(&y0)?
    }
        

    pub fn solve(&mut self, t: T) -> V {
        while self.t <= t {
            self.t = self.method.step(t);
        }
        self.method.interpolate(t)
    }
}