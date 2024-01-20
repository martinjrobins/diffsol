pub mod lu;
pub mod gmres;

#[cfg(test)]
pub mod tests {
    use crate::{Matrix, Solver, LU, Scalar, Vector, callable::{closure::Closure, Callable}, solver::{SolverOptions, SolverProblem}};
    
    // 0 = J * x - 8
    fn square<T: Scalar, V: Vector<T>, M: Matrix<T, V>>(x: &V, p: &V, y: &mut V, jac: &M) {
        jac.gemv(T::one(), x, T::zero(), y); // y = J * x
        y.add_scalar_mut(T::from(-8.0));
    }

    // J = J * dx
    fn square_jacobian<T: Scalar, V: Vector<T>, M: Matrix<T, V>>(x: &V, p: &V, v: &V, y: &mut V, jac: &M) {
        jac.gemv(T::one(), v, T::zero(), y); // y = J * v
    }

    pub fn test_linear_solver<'a, T: Scalar, V: Vector<T>, M: Matrix<T, V>, C: Callable<T, V>, S: Solver<'a, T, V, C>>(mut s: S, op: C) {
        let problem = SolverProblem::new(&op, &V::zeros(0));
        s.set_problem(problem);
        let b = V::from_vec(vec![2.0.into(), 4.0.into()]);
        let x = s.solve(&b).unwrap();
        let expect = V::from_vec(vec![(5.0).into(), 6.0.into()]);
        x.assert_eq(&expect, 1e-6.into());
    }
    
    #[test]
    fn test_lu() {
        type T = f64;
        type V = nalgebra::DVector<T>;
        type M = nalgebra::DMatrix<T>;
        type C = Closure<fn(&V, &V, &mut V, &M), fn(&V, &V, &V, &mut V, &M), M>;
        type S = LU<T>;
        let lu = LU::<T>::default();
        let op = C::new(
            square,
            square_jacobian,
            M::from_diagonal(&V::from_vec(vec![2.0.into(), 2.0.into()])), 
            2,
        );
        test_linear_solver::<T, V, M, C, S>(lu, op);
    }
}