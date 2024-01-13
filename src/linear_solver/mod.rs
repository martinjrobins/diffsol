pub mod lu;
pub mod gmres;

#[cfg(test)]
pub mod tests {
    use crate::{Matrix, Solver, LU, Scalar, Vector, callable::closure::Closure};

    
    // 0 = J * x - 8
    fn square<T: Scalar, V: Vector<T>, M: Matrix<T, V>>(x: &V, y: &mut V, jac: &M) {
        jac.mul_to(x, y);
        y.add_scalar_mut(T::from(-8.0));
    }

    // J = J * dx
    fn square_jacobian<T: Scalar, V: Vector<T>, M: Matrix<T, V>>(x: &V, v: &V, y: &mut V, jac: &M) {
        jac.mul_to(x, y);
    }

    pub fn test_linear_solver<'a, T: Scalar, V: Vector<T>, M: Matrix<T, V>, S: Solver<'a, T, V>>(mut s: S) {
        let jac = Matrix::from_diagonal(&V::from_vec(vec![2.0.into(), 2.0.into()]));
        let op = Closure::<fn(&V, &mut V, &M), fn(&V, &V, &mut V, &M), M>::new(
            square,
            square_jacobian,
            jac, 
            2,
        );
        s.set_callable(&op, &V::zeros(0));
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
        test_linear_solver::<T, V, M>(LU::default());
    }
}