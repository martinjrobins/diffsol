use crate::{Matrix, Scalar, Vector};
use anyhow::Result;

mod lu;

pub trait LinearSolver<T: Scalar, V: Vector<T>, M: Matrix<T, V>> {
    fn new(a: M) -> Self;
    fn solve(&self, b: &V) -> Result<V>;
}

#[cfg(test)]
pub mod tests {
    use super::*;

    pub fn test_linear_solver<T: Scalar, V: Vector<T>, M: Matrix<T, V>, S: LinearSolver<T, V, M>>() {
        let triplets = vec![(0, 0, 2.0.into()), (1, 1, 2.0.into())];
        let a = M::try_from_triplets(2, 2, triplets).unwrap();
        let b = V::from_vec(vec![1.0.into(), 2.0.into()]);
        let x = S::new(a).solve(&b).unwrap();
        let expect = V::from_vec(vec![(0.5).into(), 1.0.into()]);
        x.assert_eq(&expect, 1e-6.into());
    }
    
    #[test]
    fn test_lu() {
        type T = f64;
        type V = nalgebra::DVector<T>;
        type M = nalgebra::DMatrix<T>;
        type S = nalgebra::LU<T, nalgebra::Dyn, nalgebra::Dyn>;
        test_linear_solver::<T, V, M, S>();
    }
}