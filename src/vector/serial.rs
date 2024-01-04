use nalgebra::DVector;

use crate::Scalar;

use super::Vector;


impl<T: Scalar> Vector<T> for DVector<T> {
    fn abs(&self) -> Self {
        self.abs()
    }
    fn add_scalar_mut(&mut self, scalar: T) {
        self.add_scalar_mut(scalar);
    }
    fn from_element(nstates: usize, value: T) -> Self {
        Self::from_element(nstates, value)
    }
    fn len(&self) -> usize {
        self.len()
    }
    fn component_mul_assign(&mut self, other: &Self) {
        self.component_mul_assign(other);
    }
    fn component_div_assign(&mut self, other: &Self) {
        self.component_div_assign(other);
    }
    fn norm(&self) -> T {
        self.norm()
    }
    fn from_vec(vec: Vec<T>) -> Self {
        Self::from_vec(vec)
    }
    fn map_mut<F: Fn(T) -> T>(&mut self, f: F) {
        self.map_mut(f);
    }
}


// tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abs() {
        let v = DVector::from_vec(vec![1.0, -2.0, 3.0]);
        let v_abs = v.abs();
        assert_eq!(v_abs, DVector::from_vec(vec![1.0, 2.0, 3.0]));
    }
}