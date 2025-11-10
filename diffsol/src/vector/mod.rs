use crate::matrix::DenseMatrix;
use crate::scalar::Scale;
use crate::{Context, IndexType, Scalar};
use nalgebra::ComplexField;
use num_traits::Zero;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

#[cfg(feature = "faer")]
pub mod faer_serial;
#[cfg(feature = "nalgebra")]
pub mod nalgebra_serial;

#[cfg(feature = "cuda")]
pub mod cuda;

#[macro_use]
mod utils;

pub trait VectorIndex: Sized + Debug + Clone {
    type C: Context;
    fn context(&self) -> &Self::C;
    fn zeros(len: IndexType, ctx: Self::C) -> Self;
    fn len(&self) -> IndexType;
    fn clone_as_vec(&self) -> Vec<IndexType>;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn from_vec(v: Vec<IndexType>, ctx: Self::C) -> Self;
}

pub trait VectorCommon: Sized + Debug {
    type T: Scalar;
    type C: Context;
    type Inner;

    fn inner(&self) -> &Self::Inner;
}

impl<V> VectorCommon for &V
where
    V: VectorCommon,
{
    type T = V::T;
    type C = V::C;
    type Inner = V::Inner;
    fn inner(&self) -> &Self::Inner {
        V::inner(self)
    }
}

impl<V> VectorCommon for &mut V
where
    V: VectorCommon,
{
    type T = V::T;
    type C = V::C;
    type Inner = V::Inner;
    fn inner(&self) -> &Self::Inner {
        V::inner(self)
    }
}

pub trait VectorOpsByValue<Rhs = Self, Output = Self>:
    VectorCommon + Add<Rhs, Output = Output> + Sub<Rhs, Output = Output>
{
}

impl<V, Rhs, Output> VectorOpsByValue<Rhs, Output> for V where
    V: VectorCommon + Add<Rhs, Output = Output> + Sub<Rhs, Output = Output>
{
}

pub trait VectorMutOpsByValue<Rhs = Self>: VectorCommon + AddAssign<Rhs> + SubAssign<Rhs> {}

impl<V, Rhs> VectorMutOpsByValue<Rhs> for V where V: VectorCommon + AddAssign<Rhs> + SubAssign<Rhs> {}

pub trait VectorRef<V: Vector>:
    VectorOpsByValue<V, V>
    + for<'a> VectorOpsByValue<&'a V, V>
    + for<'a> VectorOpsByValue<V::View<'a>, V>
    + for<'a, 'b> VectorOpsByValue<&'a V::View<'b>, V>
    + Mul<Scale<V::T>, Output = V>
{
}

impl<RefT, V: Vector> VectorRef<V> for RefT where
    RefT: VectorOpsByValue<V, V>
        + for<'a> VectorOpsByValue<&'a V, V>
        + for<'a> VectorOpsByValue<V::View<'a>, V>
        + for<'a, 'b> VectorOpsByValue<&'a V::View<'b>, V>
        + Mul<Scale<V::T>, Output = V>
{
}

pub trait VectorViewMut<'a>:
    VectorMutOpsByValue<Self::View>
    + VectorMutOpsByValue<Self::Owned>
    + for<'b> VectorMutOpsByValue<&'b Self::View>
    + for<'b> VectorMutOpsByValue<&'b Self::Owned>
    + MulAssign<Scale<Self::T>>
{
    type Owned;
    type View;
    type Index: VectorIndex;
    fn copy_from(&mut self, other: &Self::Owned);
    fn copy_from_view(&mut self, other: &Self::View);
    fn axpy(&mut self, alpha: Self::T, x: &Self::Owned, beta: Self::T);
}

pub trait VectorView<'a>:
    VectorOpsByValue<Self, Self::Owned>
    + VectorOpsByValue<Self::Owned, Self::Owned>
    + for<'b> VectorOpsByValue<&'b Self::Owned, Self::Owned>
    + for<'b> VectorOpsByValue<&'b Self, Self::Owned>
    + Mul<Scale<Self::T>, Output = Self::Owned>
{
    type Owned;
    fn squared_norm(&self, y: &Self::Owned, atol: &Self::Owned, rtol: Self::T) -> Self::T;
    fn into_owned(self) -> Self::Owned;
}

pub trait Vector:
    VectorOpsByValue<Self>
    + for<'b> VectorOpsByValue<&'b Self>
    + for<'a> VectorOpsByValue<Self::View<'a>>
    + for<'a, 'b> VectorOpsByValue<&'b Self::View<'a>>
    + Mul<Scale<Self::T>, Output = Self>
    + Div<Scale<Self::T>, Output = Self>
    + VectorMutOpsByValue<Self>
    + for<'a> VectorMutOpsByValue<Self::View<'a>>
    + for<'b> VectorMutOpsByValue<&'b Self>
    + for<'a, 'b> VectorMutOpsByValue<&'b Self::View<'a>>
    + MulAssign<Scale<Self::T>>
    + Clone
{
    type View<'a>: VectorView<'a, T = Self::T, Owned = Self>
    where
        Self: 'a;
    type ViewMut<'a>: VectorViewMut<'a, T = Self::T, Owned = Self, View = Self::View<'a>>
    where
        Self: 'a;
    type Index: VectorIndex;

    /// get the context
    fn context(&self) -> &Self::C;

    fn inner_mut(&mut self) -> &mut Self::Inner;

    /// set the value at index `index` to `value`, might be slower than using `IndexMut`
    /// if the vector is not on the host
    fn set_index(&mut self, index: IndexType, value: Self::T);

    /// get the value at index `index`, might be slower than using `Index`
    /// if the vector is not on the host
    fn get_index(&self, index: IndexType) -> Self::T;

    /// returns (\sum_i (x_i)^k)^(1/k)
    fn norm(&self, k: i32) -> Self::T;

    /// returns \sum_i (x_i / (|y_i| * rtol + atol_i))^2
    fn squared_norm(&self, y: &Self, atol: &Self, rtol: Self::T) -> Self::T;

    /// get the length of the vector
    fn len(&self) -> IndexType;

    /// check if the vector is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// create a vector of size `nstates` with all elements set to `value`
    fn from_element(nstates: usize, value: Self::T, ctx: Self::C) -> Self;

    /// create a vector of size `nstates` with all elements set to zero
    fn zeros(nstates: usize, ctx: Self::C) -> Self {
        Self::from_element(nstates, Self::T::zero(), ctx)
    }

    /// fill the vector with `value`
    fn fill(&mut self, value: Self::T);

    /// create a view of the vector
    fn as_view(&self) -> Self::View<'_>;

    /// create a mutable view of the vector
    fn as_view_mut(&mut self) -> Self::ViewMut<'_>;

    /// copy the values from `other` to `self`
    fn copy_from(&mut self, other: &Self);

    /// copy the values from `other` to `self`
    fn copy_from_view(&mut self, other: &Self::View<'_>);

    /// create a vector from a Vec
    /// TODO: would prefer to use From trait but not implemented for faer::Col
    fn from_vec(vec: Vec<Self::T>, ctx: Self::C) -> Self;

    /// create a vector from a slice
    fn from_slice(slice: &[Self::T], ctx: Self::C) -> Self;

    /// convert the vector to a Vec
    /// TODO: would prefer to use Into trait but not implemented for faer::Col
    fn clone_as_vec(&self) -> Vec<Self::T>;

    /// axpy operation: self = alpha * x + beta * self
    fn axpy(&mut self, alpha: Self::T, x: &Self, beta: Self::T);

    /// axpy operation: self = alpha * x + beta * self
    fn axpy_v(&mut self, alpha: Self::T, x: &Self::View<'_>, beta: Self::T);

    /// element-wise multiplication
    fn component_mul_assign(&mut self, other: &Self);

    /// element-wise division
    fn component_div_assign(&mut self, other: &Self);

    /// given two vectors `g0=self` and `g1`, return:
    /// - `true` if a root is found in g1 (i.e. g1_i == 0)
    /// - for all values of i where a zero crossing is found (i.e. g0_i * g1_i < 0), return:
    ///     - max_i(abs(g1_i / (g1_i - g0_i))), 0 otherwise
    ///     - the index i at the maximum value, -1 otherwise
    fn root_finding(&self, g1: &Self) -> (bool, Self::T, i32);

    /// assign `value` to the elements of `self` at the indices specified by `indices`
    /// i.e. `self[indices[i]] = value` for all i
    fn assign_at_indices(&mut self, indices: &Self::Index, value: Self::T);

    /// copy from `other` at the indices specified by `indices`
    /// generaly `self` and `other` have the same length
    /// i.e. `self[indices[i]] = other[indices[i]]` for all i
    fn copy_from_indices(&mut self, other: &Self, indices: &Self::Index);

    /// gather values from `other` at the indices specified by `indices`
    /// i.e. `self[i] = other[indices[i]]` for all i
    fn gather(&mut self, other: &Self, indices: &Self::Index);

    /// scatter values from `self` to `other` at the indices specified by `indices`
    /// i.e. `other[indices[i]] = self[i]` for all i
    fn scatter(&self, indices: &Self::Index, other: &mut Self);

    /// assert that `self` is equal to `other` within a tolerance `tol`
    fn assert_eq_st(&self, other: &Self, tol: Self::T) {
        let tol = vec![tol; self.len()];
        Self::assert_eq_vec(self.clone_as_vec(), other.clone_as_vec(), tol);
    }

    /// assert that `self` is equal to `other` using the same norm used by the solvers
    fn assert_eq_norm(&self, other: &Self, atol: &Self, rtol: Self::T, factor: Self::T) {
        let error = self.clone() - other.clone();
        let error_norm = error.squared_norm(other, atol, rtol).sqrt();
        assert!(
            error_norm < factor,
            "error_norm: {error_norm}. self: {self:?}, other: {other:?}",
        );
    }

    /// assert that `self` is equal to `other` within a vector tolerance `tol`
    fn assert_eq(&self, other: &Self, tol: &Self) {
        assert_eq!(
            self.len(),
            other.len(),
            "Vector length mismatch: {} != {}",
            self.len(),
            other.len()
        );
        let s = self.clone_as_vec();
        let other = other.clone_as_vec();
        let tol = tol.clone_as_vec();
        Self::assert_eq_vec(s, other, tol);
    }

    fn assert_eq_vec(s: Vec<Self::T>, other: Vec<Self::T>, tol: Vec<Self::T>) {
        for i in 0..s.len() {
            if num_traits::abs(s[i] - other[i]) > tol[i] {
                eprintln!(
                    "Vector element mismatch at index {i}: {} != {}",
                    s[i], other[i]
                );
                if s.len() <= 3 {
                    eprintln!("left: {s:?}");
                    eprintln!("right: {other:?}");
                } else if i == 0 {
                    eprintln!(
                        "left: [{}, {}, {}] != [{}, {}, {}]",
                        s[0], s[1], s[2], other[0], other[1], other[2]
                    );
                } else if i == s.len() - 1 {
                    eprintln!(
                        "left: [..., {}, {}, {}] != [..., {}, {}, {}]",
                        s[i - 2],
                        s[i - 1],
                        s[i],
                        other[i - 2],
                        other[i - 1],
                        other[i]
                    );
                } else {
                    eprintln!(
                        "left: [..., {}, {}, {}, ...] != [..., {}, {}, {}, ...]",
                        s[i - 1],
                        s[i],
                        s[i + 1],
                        other[i - 1],
                        other[i],
                        other[i + 1]
                    );
                }
                panic!(
                    "Vector element mismatch at index {}: {} != {}",
                    i, s[i], other[i]
                );
            }
        }
    }
}

pub trait VectorHost:
    Vector + Index<IndexType, Output = Self::T> + IndexMut<IndexType, Output = Self::T>
{
    /// get the vector as a slice
    /// TODO: not compatible with gpu vectors? but used for diffsl api
    fn as_slice(&self) -> &[Self::T];

    /// get the vector as a mut slice
    /// TODO: not compatible with gpu vectors? but used for diffsl api
    fn as_mut_slice(&mut self) -> &mut [Self::T];
}

pub trait DefaultDenseMatrix: Vector {
    type M: DenseMatrix<V = Self, T = Self::T, C = Self::C>;
}

#[cfg(test)]
mod tests {
    use super::Vector;
    use num_traits::FromPrimitive;

    pub fn test_root_finding<V: Vector>() {
        let g0 = V::from_vec(
            vec![
                V::T::from_f64(1.0).unwrap(),
                V::T::from_f64(-2.0).unwrap(),
                V::T::from_f64(3.0).unwrap(),
            ],
            Default::default(),
        );
        let g1 = V::from_vec(
            vec![
                V::T::from_f64(1.0).unwrap(),
                V::T::from_f64(2.0).unwrap(),
                V::T::from_f64(3.0).unwrap(),
            ],
            Default::default(),
        );
        let (found_root, max_frac, max_frac_index) = g0.root_finding(&g1);
        assert!(!found_root);
        assert_eq!(max_frac, V::T::from_f64(0.5).unwrap());
        assert_eq!(max_frac_index, 1);

        let g0 = V::from_vec(
            vec![
                V::T::from_f64(1.0).unwrap(),
                V::T::from_f64(-2.0).unwrap(),
                V::T::from_f64(3.0).unwrap(),
            ],
            Default::default(),
        );
        let g1 = V::from_vec(
            vec![
                V::T::from_f64(1.0).unwrap(),
                V::T::from_f64(2.0).unwrap(),
                V::T::from_f64(0.0).unwrap(),
            ],
            Default::default(),
        );
        let (found_root, max_frac, max_frac_index) = g0.root_finding(&g1);
        assert!(found_root);
        assert_eq!(max_frac, V::T::from_f64(0.5).unwrap());
        assert_eq!(max_frac_index, 1);

        let g0 = V::from_vec(
            vec![
                V::T::from_f64(1.0).unwrap(),
                V::T::from_f64(-2.0).unwrap(),
                V::T::from_f64(3.0).unwrap(),
            ],
            Default::default(),
        );
        let g1 = V::from_vec(
            vec![
                V::T::from_f64(1.0).unwrap(),
                V::T::from_f64(-2.0).unwrap(),
                V::T::from_f64(3.0).unwrap(),
            ],
            Default::default(),
        );
        let (found_root, max_frac, max_frac_index) = g0.root_finding(&g1);
        assert!(!found_root);
        assert_eq!(max_frac, V::T::from_f64(0.0).unwrap());
        assert_eq!(max_frac_index, -1);
    }
}
