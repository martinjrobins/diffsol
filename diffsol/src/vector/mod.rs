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

/// A trait for types that represent a collection of indices into a vector.
///
/// This is used to represent subsets of vector elements, typically for algebraic constraints
/// or when operating on specific vector components.
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

/// Common interface for vector-like types, providing access to scalar type, context, and inner representation.
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

/// Operations on vectors by value (addition and subtraction).
///
/// This trait defines vector addition and subtraction when both operands are owned or references.
pub trait VectorOpsByValue<Rhs = Self, Output = Self>:
    VectorCommon + Add<Rhs, Output = Output> + Sub<Rhs, Output = Output>
{
}

impl<V, Rhs, Output> VectorOpsByValue<Rhs, Output> for V where
    V: VectorCommon + Add<Rhs, Output = Output> + Sub<Rhs, Output = Output>
{
}

/// In-place operations on vectors (addition and subtraction).
///
/// This trait defines in-place vector addition and subtraction (self += rhs, self -= rhs).
pub trait VectorMutOpsByValue<Rhs = Self>: VectorCommon + AddAssign<Rhs> + SubAssign<Rhs> {}

impl<V, Rhs> VectorMutOpsByValue<Rhs> for V where V: VectorCommon + AddAssign<Rhs> + SubAssign<Rhs> {}

/// Operations on a reference to a vector, supporting addition, subtraction, and scalar multiplication.
///
/// This trait ensures that vector references can be used in arithmetic operations with owned vectors,
/// other references, and vector views, enabling flexible composition of vector operations.
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

/// A mutable view into a vector, supporting in-place operations and modifications.
///
/// This trait represents a temporary mutable reference to a vector's data, allowing in-place
/// arithmetic operations (+=, -=, *=) and other modifications. Mutable views can be created
/// via the `as_view_mut()` method on a `Vector`.
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
    /// Copy values from an owned vector into this view.
    fn copy_from(&mut self, other: &Self::Owned);
    /// Copy values from another vector view into this view.
    fn copy_from_view(&mut self, other: &Self::View);
    /// Compute the AXPY operation: self = alpha * x + beta * self
    fn axpy(&mut self, alpha: Self::T, x: &Self::Owned, beta: Self::T);
}

/// A borrowed immutable view of a vector, supporting read-only arithmetic operations.
///
/// This trait represents a temporary immutable reference to a vector's data, allowing read-only
/// operations like addition, subtraction, and scalar multiplication. Vector views can be created
/// via the `as_view()` method on a `Vector` and are cheaper to create than cloning.
pub trait VectorView<'a>:
    VectorOpsByValue<Self, Self::Owned>
    + VectorOpsByValue<Self::Owned, Self::Owned>
    + for<'b> VectorOpsByValue<&'b Self::Owned, Self::Owned>
    + for<'b> VectorOpsByValue<&'b Self, Self::Owned>
    + Mul<Scale<Self::T>, Output = Self::Owned>
{
    type Owned;
    /// Compute the squared weighted norm: sum_i ((self_i) / (|y_i| * rtol + atol_i))^2
    ///
    /// This is commonly used for error control in ODE solvers.
    fn squared_norm(&self, y: &Self::Owned, atol: &Self::Owned, rtol: Self::T) -> Self::T;
    /// Convert this view into an owned vector, cloning the data if necessary.
    fn into_owned(self) -> Self::Owned;
}

/// A complete vector abstraction supporting arithmetic operations, norms, and index operations.
///
/// This is the main vector trait used throughout diffsol. Implementing vectors can be hosted on CPU (see `VectorHost`) or GPU.
/// Users typically do not need to implement this trait; use provided implementations like
/// `NalgebraVec` or `FaerVec`.
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

    /// Get the context associated with this vector (for device placement, threading, etc.).
    fn context(&self) -> &Self::C;

    /// Get a mutable reference to the inner representation of the vector.
    fn inner_mut(&mut self) -> &mut Self::Inner;

    /// Set the value at the specified index to `value`.
    fn set_index(&mut self, index: IndexType, value: Self::T);

    /// Get the value at the specified index.
    fn get_index(&self, index: IndexType) -> Self::T;

    /// Compute the $\ell_k$ norm: $(\sum_i |x_i|^k)^{1/k}$
    fn norm(&self, k: i32) -> Self::T;

    /// Compute the squared weighted norm for error control: $\sum_i (x_i / (|y_i| \cdot \text{rtol} + \text{atol}_i))^2$
    ///
    /// This norm is used by ODE solvers for adaptive error control.
    fn squared_norm(&self, y: &Self, atol: &Self, rtol: Self::T) -> Self::T;

    /// Get the length (number of elements) in this vector.
    fn len(&self) -> IndexType;

    /// Check if the vector is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Create a vector of length `nstates` with all elements initialized to `value`.
    fn from_element(nstates: usize, value: Self::T, ctx: Self::C) -> Self;

    /// Create a vector of length `nstates` with all elements set to zero.
    fn zeros(nstates: usize, ctx: Self::C) -> Self {
        Self::from_element(nstates, Self::T::zero(), ctx)
    }

    /// Fill all elements of this vector with `value`.
    fn fill(&mut self, value: Self::T);

    /// Create an immutable view of this vector.
    fn as_view(&self) -> Self::View<'_>;

    /// Create a mutable view of this vector.
    fn as_view_mut(&mut self) -> Self::ViewMut<'_>;

    /// Copy all values from `other` into this vector.
    fn copy_from(&mut self, other: &Self);

    /// Copy all values from a vector view into this vector.
    fn copy_from_view(&mut self, other: &Self::View<'_>);

    // TODO: would prefer to use From trait but not implemented for faer::Col
    /// Create a vector from a Rust `Vec`.
    fn from_vec(vec: Vec<Self::T>, ctx: Self::C) -> Self;

    /// Create a vector from a slice.
    fn from_slice(slice: &[Self::T], ctx: Self::C) -> Self;

    // TODO: would prefer to use From trait but not implemented for faer::Col
    /// Clone this vector as a Rust `Vec`.
    fn clone_as_vec(&self) -> Vec<Self::T>;

    /// Compute the AXPY operation: self = alpha * x + beta * self
    fn axpy(&mut self, alpha: Self::T, x: &Self, beta: Self::T);

    /// Compute the AXPY operation with a vector view: self = alpha * x + beta * self
    fn axpy_v(&mut self, alpha: Self::T, x: &Self::View<'_>, beta: Self::T);

    /// Element-wise multiplication: self_i *= other_i
    fn component_mul_assign(&mut self, other: &Self);

    /// Element-wise division: self_i /= other_i
    fn component_div_assign(&mut self, other: &Self);

    /// Detect roots (zero crossings) between this vector (as g0) and another vector (g1).
    ///
    /// Returns a tuple of:
    /// - `bool`: true if a zero crossing is found (g1_i == 0 for some i)
    /// - `T`: the interpolation factor at the maximum crossing (0 if none found) (given by maxmimum |g0_i / (g1_i - g0_i)|)
    /// - `i32`: the index of the maximum crossing (-1 if none found)
    fn root_finding(&self, g1: &Self) -> (bool, Self::T, i32);

    /// Assign `value` to all elements at the specified indices.
    fn assign_at_indices(&mut self, indices: &Self::Index, value: Self::T);

    /// Copy values from `other` at the specified indices: self\[indices\[i\]\] = other\[indices\[i\]\]
    fn copy_from_indices(&mut self, other: &Self, indices: &Self::Index);

    /// Gather values from `other` at indices: self\[i\] = other\[indices\[i\]\]
    fn gather(&mut self, other: &Self, indices: &Self::Index);

    /// Scatter values to `other` at indices: other\[indices\[i\]\] = self\[i\]
    fn scatter(&self, indices: &Self::Index, other: &mut Self);

    /// Assert that this vector equals `other` within a scalar tolerance `tol`.
    fn assert_eq_st(&self, other: &Self, tol: Self::T) {
        let tol = vec![tol; self.len()];
        Self::assert_eq_vec(self.clone_as_vec(), other.clone_as_vec(), tol);
    }

    /// Assert that this vector equals `other` using a weighted norm (same as used by ODE solvers).
    ///
    /// Uses `squared_norm` internally with the scaling factors, and asserts that the resulting
    /// norm is less than `factor`.
    fn assert_eq_norm(&self, other: &Self, atol: &Self, rtol: Self::T, factor: Self::T) {
        let error = self.clone() - other.clone();
        let error_norm = error.squared_norm(other, atol, rtol).sqrt();
        assert!(
            error_norm < factor,
            "error_norm: {error_norm}. self: {self:?}, other: {other:?}",
        );
    }

    /// Assert that this vector equals `other` using a vector of per-element tolerances.
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

/// A vector hosted on the CPU, supporting direct indexing and slice access.
///
/// This trait extends `Vector` with the ability to directly access vector elements via indexing
/// (`v[i]`) and to get slices of the underlying data. This is useful for algorithms that need
/// direct CPU-side access to vector data. GPU vectors typically do not implement this trait.
pub trait VectorHost:
    Vector + Index<IndexType, Output = Self::T> + IndexMut<IndexType, Output = Self::T>
{
    /// Get the vector data as an immutable slice.
    fn as_slice(&self) -> &[Self::T];

    /// Get the vector data as a mutable slice.
    fn as_mut_slice(&mut self) -> &mut [Self::T];
}

/// Marker trait for vectors that have a default associated dense matrix type.
///
/// This trait associates a vector type with its corresponding dense matrix representation,
/// enabling vectors to be easily combined with matrix types for linear algebra operations.
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
