use crate::matrix::DenseMatrix;
use crate::scalar::Scale;
use crate::{Context, IndexType, Scalar};
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
    /// Set the value at the specified index (sets across all batches when nbatch > 1).
    fn set_index(&mut self, index: IndexType, value: Self::T);
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
    /// Get the value at the specified index (panics if nbatch > 1).
    fn get_index(&self, index: IndexType) -> Self::T;
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
    + Send
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

    /// Get the per-batch length (number of states) in this vector.
    /// For batched vectors, this returns `nstates`, not `nstates * nbatch`.
    fn len(&self) -> IndexType;

    /// Get the total number of elements stored, including all batches.
    /// Returns `self.len() * self.context().nbatch()`.
    fn total_len(&self) -> IndexType {
        self.len() * self.context().nbatch()
    }

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

    /// Get an immutable view of a single batch (with nbatch=1 context).
    fn get_batch(&self, batch: usize) -> Self::View<'_>;

    /// Get a mutable view of a single batch (with nbatch=1 context).
    fn get_batch_mut(&mut self, batch: usize) -> Self::ViewMut<'_>;

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

    /// Per-batch AXPY: self[i]_b = alpha[b] * x[i]_b + beta * self[i]_b
    ///
    /// `alpha` must have length equal to `self.context().nbatch()`.
    /// Each batch uses its own scalar multiplier `alpha[b]`.
    /// The `x` operand may broadcast if its `nbatch == 1`.
    fn batched_axpy(&mut self, alpha: &[Self::T], x: &Self, beta: Self::T);

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
        let tol = vec![tol; self.total_len()];
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
pub(crate) mod tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use super::{Vector, VectorCommon, VectorIndex, VectorView, VectorViewMut};
    use crate::context::nalgebra::NalgebraContext;
    use crate::scalar::Scale;
    use crate::vector::nalgebra_serial::NalgebraVec;
    use crate::Context;
    use num_traits::FromPrimitive;

    fn f<V: Vector>(x: f64) -> V::T {
        V::T::from_f64(x).unwrap()
    }

    fn fv<V: Vector>(xs: &[f64]) -> Vec<V::T> {
        xs.iter().map(|&x| f::<V>(x)).collect()
    }

    pub fn test_root_finding<V: Vector>() {
        let g0 = V::from_vec(fv::<V>(&[1.0, -2.0, 3.0]), Default::default());
        let g1 = V::from_vec(fv::<V>(&[1.0, 2.0, 3.0]), Default::default());
        let (found_root, max_frac, max_frac_index) = g0.root_finding(&g1);
        assert!(!found_root);
        assert_eq!(max_frac, f::<V>(0.5));
        assert_eq!(max_frac_index, 1);

        let g0 = V::from_vec(fv::<V>(&[1.0, -2.0, 3.0]), Default::default());
        let g1 = V::from_vec(fv::<V>(&[1.0, 2.0, 0.0]), Default::default());
        let (found_root, max_frac, max_frac_index) = g0.root_finding(&g1);
        assert!(found_root);
        assert_eq!(max_frac, f::<V>(0.5));
        assert_eq!(max_frac_index, 1);

        let g0 = V::from_vec(fv::<V>(&[1.0, -2.0, 3.0]), Default::default());
        let g1 = V::from_vec(fv::<V>(&[1.0, -2.0, 3.0]), Default::default());
        let (found_root, max_frac, max_frac_index) = g0.root_finding(&g1);
        assert!(!found_root);
        assert_eq!(max_frac, f::<V>(0.0));
        assert_eq!(max_frac_index, -1);
    }

    pub fn test_from_slice<V: Vector>() {
        let slice = fv::<V>(&[1.0, 2.0, 3.0]);
        let v = V::from_slice(&slice, Default::default());
        assert_eq!(v.clone_as_vec(), slice);
    }

    pub fn test_mul_scalar<V: Vector>() {
        let v = V::from_vec(fv::<V>(&[1.0, 2.0, 3.0]), Default::default());
        let result = v * Scale(f::<V>(2.0));
        assert_eq!(result.clone_as_vec(), fv::<V>(&[2.0, 4.0, 6.0]));
    }

    pub fn test_div_scalar<V: Vector>() {
        let v = V::from_vec(fv::<V>(&[2.0, 4.0, 6.0]), Default::default());
        let result = v / Scale(f::<V>(2.0));
        assert_eq!(result.clone_as_vec(), fv::<V>(&[1.0, 2.0, 3.0]));
    }

    pub fn test_axpy<V: Vector>() {
        let mut y = V::from_vec(fv::<V>(&[1.0, 2.0, 3.0]), Default::default());
        let x = V::from_vec(fv::<V>(&[4.0, 5.0, 6.0]), Default::default());
        y.axpy(f::<V>(2.0), &x, f::<V>(1.0));
        assert_eq!(y.clone_as_vec(), fv::<V>(&[9.0, 12.0, 15.0]));
        y.axpy(f::<V>(2.0), &x, f::<V>(0.0));
        assert_eq!(y.clone_as_vec(), fv::<V>(&[8.0, 10.0, 12.0]));
        y.axpy(f::<V>(0.0), &x, f::<V>(1.0));
        assert_eq!(y.clone_as_vec(), fv::<V>(&[8.0, 10.0, 12.0]));
    }

    pub fn test_copy_from_indices<V: Vector>() {
        let mut v1 = V::zeros(5, Default::default());
        let v2 = V::from_vec(fv::<V>(&[10.0, 20.0, 30.0, 40.0, 50.0]), Default::default());
        let indices = V::Index::from_vec(vec![0, 2, 4], Default::default());
        v1.copy_from_indices(&v2, &indices);
        assert_eq!(v1.clone_as_vec(), fv::<V>(&[10.0, 0.0, 30.0, 0.0, 50.0]));
    }

    pub fn test_gather<V: Vector>() {
        let mut result = V::zeros(3, Default::default());
        let v = V::from_vec(fv::<V>(&[10.0, 20.0, 30.0, 40.0]), Default::default());
        let indices = V::Index::from_vec(vec![3, 0, 2], Default::default());
        result.gather(&v, &indices);
        assert_eq!(result.clone_as_vec(), fv::<V>(&[40.0, 10.0, 30.0]));
    }

    pub fn test_scatter<V: Vector>() {
        let v = V::from_vec(fv::<V>(&[40.0, 10.0, 30.0]), Default::default());
        let indices = V::Index::from_vec(vec![3, 0, 2], Default::default());
        let mut result = V::zeros(4, Default::default());
        v.scatter(&indices, &mut result);
        assert_eq!(result.clone_as_vec(), fv::<V>(&[10.0, 0.0, 30.0, 40.0]));
    }

    pub fn test_copy_from_via_view_mut<V: Vector>() {
        let mut v1 = V::zeros(3, Default::default());
        let v2 = V::from_vec(fv::<V>(&[1.0, 2.0, 3.0]), Default::default());
        v1.as_view_mut().copy_from(&v2);
        assert_eq!(v1.clone_as_vec(), fv::<V>(&[1.0, 2.0, 3.0]));
    }

    pub fn test_set_index<V: Vector>() {
        let mut v = V::zeros(3, Default::default());
        v.set_index(1, f::<V>(42.0));
        assert_eq!(v.clone_as_vec(), fv::<V>(&[0.0, 42.0, 0.0]));
    }

    pub fn test_get_index<V: Vector>() {
        let v = V::from_vec(fv::<V>(&[1.0, 2.0, 3.0]), Default::default());
        assert_eq!(v.get_index(0), f::<V>(1.0));
        assert_eq!(v.get_index(2), f::<V>(3.0));
    }

    pub fn test_norm<V: Vector>() {
        let v = V::from_vec(fv::<V>(&[3.0, 4.0]), Default::default());
        let norm = v.norm(2);
        let diff = norm - f::<V>(5.0);
        assert!(num_traits::abs(diff) < f::<V>(1e-12));
    }

    pub fn test_norm_l1<V: Vector>() {
        let v = V::from_vec(fv::<V>(&[3.0, -4.0]), Default::default());
        let norm = v.norm(1);
        let diff = norm - f::<V>(7.0);
        assert!(num_traits::abs(diff) < f::<V>(1e-12));
    }

    pub fn test_squared_norm<V: Vector>() {
        let x = V::from_vec(fv::<V>(&[1.0, 2.0]), Default::default());
        let y = V::from_vec(fv::<V>(&[1.0, 1.0]), Default::default());
        let atol = V::from_vec(fv::<V>(&[1e-3, 1e-3]), Default::default());
        let rtol = f::<V>(1e-2);
        let norm = x.squared_norm(&y, &atol, rtol);
        let denom = f::<V>(1.0) * rtol + f::<V>(1e-3);
        let err0 = f::<V>(1.0) / denom;
        let err1 = f::<V>(2.0) / denom;
        let expected = (err0 * err0 + err1 * err1) / f::<V>(2.0);
        assert!(num_traits::abs(norm - expected) < f::<V>(1e-12));
    }

    pub fn test_fill<V: Vector>() {
        let mut v = V::zeros(3, Default::default());
        v.fill(f::<V>(7.0));
        assert_eq!(v.clone_as_vec(), fv::<V>(&[7.0, 7.0, 7.0]));
    }

    pub fn test_from_element<V: Vector>() {
        let v = V::from_element(2, f::<V>(5.0), Default::default());
        assert_eq!(v.clone_as_vec(), fv::<V>(&[5.0, 5.0]));
    }

    pub fn test_copy_from<V: Vector>() {
        let mut v1 = V::zeros(3, Default::default());
        let v2 = V::from_vec(fv::<V>(&[1.0, 2.0, 3.0]), Default::default());
        v1.copy_from(&v2);
        assert_eq!(v1.clone_as_vec(), fv::<V>(&[1.0, 2.0, 3.0]));
    }

    pub fn test_from_vec<V: Vector>() {
        let v = V::from_vec(fv::<V>(&[1.0, 2.0, 3.0]), Default::default());
        assert_eq!(v.clone_as_vec(), fv::<V>(&[1.0, 2.0, 3.0]));
    }

    pub fn test_component_mul_assign<V: Vector>() {
        let mut a = V::from_vec(fv::<V>(&[2.0, 3.0]), Default::default());
        let b = V::from_vec(fv::<V>(&[10.0, 20.0]), Default::default());
        a.component_mul_assign(&b);
        assert_eq!(a.clone_as_vec(), fv::<V>(&[20.0, 60.0]));
    }

    pub fn test_component_div_assign<V: Vector>() {
        let mut a = V::from_vec(fv::<V>(&[6.0, 12.0]), Default::default());
        let b = V::from_vec(fv::<V>(&[2.0, 3.0]), Default::default());
        a.component_div_assign(&b);
        assert_eq!(a.clone_as_vec(), fv::<V>(&[3.0, 4.0]));
    }

    pub fn test_assign_at_indices<V: Vector>() {
        let mut v = V::from_vec(fv::<V>(&[1.0, 2.0, 3.0]), Default::default());
        let indices = V::Index::from_vec(vec![0, 2], Default::default());
        v.assign_at_indices(&indices, f::<V>(0.0));
        assert_eq!(v.clone_as_vec(), fv::<V>(&[0.0, 2.0, 0.0]));
    }

    pub fn test_add<V: Vector>() {
        let a = V::from_vec(fv::<V>(&[1.0, 2.0, 3.0]), Default::default());
        let b = V::from_vec(fv::<V>(&[10.0, 20.0, 30.0]), Default::default());
        let c = a + b;
        assert_eq!(c.clone_as_vec(), fv::<V>(&[11.0, 22.0, 33.0]));
    }

    pub fn test_sub<V: Vector>() {
        let a = V::from_vec(fv::<V>(&[10.0, 20.0, 30.0]), Default::default());
        let b = V::from_vec(fv::<V>(&[1.0, 2.0, 3.0]), Default::default());
        let c = a - b;
        assert_eq!(c.clone_as_vec(), fv::<V>(&[9.0, 18.0, 27.0]));
    }

    pub fn test_add_assign<V: Vector>() {
        let mut a = V::from_vec(fv::<V>(&[1.0, 2.0, 3.0]), Default::default());
        let b = V::from_vec(fv::<V>(&[10.0, 20.0, 30.0]), Default::default());
        a += b;
        assert_eq!(a.clone_as_vec(), fv::<V>(&[11.0, 22.0, 33.0]));
    }

    pub fn test_sub_assign<V: Vector>() {
        let mut a = V::from_vec(fv::<V>(&[10.0, 20.0, 30.0]), Default::default());
        let b = V::from_vec(fv::<V>(&[1.0, 2.0, 3.0]), Default::default());
        a -= b;
        assert_eq!(a.clone_as_vec(), fv::<V>(&[9.0, 18.0, 27.0]));
    }

    pub fn test_axpy_v<V: Vector>() {
        let mut y = V::from_vec(fv::<V>(&[1.0, 2.0, 3.0]), Default::default());
        let x = V::from_vec(fv::<V>(&[4.0, 5.0, 6.0]), Default::default());
        let x_view = x.as_view();
        y.axpy_v(f::<V>(2.0), &x_view, f::<V>(1.0));
        assert_eq!(y.clone_as_vec(), fv::<V>(&[9.0, 12.0, 15.0]));
    }

    pub fn test_mul_assign_scalar<V: Vector>() {
        let mut v = V::from_vec(fv::<V>(&[1.0, 2.0, 3.0]), Default::default());
        v *= Scale(f::<V>(2.0));
        assert_eq!(v.clone_as_vec(), fv::<V>(&[2.0, 4.0, 6.0]));
    }

    pub fn test_copy_from_view<V: Vector>() {
        let mut v1 = V::zeros(3, Default::default());
        let v2 = V::from_vec(fv::<V>(&[1.0, 2.0, 3.0]), Default::default());
        let view = v2.as_view();
        v1.copy_from_view(&view);
        assert_eq!(v1.clone_as_vec(), fv::<V>(&[1.0, 2.0, 3.0]));
    }

    pub fn test_view_squared_norm<V: Vector>() {
        let x = V::from_vec(fv::<V>(&[1.0, 2.0]), Default::default());
        let y = V::from_vec(fv::<V>(&[1.0, 1.0]), Default::default());
        let atol = V::from_vec(fv::<V>(&[1e-3, 1e-3]), Default::default());
        let rtol = f::<V>(1e-2);
        let view = x.as_view();
        let norm = VectorView::squared_norm(&view, &y, &atol, rtol);
        let denom = f::<V>(1.0) * rtol + f::<V>(1e-3);
        let err0 = f::<V>(1.0) / denom;
        let err1 = f::<V>(2.0) / denom;
        let expected = (err0 * err0 + err1 * err1) / f::<V>(2.0);
        assert!(num_traits::abs(norm - expected) < f::<V>(1e-12));
    }

    pub fn test_view_into_owned<V: Vector>() {
        let v = V::from_vec(fv::<V>(&[1.0, 2.0, 3.0]), Default::default());
        let view = v.as_view();
        let owned = view.into_owned();
        assert_eq!(owned.clone_as_vec(), fv::<V>(&[1.0, 2.0, 3.0]));
    }

    pub fn test_view_get_index<V: Vector>() {
        let v = V::from_vec(fv::<V>(&[1.0, 2.0, 3.0]), Default::default());
        let view = v.as_view();
        assert_eq!(view.get_index(1), f::<V>(2.0));
        assert_eq!(view.get_index(2), f::<V>(3.0));
    }

    pub fn test_view_mut_axpy<V: Vector>() {
        let mut y = V::from_vec(fv::<V>(&[1.0, 2.0, 3.0]), Default::default());
        let x = V::from_vec(fv::<V>(&[4.0, 5.0, 6.0]), Default::default());
        {
            let mut y_view = y.as_view_mut();
            y_view.axpy(f::<V>(2.0), &x, f::<V>(1.0));
        }
        assert_eq!(y.clone_as_vec(), fv::<V>(&[9.0, 12.0, 15.0]));
    }

    pub fn test_view_mut_set_index<V: Vector>() {
        let mut v = V::zeros(3, Default::default());
        {
            let mut view = v.as_view_mut();
            view.set_index(1, f::<V>(42.0));
        }
        assert_eq!(v.clone_as_vec(), fv::<V>(&[0.0, 42.0, 0.0]));
    }

    pub fn test_view_mut_mul_assign_scalar<V: Vector>() {
        let mut v = V::from_vec(fv::<V>(&[1.0, 2.0, 3.0]), Default::default());
        {
            let mut view = v.as_view_mut();
            view *= Scale(f::<V>(2.0));
        }
        assert_eq!(v.clone_as_vec(), fv::<V>(&[2.0, 4.0, 6.0]));
    }

    pub fn test_view_mut_copy_from_view<V: Vector>() {
        let mut v1 = V::zeros(3, Default::default());
        let v2 = V::from_vec(fv::<V>(&[1.0, 2.0, 3.0]), Default::default());
        {
            let v2_view = v2.as_view();
            let mut v1_view = v1.as_view_mut();
            v1_view.copy_from_view(&v2_view);
        }
        assert_eq!(v1.clone_as_vec(), fv::<V>(&[1.0, 2.0, 3.0]));
    }

    pub fn test_batched_len_and_total_len<V: Vector>(ctx: V::C) {
        let nbatch = ctx.nbatch();
        assert!(nbatch > 1);
        let v = V::zeros(4, ctx);
        assert_eq!(v.len(), 4);
        assert_eq!(v.total_len(), 4 * nbatch);
    }

    pub fn test_batched_from_vec<V: Vector>(ctx: V::C) {
        assert_eq!(ctx.nbatch(), 2);
        let v = V::from_vec(fv::<V>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), ctx);
        assert_eq!(v.len(), 3);
        assert_eq!(v.total_len(), 6);
        assert_eq!(v.clone_as_vec(), fv::<V>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    }

    pub fn test_batched_from_vec_bad_length<V: Vector>(ctx: V::C) {
        assert_eq!(ctx.nbatch(), 2);
        let _v = V::from_vec(fv::<V>(&[1.0, 2.0, 3.0]), ctx);
    }

    pub fn test_batched_from_element<V: Vector>(ctx: V::C) {
        assert_eq!(ctx.nbatch(), 3);
        let v = V::from_element(2, f::<V>(5.0), ctx);
        assert_eq!(v.len(), 2);
        assert_eq!(v.clone_as_vec(), fv::<V>(&[5.0, 5.0, 5.0, 5.0, 5.0, 5.0]));
    }

    pub fn test_batched_axpy<V: Vector>(ctx: V::C) {
        assert_eq!(ctx.nbatch(), 2);
        let mut y = V::from_vec(fv::<V>(&[1.0, 2.0, 10.0, 20.0]), ctx.clone());
        let x = V::from_vec(fv::<V>(&[3.0, 4.0, 30.0, 40.0]), ctx);
        y.axpy(f::<V>(2.0), &x, f::<V>(1.0));
        assert_eq!(y.clone_as_vec(), fv::<V>(&[7.0, 10.0, 70.0, 100.0]));
    }

    pub fn test_batched_add<V: Vector>(ctx: V::C) {
        assert_eq!(ctx.nbatch(), 2);
        let a = V::from_vec(fv::<V>(&[1.0, 2.0, 3.0, 4.0]), ctx.clone());
        let b = V::from_vec(fv::<V>(&[10.0, 20.0, 30.0, 40.0]), ctx);
        let c = a + b;
        assert_eq!(c.len(), 2);
        assert_eq!(c.clone_as_vec(), fv::<V>(&[11.0, 22.0, 33.0, 44.0]));
    }

    pub fn test_batched_norm_max_across_batches<V: Vector>(ctx: V::C) {
        assert_eq!(ctx.nbatch(), 2);
        let v = V::from_vec(fv::<V>(&[1.0, 0.0, 0.0, 3.0]), ctx);
        let norm = v.norm(2);
        let diff = norm - f::<V>(3.0);
        assert!(num_traits::abs(diff) < f::<V>(1e-12));
    }

    pub fn test_batched_norm_l1<V: Vector>(ctx: V::C) {
        assert_eq!(ctx.nbatch(), 2);
        let v = V::from_vec(fv::<V>(&[1.0, -2.0, 3.0, 0.0]), ctx);
        let norm = v.norm(1);
        // batch0: |1|+|-2| = 3, batch1: |3|+|0| = 3, max = 3
        let diff = norm - f::<V>(3.0);
        assert!(num_traits::abs(diff) < f::<V>(1e-12));
    }

    pub fn test_batched_squared_norm<V: Vector>(ctx: V::C) {
        assert_eq!(ctx.nbatch(), 2);
        let x = V::from_vec(fv::<V>(&[1.0, 2.0, 3.0, 4.0]), ctx.clone());
        let y = V::from_vec(fv::<V>(&[1.0, 1.0, 1.0, 1.0]), ctx);
        let atol = V::from_vec(fv::<V>(&[1e-3, 1e-3]), V::C::default());
        let rtol = f::<V>(1e-2);
        let norm = x.squared_norm(&y, &atol, rtol);
        let denom = f::<V>(1.0) * rtol + f::<V>(1e-3);
        let err3 = f::<V>(3.0) / denom;
        let err4 = f::<V>(4.0) / denom;
        let batch1 = (err3 * err3 + err4 * err4) / f::<V>(2.0);
        let diff = norm - batch1;
        assert!(num_traits::abs(diff) < f::<V>(1e-12));
    }

    pub fn test_batched_set_index<V: Vector>(ctx: V::C) {
        assert_eq!(ctx.nbatch(), 3);
        let mut v = V::zeros(2, ctx);
        v.set_index(0, f::<V>(42.0));
        assert_eq!(
            v.clone_as_vec(),
            fv::<V>(&[42.0, 0.0, 42.0, 0.0, 42.0, 0.0])
        );
    }

    pub fn test_batched_get_index_panics<V: Vector>(ctx: V::C) {
        assert!(ctx.nbatch() > 1);
        let v = V::from_vec(fv::<V>(&[1.0, 2.0, 3.0, 4.0]), ctx);
        let _val = v.get_index(0);
    }

    pub fn test_batched_fill<V: Vector>(ctx: V::C) {
        assert_eq!(ctx.nbatch(), 2);
        let mut v = V::zeros(3, ctx);
        v.fill(f::<V>(7.0));
        assert_eq!(v.clone_as_vec(), fv::<V>(&[7.0, 7.0, 7.0, 7.0, 7.0, 7.0]));
    }

    pub fn test_batched_component_mul<V: Vector>(ctx: V::C) {
        assert_eq!(ctx.nbatch(), 2);
        let mut a = V::from_vec(fv::<V>(&[2.0, 3.0, 4.0, 5.0]), ctx.clone());
        let b = V::from_vec(fv::<V>(&[10.0, 20.0, 30.0, 40.0]), ctx);
        a.component_mul_assign(&b);
        assert_eq!(a.clone_as_vec(), fv::<V>(&[20.0, 60.0, 120.0, 200.0]));
    }

    pub fn test_batched_assign_at_indices<V: Vector>(ctx: V::C) {
        assert_eq!(ctx.nbatch(), 2);
        let mut v = V::from_vec(fv::<V>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), ctx);
        let indices = V::Index::from_vec(vec![0, 2], Default::default());
        v.assign_at_indices(&indices, f::<V>(0.0));
        assert_eq!(v.clone_as_vec(), fv::<V>(&[0.0, 2.0, 0.0, 0.0, 5.0, 0.0]));
    }

    pub fn test_batched_root_finding_consistent<V: Vector>(ctx: V::C) {
        assert_eq!(ctx.nbatch(), 2);
        let g0 = V::from_vec(fv::<V>(&[1.0, -1.0, 1.0, -1.0]), ctx.clone());
        let g1 = V::from_vec(fv::<V>(&[-1.0, 1.0, -1.0, 1.0]), ctx);
        let (found, _frac, idx) = g0.root_finding(&g1);
        assert!(!found);
        assert!(idx >= 0);
    }

    pub fn test_batched_root_finding_inconsistent<V: Vector>(ctx: V::C) {
        assert_eq!(ctx.nbatch(), 2);
        let g0 = V::from_vec(fv::<V>(&[1.0, 1.0, 1.0, -1.0]), ctx.clone());
        let g1 = V::from_vec(fv::<V>(&[-1.0, 1.0, 1.0, 1.0]), ctx);
        let _result = g0.root_finding(&g1);
    }

    // --- Broadcasting tests ---

    pub fn test_batched_axpy_broadcast<V: Vector>(ctx: V::C) {
        assert_eq!(ctx.nbatch(), 2);
        // self has nbatch=2, x has nbatch=1
        let mut y = V::from_vec(fv::<V>(&[1.0, 2.0, 10.0, 20.0]), ctx);
        let x = V::from_vec(fv::<V>(&[3.0, 4.0]), V::C::default());
        y.axpy(f::<V>(2.0), &x, f::<V>(1.0));
        // batch0: [1+6, 2+8]=[7,10], batch1: [10+6, 20+8]=[16,28]
        assert_eq!(y.clone_as_vec(), fv::<V>(&[7.0, 10.0, 16.0, 28.0]));
    }

    pub fn test_batched_copy_from_broadcast<V: Vector>(ctx: V::C) {
        assert_eq!(ctx.nbatch(), 2);
        let mut y = V::zeros(2, ctx);
        let x = V::from_vec(fv::<V>(&[5.0, 7.0]), V::C::default());
        y.copy_from(&x);
        // both batches get [5, 7]
        assert_eq!(y.clone_as_vec(), fv::<V>(&[5.0, 7.0, 5.0, 7.0]));
    }

    pub fn test_batched_component_div<V: Vector>(ctx: V::C) {
        assert_eq!(ctx.nbatch(), 2);
        let mut a = V::from_vec(fv::<V>(&[6.0, 8.0, 12.0, 20.0]), ctx.clone());
        let b = V::from_vec(fv::<V>(&[2.0, 4.0, 3.0, 5.0]), ctx);
        a.component_div_assign(&b);
        assert_eq!(a.clone_as_vec(), fv::<V>(&[3.0, 2.0, 4.0, 4.0]));
    }

    pub fn test_batched_component_mul_broadcast<V: Vector>(ctx: V::C) {
        assert_eq!(ctx.nbatch(), 2);
        let mut a = V::from_vec(fv::<V>(&[2.0, 3.0, 4.0, 5.0]), ctx);
        let b = V::from_vec(fv::<V>(&[10.0, 20.0]), V::C::default());
        a.component_mul_assign(&b);
        // batch0: [20, 60], batch1: [40, 100]
        assert_eq!(a.clone_as_vec(), fv::<V>(&[20.0, 60.0, 40.0, 100.0]));
    }

    pub fn test_batched_component_div_broadcast<V: Vector>(ctx: V::C) {
        assert_eq!(ctx.nbatch(), 2);
        let mut a = V::from_vec(fv::<V>(&[6.0, 8.0, 12.0, 20.0]), ctx);
        let b = V::from_vec(fv::<V>(&[2.0, 4.0]), V::C::default());
        a.component_div_assign(&b);
        // batch0: [3, 2], batch1: [6, 5]
        assert_eq!(a.clone_as_vec(), fv::<V>(&[3.0, 2.0, 6.0, 5.0]));
    }

    pub fn test_batched_add_assign_broadcast<V: Vector>(ctx: V::C) {
        assert_eq!(ctx.nbatch(), 2);
        let mut a = V::from_vec(fv::<V>(&[1.0, 2.0, 3.0, 4.0]), ctx);
        let b = V::from_vec(fv::<V>(&[10.0, 20.0]), V::C::default());
        a += &b;
        assert_eq!(a.clone_as_vec(), fv::<V>(&[11.0, 22.0, 13.0, 24.0]));
    }

    pub fn test_batched_sub_assign_broadcast<V: Vector>(ctx: V::C) {
        assert_eq!(ctx.nbatch(), 2);
        let mut a = V::from_vec(fv::<V>(&[10.0, 20.0, 30.0, 40.0]), ctx);
        let b = V::from_vec(fv::<V>(&[1.0, 2.0]), V::C::default());
        a -= &b;
        assert_eq!(a.clone_as_vec(), fv::<V>(&[9.0, 18.0, 29.0, 38.0]));
    }

    pub fn test_batched_sub<V: Vector>(ctx: V::C) {
        assert_eq!(ctx.nbatch(), 2);
        let a = V::from_vec(fv::<V>(&[10.0, 20.0, 30.0, 40.0]), ctx.clone());
        let b = V::from_vec(fv::<V>(&[1.0, 2.0, 3.0, 4.0]), ctx);
        let c = a - b;
        assert_eq!(c.clone_as_vec(), fv::<V>(&[9.0, 18.0, 27.0, 36.0]));
    }

    pub fn test_batched_sub_assign<V: Vector>(ctx: V::C) {
        assert_eq!(ctx.nbatch(), 2);
        let mut a = V::from_vec(fv::<V>(&[10.0, 20.0, 30.0, 40.0]), ctx.clone());
        let b = V::from_vec(fv::<V>(&[1.0, 2.0, 3.0, 4.0]), ctx);
        a -= b;
        assert_eq!(a.clone_as_vec(), fv::<V>(&[9.0, 18.0, 27.0, 36.0]));
    }

    pub fn test_batched_from_slice<V: Vector>(ctx: V::C) {
        assert_eq!(ctx.nbatch(), 2);
        let slice = fv::<V>(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        let v = V::from_slice(&slice, ctx);
        assert_eq!(v.clone_as_vec(), fv::<V>(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]));
    }

    pub fn test_batched_mul_scalar<V: Vector>(ctx: V::C) {
        assert_eq!(ctx.nbatch(), 2);
        let v = V::from_vec(fv::<V>(&[1.0, 2.0, 10.0, 20.0]), ctx);
        let result = v * Scale(f::<V>(2.0));
        assert_eq!(result.clone_as_vec(), fv::<V>(&[2.0, 4.0, 20.0, 40.0]));
    }

    pub fn test_batched_div_scalar<V: Vector>(ctx: V::C) {
        assert_eq!(ctx.nbatch(), 2);
        let v = V::from_vec(fv::<V>(&[2.0, 4.0, 20.0, 40.0]), ctx);
        let result = v / Scale(f::<V>(2.0));
        assert_eq!(result.clone_as_vec(), fv::<V>(&[1.0, 2.0, 10.0, 20.0]));
    }

    pub fn test_batched_copy_from_indices<V: Vector>(ctx: V::C) {
        assert_eq!(ctx.nbatch(), 2);
        let mut v1 = V::zeros(4, ctx.clone());
        let v2 = V::from_vec(
            fv::<V>(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]),
            ctx,
        );
        let indices = V::Index::from_vec(vec![0, 2, 3], Default::default());
        v1.copy_from_indices(&v2, &indices);
        assert_eq!(
            v1.clone_as_vec(),
            fv::<V>(&[10.0, 0.0, 30.0, 40.0, 50.0, 0.0, 70.0, 80.0])
        );
    }

    pub fn test_batched_gather<V: Vector>(ctx: V::C) {
        assert_eq!(ctx.nbatch(), 2);
        let mut result = V::zeros(3, ctx.clone());
        let v = V::from_vec(
            fv::<V>(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]),
            ctx,
        );
        let indices = V::Index::from_vec(vec![3, 0, 2], Default::default());
        result.gather(&v, &indices);
        assert_eq!(
            result.clone_as_vec(),
            fv::<V>(&[40.0, 10.0, 30.0, 80.0, 50.0, 70.0])
        );
    }

    pub fn test_batched_scatter<V: Vector>(ctx: V::C) {
        assert_eq!(ctx.nbatch(), 2);
        let v = V::from_vec(fv::<V>(&[40.0, 10.0, 30.0, 80.0, 50.0, 70.0]), ctx.clone());
        let indices = V::Index::from_vec(vec![3, 0, 2], Default::default());
        let mut result = V::zeros(4, ctx);
        v.scatter(&indices, &mut result);
        assert_eq!(
            result.clone_as_vec(),
            fv::<V>(&[10.0, 0.0, 30.0, 40.0, 50.0, 0.0, 70.0, 80.0])
        );
    }

    pub fn test_batched_get_batch<V: Vector>(ctx: V::C) {
        assert_eq!(ctx.nbatch(), 2);
        let v = V::from_vec(fv::<V>(&[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]), ctx);
        let batch0 = v.get_batch(0);
        assert_eq!(batch0.get_index(0), f::<V>(1.0));
        assert_eq!(batch0.get_index(1), f::<V>(2.0));
        assert_eq!(batch0.get_index(2), f::<V>(3.0));
        let batch1 = v.get_batch(1);
        assert_eq!(batch1.get_index(0), f::<V>(10.0));
        assert_eq!(batch1.get_index(1), f::<V>(20.0));
        assert_eq!(batch1.get_index(2), f::<V>(30.0));
    }

    pub fn test_batched_get_batch_mut<V: Vector>(ctx: V::C) {
        assert_eq!(ctx.nbatch(), 2);
        let mut v = V::from_vec(fv::<V>(&[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]), ctx);
        {
            let mut batch0 = v.get_batch_mut(0);
            batch0.set_index(1, f::<V>(99.0));
        }
        assert_eq!(
            v.clone_as_vec(),
            fv::<V>(&[1.0, 99.0, 3.0, 10.0, 20.0, 30.0])
        );
    }

    pub fn test_batched_axpy_v<V: Vector>(ctx: V::C) {
        assert_eq!(ctx.nbatch(), 2);
        let mut y = V::from_vec(fv::<V>(&[1.0, 2.0, 10.0, 20.0]), ctx.clone());
        let x = V::from_vec(fv::<V>(&[3.0, 4.0, 30.0, 40.0]), ctx);
        let x_view = x.as_view();
        y.axpy_v(f::<V>(2.0), &x_view, f::<V>(1.0));
        assert_eq!(y.clone_as_vec(), fv::<V>(&[7.0, 10.0, 70.0, 100.0]));
    }

    pub fn test_batched_mul_assign_scalar<V: Vector>(ctx: V::C) {
        assert_eq!(ctx.nbatch(), 2);
        let mut v = V::from_vec(fv::<V>(&[1.0, 2.0, 10.0, 20.0]), ctx);
        v *= Scale(f::<V>(2.0));
        assert_eq!(v.clone_as_vec(), fv::<V>(&[2.0, 4.0, 20.0, 40.0]));
    }

    pub fn test_batched_copy_from_view<V: Vector>(ctx: V::C) {
        assert_eq!(ctx.nbatch(), 2);
        let mut v1 = V::zeros(2, ctx);
        let v2 = V::from_vec(fv::<V>(&[5.0, 7.0]), V::C::default());
        let view = v2.as_view();
        v1.copy_from_view(&view);
        assert_eq!(v1.clone_as_vec(), fv::<V>(&[5.0, 7.0, 5.0, 7.0]));
    }

    // --- batched_axpy tests ---

    pub fn test_batched_axpy_new<V: Vector>(ctx: V::C) {
        assert_eq!(ctx.nbatch(), 2);
        let mut y = V::from_vec(fv::<V>(&[1.0, 2.0, 10.0, 20.0]), ctx.clone());
        let x = V::from_vec(fv::<V>(&[3.0, 4.0, 30.0, 40.0]), ctx);
        y.batched_axpy(&[f::<V>(2.0), f::<V>(0.5)], &x, f::<V>(1.0));
        // batch0: [1,2] + 2*[3,4] = [7,10]
        // batch1: [10,20] + 0.5*[30,40] = [25,40]
        assert_eq!(y.clone_as_vec(), fv::<V>(&[7.0, 10.0, 25.0, 40.0]));
    }

    pub fn test_batched_axpy_new_broadcast<V: Vector>(ctx: V::C) {
        assert_eq!(ctx.nbatch(), 2);
        let mut y = V::from_vec(fv::<V>(&[1.0, 2.0, 10.0, 20.0]), ctx);
        let x = V::from_vec(fv::<V>(&[3.0, 4.0]), V::C::default());
        y.batched_axpy(&[f::<V>(2.0), f::<V>(0.5)], &x, f::<V>(1.0));
        // both batches: beta*y + alpha_b * x
        // batch0: [1,2] + 2*[3,4] = [7,10]
        // batch1: [10,20] + 0.5*[3,4] = [11.5,22]
        assert_eq!(y.clone_as_vec(), fv::<V>(&[7.0, 10.0, 11.5, 22.0]));
    }

    #[allow(clippy::type_complexity)]
    pub fn test_batched_axpy_new_bad_length<V: Vector>(ctx: V::C) {
        assert_eq!(ctx.nbatch(), 2);
        let mut y = V::zeros(2, ctx);
        let x = V::zeros(2, V::C::default());
        y.batched_axpy(&[f::<V>(1.0)], &x, f::<V>(0.0));
    }

    // --- Incompatible batch tests ---

    pub fn test_batched_axpy_incompatible<V: Vector>(ctx2: V::C, ctx3: V::C) {
        assert_eq!(ctx2.nbatch(), 2);
        assert_eq!(ctx3.nbatch(), 3);
        let mut y = V::zeros(2, ctx2);
        let x = V::zeros(2, ctx3);
        y.axpy(f::<V>(1.0), &x, f::<V>(1.0));
    }

    pub fn test_batched_copy_from_incompatible<V: Vector>(ctx2: V::C, ctx3: V::C) {
        assert_eq!(ctx2.nbatch(), 2);
        assert_eq!(ctx3.nbatch(), 3);
        let mut y = V::zeros(2, ctx2);
        let x = V::zeros(2, ctx3);
        y.copy_from(&x);
    }

    pub fn test_batched_add_assign_incompatible<V: Vector>(ctx2: V::C, ctx3: V::C) {
        assert_eq!(ctx2.nbatch(), 2);
        assert_eq!(ctx3.nbatch(), 3);
        let mut a = V::zeros(2, ctx2);
        let b = V::zeros(2, ctx3);
        a += &b;
    }

    pub fn test_batched_component_mul_incompatible<V: Vector>(ctx2: V::C, ctx3: V::C) {
        assert_eq!(ctx2.nbatch(), 2);
        assert_eq!(ctx3.nbatch(), 3);
        let mut a = V::zeros(2, ctx2);
        let b = V::zeros(2, ctx3);
        a.component_mul_assign(&b);
    }

    #[test]
    fn vector_common_for_references_and_default_helpers_work() {
        let mut v = NalgebraVec::from_vec(vec![1.0, 2.0], NalgebraContext::default());
        assert_eq!(<NalgebraVec<f64> as VectorCommon>::inner(&v).len(), 2);
        assert_eq!(<&NalgebraVec<f64> as VectorCommon>::inner(&&v).len(), 2);
        assert_eq!(
            <&mut NalgebraVec<f64> as VectorCommon>::inner(&&mut v).len(),
            2
        );

        let empty = NalgebraVec::<f64>::zeros(0, NalgebraContext::default());
        assert!(empty.is_empty());

        let non_empty = NalgebraVec::<f64>::zeros(2, NalgebraContext::default());
        assert!(!non_empty.is_empty());
        assert_eq!(non_empty.clone_as_vec(), vec![0.0, 0.0]);
    }

    #[test]
    fn vector_assert_eq_panics_for_length_mismatch() {
        let left = NalgebraVec::from_vec(vec![1.0, 2.0], NalgebraContext::default());
        let right = NalgebraVec::from_vec(vec![1.0], NalgebraContext::default());
        let tol = NalgebraVec::from_vec(vec![0.0, 0.0], NalgebraContext::default());
        assert!(catch_unwind(AssertUnwindSafe(|| left.assert_eq(&right, &tol))).is_err());
    }

    #[test]
    fn vector_assert_helpers_cover_success_and_failure_paths() {
        let left = NalgebraVec::from_vec(vec![1.0, 2.0, 3.0], NalgebraContext::default());
        let right = NalgebraVec::from_vec(vec![1.0, 2.0, 3.0], NalgebraContext::default());
        let tol = NalgebraVec::from_vec(vec![0.0, 0.0, 0.0], NalgebraContext::default());
        left.assert_eq(&right, &tol);
        left.assert_eq_st(&right, 0.0);
        left.assert_eq_norm(&right, &tol, 1e-6, 1.0);

        let perturbed = NalgebraVec::from_vec(vec![1.1, 2.0, 3.0], NalgebraContext::default());
        assert!(catch_unwind(AssertUnwindSafe(
            || left.assert_eq_norm(&perturbed, &tol, 1e-6, 0.01)
        ))
        .is_err());
    }

    #[test]
    fn vector_assert_eq_vec_panics_for_short_vector_mismatch() {
        assert!(catch_unwind(AssertUnwindSafe(|| {
            <NalgebraVec<f64> as Vector>::assert_eq_vec(
                vec![1.0, 2.0, 3.0],
                vec![0.0, 2.0, 3.0],
                vec![0.0, 0.0, 0.0],
            )
        }))
        .is_err());
    }

    #[test]
    fn vector_assert_eq_vec_panics_for_first_middle_and_last_mismatch_in_long_vectors() {
        assert!(catch_unwind(AssertUnwindSafe(|| {
            <NalgebraVec<f64> as Vector>::assert_eq_vec(
                vec![1.0, 2.0, 3.0, 4.0],
                vec![0.0, 2.0, 3.0, 4.0],
                vec![0.0, 0.0, 0.0, 0.0],
            )
        }))
        .is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| {
            <NalgebraVec<f64> as Vector>::assert_eq_vec(
                vec![1.0, 2.0, 3.0, 4.0, 5.0],
                vec![1.0, 2.0, 0.0, 4.0, 5.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
            )
        }))
        .is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| {
            <NalgebraVec<f64> as Vector>::assert_eq_vec(
                vec![1.0, 2.0, 3.0, 4.0],
                vec![1.0, 2.0, 3.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0],
            )
        }))
        .is_err());
    }

    use crate::matrix::DenseMatrix;

    fn make_strided_test_matrix<M: DenseMatrix>(nbatch: usize) -> M {
        let ctx = M::C::default().clone_with_nbatch(nbatch).unwrap();
        let nrows = 3;
        let ncols = 4;
        let mut data = Vec::with_capacity(nrows * ncols * nbatch);
        for b in 0..nbatch {
            for col in 0..ncols {
                for row in 0..nrows {
                    data.push(f::<M::V>(row as f64 + col as f64 * 10.0 + b as f64 * 100.0));
                }
            }
        }
        M::from_vec(nrows, ncols, data, ctx)
    }

    pub fn test_strided_view_set_index<M: DenseMatrix>(ctx: M::C) {
        let mut matrix = make_strided_test_matrix::<M>(ctx.nbatch());
        {
            let mut col1 = matrix.column_mut(1);
            col1.set_index(1, f::<M::V>(99.0));
        }
        let owned = matrix.column(1).into_owned();
        let b0 = owned.get_batch(0);
        let b1 = owned.get_batch(1);
        assert_eq!(b0.get_index(1), f::<M::V>(99.0));
        assert_eq!(b1.get_index(1), f::<M::V>(99.0));
    }

    pub fn test_strided_view_mut_copy_from<M: DenseMatrix>(ctx: M::C) {
        let mut matrix = make_strided_test_matrix::<M>(ctx.nbatch());
        let owned = M::V::from_vec(
            vec![f::<M::V>(50.0), f::<M::V>(51.0), f::<M::V>(52.0)],
            M::C::default(),
        );
        {
            let mut col1 = matrix.column_mut(1);
            col1.copy_from(&owned);
        }
        let owned_v = matrix.column(1).into_owned();
        let b0 = owned_v.get_batch(0);
        let b1 = owned_v.get_batch(1);
        assert_eq!(b0.get_index(0), f::<M::V>(50.0));
        assert_eq!(b0.get_index(1), f::<M::V>(51.0));
        assert_eq!(b0.get_index(2), f::<M::V>(52.0));
        assert_eq!(b1.get_index(0), f::<M::V>(50.0));
        assert_eq!(b1.get_index(1), f::<M::V>(51.0));
        assert_eq!(b1.get_index(2), f::<M::V>(52.0));
    }

    pub fn test_strided_view_mut_axpy<M: DenseMatrix>(ctx: M::C) {
        let mut matrix = make_strided_test_matrix::<M>(ctx.nbatch());
        let x = M::V::from_vec(
            vec![f::<M::V>(10.0), f::<M::V>(10.0), f::<M::V>(10.0)],
            M::C::default(),
        );
        {
            let mut col1 = matrix.column_mut(1);
            col1.axpy(f::<M::V>(2.0), &x, f::<M::V>(1.0));
        }
        let owned_v = matrix.column(1).into_owned();
        let b1 = owned_v.get_batch(1);
        assert_eq!(b1.get_index(0), f::<M::V>(130.0));
        assert_eq!(b1.get_index(1), f::<M::V>(131.0));
        assert_eq!(b1.get_index(2), f::<M::V>(132.0));
    }

    pub fn test_strided_view_mut_mul_assign_scalar<M: DenseMatrix>(ctx: M::C) {
        let mut matrix = make_strided_test_matrix::<M>(ctx.nbatch());
        {
            let mut col1 = matrix.column_mut(1);
            col1 *= Scale(f::<M::V>(2.0));
        }
        let owned_v = matrix.column(1).into_owned();
        let b0 = owned_v.get_batch(0);
        let b1 = owned_v.get_batch(1);
        assert_eq!(b0.get_index(0), f::<M::V>(20.0));
        assert_eq!(b1.get_index(0), f::<M::V>(220.0));
    }

    pub fn test_strided_view_mut_add_assign<M: DenseMatrix>(ctx: M::C) {
        let mut matrix = make_strided_test_matrix::<M>(ctx.nbatch());
        let rhs = M::V::from_vec(
            vec![
                f::<M::V>(5.0),
                f::<M::V>(5.0),
                f::<M::V>(5.0),
                f::<M::V>(10.0),
                f::<M::V>(10.0),
                f::<M::V>(10.0),
            ],
            ctx.clone(),
        );
        {
            let mut col1 = matrix.column_mut(1);
            col1 += &rhs;
        }
        let owned_v = matrix.column(1).into_owned();
        let b0 = owned_v.get_batch(0);
        let b1 = owned_v.get_batch(1);
        assert_eq!(b0.get_index(0), f::<M::V>(15.0));
        assert_eq!(b0.get_index(1), f::<M::V>(16.0));
        assert_eq!(b0.get_index(2), f::<M::V>(17.0));
        assert_eq!(b1.get_index(0), f::<M::V>(120.0));
        assert_eq!(b1.get_index(1), f::<M::V>(121.0));
        assert_eq!(b1.get_index(2), f::<M::V>(122.0));
    }

    pub fn test_strided_view_mut_sub_assign<M: DenseMatrix>(ctx: M::C) {
        let mut matrix = make_strided_test_matrix::<M>(ctx.nbatch());
        let rhs = M::V::from_vec(
            vec![
                f::<M::V>(1.0),
                f::<M::V>(1.0),
                f::<M::V>(1.0),
                f::<M::V>(1.0),
                f::<M::V>(1.0),
                f::<M::V>(1.0),
            ],
            ctx.clone(),
        );
        {
            let mut col1 = matrix.column_mut(1);
            col1 -= &rhs;
        }
        let owned_v = matrix.column(1).into_owned();
        let b0 = owned_v.get_batch(0);
        let b1 = owned_v.get_batch(1);
        assert_eq!(b0.get_index(0), f::<M::V>(9.0));
        assert_eq!(b1.get_index(0), f::<M::V>(109.0));
    }

    pub fn test_strided_view_add_assign_broadcast<M: DenseMatrix>(ctx: M::C) {
        let mut matrix = make_strided_test_matrix::<M>(ctx.nbatch());
        let rhs = M::V::from_vec(
            vec![f::<M::V>(5.0), f::<M::V>(5.0), f::<M::V>(5.0)],
            M::C::default(),
        );
        {
            let mut col1 = matrix.column_mut(1);
            col1 += &rhs;
        }
        let owned_v = matrix.column(1).into_owned();
        let b0 = owned_v.get_batch(0);
        let b1 = owned_v.get_batch(1);
        assert_eq!(b0.get_index(0), f::<M::V>(15.0));
        assert_eq!(b1.get_index(0), f::<M::V>(115.0));
    }

    pub fn test_strided_view_add_owned<M: DenseMatrix>(ctx: M::C) {
        let matrix = make_strided_test_matrix::<M>(ctx.nbatch());
        let nbatch = ctx.nbatch();
        let mut rhs_data = Vec::with_capacity(3 * nbatch);
        for _ in 0..nbatch {
            rhs_data.extend_from_slice(&[f::<M::V>(5.0), f::<M::V>(5.0), f::<M::V>(5.0)]);
        }
        let rhs = M::V::from_vec(rhs_data, ctx.clone());
        let col1 = matrix.column(1);
        let result = col1 + &rhs;
        let b0 = result.get_batch(0);
        let b1 = result.get_batch(1);
        assert_eq!(b0.get_index(0), f::<M::V>(15.0));
        assert_eq!(b1.get_index(0), f::<M::V>(115.0));
    }

    pub fn test_strided_view_squared_norm<M: DenseMatrix>(ctx: M::C) {
        let matrix = make_strided_test_matrix::<M>(ctx.nbatch());
        let nbatch = ctx.nbatch();
        let mut y_data = Vec::with_capacity(3 * nbatch);
        for _ in 0..nbatch {
            y_data.extend_from_slice(&[f::<M::V>(1.0), f::<M::V>(1.0), f::<M::V>(1.0)]);
        }
        let y = M::V::from_vec(y_data, ctx.clone());
        let atol = M::V::from_vec(
            vec![f::<M::V>(1e-3), f::<M::V>(1e-3), f::<M::V>(1e-3)],
            M::C::default(),
        );
        let col1 = matrix.column(1);
        let norm = col1.squared_norm(&y, &atol, f::<M::V>(1e-2));
        assert!(norm > f::<M::V>(0.0));
    }

    pub fn test_strided_view_into_owned<M: DenseMatrix>(ctx: M::C) {
        let matrix = make_strided_test_matrix::<M>(ctx.nbatch());
        let col1 = matrix.column(1);
        let owned = col1.into_owned();
        let b0 = owned.get_batch(0);
        let b1 = owned.get_batch(1);
        assert_eq!(b0.get_index(0), f::<M::V>(10.0));
        assert_eq!(b0.get_index(1), f::<M::V>(11.0));
        assert_eq!(b1.get_index(0), f::<M::V>(110.0));
        assert_eq!(b1.get_index(1), f::<M::V>(111.0));
    }

    pub fn test_strided_view_component_mul<M: DenseMatrix>(ctx: M::C) {
        let matrix = make_strided_test_matrix::<M>(ctx.nbatch());
        let rhs = M::V::from_vec(
            vec![f::<M::V>(10.0), f::<M::V>(1.0), f::<M::V>(0.0)],
            M::C::default(),
        );
        let col1 = matrix.column(1);
        let mut owned = col1.into_owned();
        owned.component_mul_assign(&rhs);
        let b0 = owned.get_batch(0);
        let b1 = owned.get_batch(1);
        assert_eq!(b0.get_index(0), f::<M::V>(100.0));
        assert_eq!(b0.get_index(1), f::<M::V>(11.0));
        assert_eq!(b1.get_index(0), f::<M::V>(1100.0));
        assert_eq!(b1.get_index(1), f::<M::V>(111.0));
    }

    pub fn test_strided_view_component_div<M: DenseMatrix>(ctx: M::C) {
        let matrix = make_strided_test_matrix::<M>(ctx.nbatch());
        let rhs = M::V::from_vec(
            vec![f::<M::V>(10.0), f::<M::V>(11.0), f::<M::V>(12.0)],
            M::C::default(),
        );
        let col1 = matrix.column(1);
        let mut owned = col1.into_owned();
        owned.component_div_assign(&rhs);
        let b0 = owned.get_batch(0);
        let b1 = owned.get_batch(1);
        assert_eq!(b0.get_index(0), f::<M::V>(1.0));
        assert_eq!(b1.get_index(0), f::<M::V>(11.0));
    }

    pub fn test_strided_view_mul_scalar<M: DenseMatrix>(ctx: M::C) {
        let matrix = make_strided_test_matrix::<M>(ctx.nbatch());
        let col1 = matrix.column(1);
        let result = col1 * Scale(f::<M::V>(2.0));
        let b0 = result.get_batch(0);
        let b1 = result.get_batch(1);
        assert_eq!(b0.get_index(0), f::<M::V>(20.0));
        assert_eq!(b1.get_index(0), f::<M::V>(220.0));
    }

    pub fn test_strided_view_fill<M: DenseMatrix>(ctx: M::C) {
        let matrix = make_strided_test_matrix::<M>(ctx.nbatch());
        let col1 = matrix.column(1);
        let mut owned = col1.into_owned();
        owned.fill(f::<M::V>(7.0));
        let b0 = owned.get_batch(0);
        let b1 = owned.get_batch(1);
        assert_eq!(b0.get_index(0), f::<M::V>(7.0));
        assert_eq!(b1.get_index(0), f::<M::V>(7.0));
    }

    pub fn test_strided_view_assign_at_indices<M: DenseMatrix>(ctx: M::C) {
        let matrix = make_strided_test_matrix::<M>(ctx.nbatch());
        let indices = <M::V as Vector>::Index::from_vec(vec![0, 2], M::C::default());
        let col1 = matrix.column(1);
        let mut owned = col1.into_owned();
        owned.assign_at_indices(&indices, f::<M::V>(0.0));
        let b0 = owned.get_batch(0);
        let b1 = owned.get_batch(1);
        assert_eq!(b0.get_index(0), f::<M::V>(0.0));
        assert_eq!(b1.get_index(0), f::<M::V>(0.0));
    }

    pub fn test_strided_view_copy_from_indices<M: DenseMatrix>(ctx: M::C) {
        let matrix = make_strided_test_matrix::<M>(ctx.nbatch());
        let nbatch = ctx.nbatch();
        let mut other_data = Vec::with_capacity(3 * nbatch);
        for _ in 0..nbatch {
            other_data.extend_from_slice(&[f::<M::V>(50.0), f::<M::V>(0.0), f::<M::V>(0.0)]);
        }
        let other = M::V::from_vec(other_data, ctx.clone());
        let indices = <M::V as Vector>::Index::from_vec(vec![0], M::C::default());
        let col1 = matrix.column(1);
        let mut owned = col1.into_owned();
        owned.copy_from_indices(&other, &indices);
        let b0 = owned.get_batch(0);
        let b1 = owned.get_batch(1);
        assert_eq!(b0.get_index(0), f::<M::V>(50.0));
        assert_eq!(b1.get_index(0), f::<M::V>(50.0));
    }

    pub fn test_strided_view_gather<M: DenseMatrix>(ctx: M::C) {
        let matrix = make_strided_test_matrix::<M>(ctx.nbatch());
        let nbatch = ctx.nbatch();
        let mut result = M::V::zeros(2, M::C::default().clone_with_nbatch(nbatch).unwrap());
        let indices = <M::V as Vector>::Index::from_vec(vec![0, 2], M::C::default());
        let col1 = matrix.column(1);
        let owned = col1.into_owned();
        result.gather(&owned, &indices);
        let b1 = result.get_batch(1);
        assert_eq!(b1.get_index(0), f::<M::V>(110.0));
    }

    pub fn test_strided_view_scatter<M: DenseMatrix>(ctx: M::C) {
        let matrix = make_strided_test_matrix::<M>(ctx.nbatch());
        let nbatch = ctx.nbatch();
        let col1 = matrix.column(1);
        let owned = col1.into_owned();
        let indices = <M::V as Vector>::Index::from_vec(vec![0, 1, 2], M::C::default());
        let mut result = M::V::zeros(3, M::C::default().clone_with_nbatch(nbatch).unwrap());
        owned.scatter(&indices, &mut result);
        let b1 = result.get_batch(1);
        assert_eq!(b1.get_index(0), f::<M::V>(110.0));
    }
}

#[cfg(test)]
macro_rules! generate_vector_tests_nonbatched {
    ($suffix:ident, $V:ty) => {
        paste::paste! {
            #[test]
            fn [<test_root_finding_ $suffix>]() {
                $crate::vector::tests::test_root_finding::<$V>();
            }
            #[test]
            fn [<test_from_slice_ $suffix>]() {
                $crate::vector::tests::test_from_slice::<$V>();
            }
            #[test]
            fn [<test_mul_scalar_ $suffix>]() {
                $crate::vector::tests::test_mul_scalar::<$V>();
            }
            #[test]
            fn [<test_div_scalar_ $suffix>]() {
                $crate::vector::tests::test_div_scalar::<$V>();
            }
            #[test]
            fn [<test_axpy_ $suffix>]() {
                $crate::vector::tests::test_axpy::<$V>();
            }
            #[test]
            fn [<test_copy_from_indices_ $suffix>]() {
                $crate::vector::tests::test_copy_from_indices::<$V>();
            }
            #[test]
            fn [<test_gather_ $suffix>]() {
                $crate::vector::tests::test_gather::<$V>();
            }
            #[test]
            fn [<test_scatter_ $suffix>]() {
                $crate::vector::tests::test_scatter::<$V>();
            }
            #[test]
            fn [<test_copy_from_via_view_mut_ $suffix>]() {
                $crate::vector::tests::test_copy_from_via_view_mut::<$V>();
            }
            #[test]
            fn [<test_set_index_ $suffix>]() {
                $crate::vector::tests::test_set_index::<$V>();
            }
            #[test]
            fn [<test_get_index_ $suffix>]() {
                $crate::vector::tests::test_get_index::<$V>();
            }
            #[test]
            fn [<test_norm_ $suffix>]() {
                $crate::vector::tests::test_norm::<$V>();
            }
            #[test]
            fn [<test_norm_l1_ $suffix>]() {
                $crate::vector::tests::test_norm_l1::<$V>();
            }
            #[test]
            fn [<test_squared_norm_ $suffix>]() {
                $crate::vector::tests::test_squared_norm::<$V>();
            }
            #[test]
            fn [<test_fill_ $suffix>]() {
                $crate::vector::tests::test_fill::<$V>();
            }
            #[test]
            fn [<test_from_element_ $suffix>]() {
                $crate::vector::tests::test_from_element::<$V>();
            }
            #[test]
            fn [<test_copy_from_ $suffix>]() {
                $crate::vector::tests::test_copy_from::<$V>();
            }
            #[test]
            fn [<test_from_vec_ $suffix>]() {
                $crate::vector::tests::test_from_vec::<$V>();
            }
            #[test]
            fn [<test_component_mul_assign_ $suffix>]() {
                $crate::vector::tests::test_component_mul_assign::<$V>();
            }
            #[test]
            fn [<test_component_div_assign_ $suffix>]() {
                $crate::vector::tests::test_component_div_assign::<$V>();
            }
            #[test]
            fn [<test_assign_at_indices_ $suffix>]() {
                $crate::vector::tests::test_assign_at_indices::<$V>();
            }
            #[test]
            fn [<test_add_ $suffix>]() {
                $crate::vector::tests::test_add::<$V>();
            }
            #[test]
            fn [<test_sub_ $suffix>]() {
                $crate::vector::tests::test_sub::<$V>();
            }
            #[test]
            fn [<test_add_assign_ $suffix>]() {
                $crate::vector::tests::test_add_assign::<$V>();
            }
            #[test]
            fn [<test_sub_assign_ $suffix>]() {
                $crate::vector::tests::test_sub_assign::<$V>();
            }
            #[test]
            fn [<test_axpy_v_ $suffix>]() {
                $crate::vector::tests::test_axpy_v::<$V>();
            }
            #[test]
            fn [<test_mul_assign_scalar_ $suffix>]() {
                $crate::vector::tests::test_mul_assign_scalar::<$V>();
            }
            #[test]
            fn [<test_copy_from_view_ $suffix>]() {
                $crate::vector::tests::test_copy_from_view::<$V>();
            }
            #[test]
            fn [<test_view_squared_norm_ $suffix>]() {
                $crate::vector::tests::test_view_squared_norm::<$V>();
            }
            #[test]
            fn [<test_view_into_owned_ $suffix>]() {
                $crate::vector::tests::test_view_into_owned::<$V>();
            }
            #[test]
            fn [<test_view_get_index_ $suffix>]() {
                $crate::vector::tests::test_view_get_index::<$V>();
            }
            #[test]
            fn [<test_view_mut_axpy_ $suffix>]() {
                $crate::vector::tests::test_view_mut_axpy::<$V>();
            }
            #[test]
            fn [<test_view_mut_set_index_ $suffix>]() {
                $crate::vector::tests::test_view_mut_set_index::<$V>();
            }
            #[test]
            fn [<test_view_mut_mul_assign_scalar_ $suffix>]() {
                $crate::vector::tests::test_view_mut_mul_assign_scalar::<$V>();
            }
            #[test]
            fn [<test_view_mut_copy_from_view_ $suffix>]() {
                $crate::vector::tests::test_view_mut_copy_from_view::<$V>();
            }
        }
    };
}

#[cfg(test)]
macro_rules! generate_vector_tests_batched {
    ($suffix:ident, $V:ty, $ctx2:expr, $ctx3:expr) => {
        paste::paste! {
            #[test]
            fn [<test_batched_len_and_total_len_ $suffix>]() {
                $crate::vector::tests::test_batched_len_and_total_len::<$V>($ctx3);
            }
            #[test]
            fn [<test_batched_from_vec_ $suffix>]() {
                $crate::vector::tests::test_batched_from_vec::<$V>($ctx2);
            }
            #[test]
            #[should_panic(expected = "divisible by nbatch")]
            fn [<test_batched_from_vec_bad_length_ $suffix>]() {
                $crate::vector::tests::test_batched_from_vec_bad_length::<$V>($ctx2);
            }
            #[test]
            fn [<test_batched_from_element_ $suffix>]() {
                $crate::vector::tests::test_batched_from_element::<$V>($ctx3);
            }
            #[test]
            fn [<test_batched_axpy_ $suffix>]() {
                $crate::vector::tests::test_batched_axpy::<$V>($ctx2);
            }
            #[test]
            fn [<test_batched_add_ $suffix>]() {
                $crate::vector::tests::test_batched_add::<$V>($ctx2);
            }
            #[test]
            fn [<test_batched_norm_max_across_batches_ $suffix>]() {
                $crate::vector::tests::test_batched_norm_max_across_batches::<$V>($ctx2);
            }
            #[test]
            fn [<test_batched_norm_l1_ $suffix>]() {
                $crate::vector::tests::test_batched_norm_l1::<$V>($ctx2);
            }
            #[test]
            fn [<test_batched_squared_norm_ $suffix>]() {
                $crate::vector::tests::test_batched_squared_norm::<$V>($ctx2);
            }
            #[test]
            fn [<test_batched_set_index_ $suffix>]() {
                $crate::vector::tests::test_batched_set_index::<$V>($ctx3);
            }
            #[test]
            #[should_panic(expected = "not supported for batched")]
            fn [<test_batched_get_index_panics_ $suffix>]() {
                $crate::vector::tests::test_batched_get_index_panics::<$V>($ctx2);
            }
            #[test]
            fn [<test_batched_fill_ $suffix>]() {
                $crate::vector::tests::test_batched_fill::<$V>($ctx2);
            }
            #[test]
            fn [<test_batched_component_mul_ $suffix>]() {
                $crate::vector::tests::test_batched_component_mul::<$V>($ctx2);
            }
            #[test]
            fn [<test_batched_assign_at_indices_ $suffix>]() {
                $crate::vector::tests::test_batched_assign_at_indices::<$V>($ctx2);
            }
            #[test]
            fn [<test_batched_root_finding_consistent_ $suffix>]() {
                $crate::vector::tests::test_batched_root_finding_consistent::<$V>($ctx2);
            }
            #[test]
            #[should_panic(expected = "differ across batches")]
            fn [<test_batched_root_finding_inconsistent_ $suffix>]() {
                $crate::vector::tests::test_batched_root_finding_inconsistent::<$V>($ctx2);
            }
            #[test]
            fn [<test_batched_axpy_broadcast_ $suffix>]() {
                $crate::vector::tests::test_batched_axpy_broadcast::<$V>($ctx2);
            }
            #[test]
            fn [<test_batched_copy_from_broadcast_ $suffix>]() {
                $crate::vector::tests::test_batched_copy_from_broadcast::<$V>($ctx2);
            }
            #[test]
            fn [<test_batched_component_div_ $suffix>]() {
                $crate::vector::tests::test_batched_component_div::<$V>($ctx2);
            }
            #[test]
            fn [<test_batched_component_mul_broadcast_ $suffix>]() {
                $crate::vector::tests::test_batched_component_mul_broadcast::<$V>($ctx2);
            }
            #[test]
            fn [<test_batched_component_div_broadcast_ $suffix>]() {
                $crate::vector::tests::test_batched_component_div_broadcast::<$V>($ctx2);
            }
            #[test]
            fn [<test_batched_add_assign_broadcast_ $suffix>]() {
                $crate::vector::tests::test_batched_add_assign_broadcast::<$V>($ctx2);
            }
            #[test]
            fn [<test_batched_sub_assign_broadcast_ $suffix>]() {
                $crate::vector::tests::test_batched_sub_assign_broadcast::<$V>($ctx2);
            }
            #[test]
            fn [<test_batched_sub_ $suffix>]() {
                $crate::vector::tests::test_batched_sub::<$V>($ctx2);
            }
            #[test]
            fn [<test_batched_sub_assign_ $suffix>]() {
                $crate::vector::tests::test_batched_sub_assign::<$V>($ctx2);
            }
            #[test]
            fn [<test_batched_from_slice_ $suffix>]() {
                $crate::vector::tests::test_batched_from_slice::<$V>($ctx2);
            }
            #[test]
            fn [<test_batched_mul_scalar_ $suffix>]() {
                $crate::vector::tests::test_batched_mul_scalar::<$V>($ctx2);
            }
            #[test]
            fn [<test_batched_div_scalar_ $suffix>]() {
                $crate::vector::tests::test_batched_div_scalar::<$V>($ctx2);
            }
            #[test]
            fn [<test_batched_copy_from_indices_ $suffix>]() {
                $crate::vector::tests::test_batched_copy_from_indices::<$V>($ctx2);
            }
            #[test]
            fn [<test_batched_gather_ $suffix>]() {
                $crate::vector::tests::test_batched_gather::<$V>($ctx2);
            }
            #[test]
            fn [<test_batched_scatter_ $suffix>]() {
                $crate::vector::tests::test_batched_scatter::<$V>($ctx2);
            }
            #[test]
            fn [<test_batched_get_batch_ $suffix>]() {
                $crate::vector::tests::test_batched_get_batch::<$V>($ctx2);
            }
            #[test]
            fn [<test_batched_get_batch_mut_ $suffix>]() {
                $crate::vector::tests::test_batched_get_batch_mut::<$V>($ctx2);
            }
            #[test]
            fn [<test_batched_axpy_v_ $suffix>]() {
                $crate::vector::tests::test_batched_axpy_v::<$V>($ctx2);
            }
            #[test]
            fn [<test_batched_mul_assign_scalar_ $suffix>]() {
                $crate::vector::tests::test_batched_mul_assign_scalar::<$V>($ctx2);
            }
            #[test]
            fn [<test_batched_copy_from_view_ $suffix>]() {
                $crate::vector::tests::test_batched_copy_from_view::<$V>($ctx2);
            }
            #[test]
            fn [<test_batched_axpy_new_ $suffix>]() {
                $crate::vector::tests::test_batched_axpy_new::<$V>($ctx2);
            }
            #[test]
            fn [<test_batched_axpy_new_broadcast_ $suffix>]() {
                $crate::vector::tests::test_batched_axpy_new_broadcast::<$V>($ctx2);
            }
            #[test]
            #[should_panic(expected = "alpha.len() must equal")]
            fn [<test_batched_axpy_new_bad_length_ $suffix>]() {
                $crate::vector::tests::test_batched_axpy_new_bad_length::<$V>($ctx2);
            }
            #[test]
            #[should_panic(expected = "incompatible nbatch")]
            fn [<test_batched_axpy_incompatible_ $suffix>]() {
                $crate::vector::tests::test_batched_axpy_incompatible::<$V>($ctx2, $ctx3);
            }
            #[test]
            #[should_panic(expected = "incompatible nbatch")]
            fn [<test_batched_copy_from_incompatible_ $suffix>]() {
                $crate::vector::tests::test_batched_copy_from_incompatible::<$V>($ctx2, $ctx3);
            }
            #[test]
            #[should_panic(expected = "incompatible nbatch")]
            fn [<test_batched_add_assign_incompatible_ $suffix>]() {
                $crate::vector::tests::test_batched_add_assign_incompatible::<$V>($ctx2, $ctx3);
            }
            #[test]
            #[should_panic(expected = "incompatible nbatch")]
            fn [<test_batched_component_mul_incompatible_ $suffix>]() {
                $crate::vector::tests::test_batched_component_mul_incompatible::<$V>($ctx2, $ctx3);
            }

            // --- Strided view tests (batched, using $ctx2) ---
            #[test]
            fn [<test_strided_view_set_index_ $suffix>]() {
                $crate::vector::tests::test_strided_view_set_index::<<$V as $crate::DefaultDenseMatrix>::M>($ctx2);
            }
            #[test]
            fn [<test_strided_view_mut_copy_from_ $suffix>]() {
                $crate::vector::tests::test_strided_view_mut_copy_from::<<$V as $crate::DefaultDenseMatrix>::M>($ctx2);
            }
            #[test]
            fn [<test_strided_view_mut_axpy_ $suffix>]() {
                $crate::vector::tests::test_strided_view_mut_axpy::<<$V as $crate::DefaultDenseMatrix>::M>($ctx2);
            }
            #[test]
            fn [<test_strided_view_mut_mul_assign_scalar_ $suffix>]() {
                $crate::vector::tests::test_strided_view_mut_mul_assign_scalar::<<$V as $crate::DefaultDenseMatrix>::M>($ctx2);
            }
            #[test]
            fn [<test_strided_view_mut_add_assign_ $suffix>]() {
                $crate::vector::tests::test_strided_view_mut_add_assign::<<$V as $crate::DefaultDenseMatrix>::M>($ctx2);
            }
            #[test]
            fn [<test_strided_view_mut_sub_assign_ $suffix>]() {
                $crate::vector::tests::test_strided_view_mut_sub_assign::<<$V as $crate::DefaultDenseMatrix>::M>($ctx2);
            }
            #[test]
            fn [<test_strided_view_add_assign_broadcast_ $suffix>]() {
                $crate::vector::tests::test_strided_view_add_assign_broadcast::<<$V as $crate::DefaultDenseMatrix>::M>($ctx2);
            }
            #[test]
            fn [<test_strided_view_add_owned_ $suffix>]() {
                $crate::vector::tests::test_strided_view_add_owned::<<$V as $crate::DefaultDenseMatrix>::M>($ctx2);
            }
            #[test]
            fn [<test_strided_view_squared_norm_ $suffix>]() {
                $crate::vector::tests::test_strided_view_squared_norm::<<$V as $crate::DefaultDenseMatrix>::M>($ctx2);
            }
            #[test]
            fn [<test_strided_view_into_owned_ $suffix>]() {
                $crate::vector::tests::test_strided_view_into_owned::<<$V as $crate::DefaultDenseMatrix>::M>($ctx2);
            }
            #[test]
            fn [<test_strided_view_component_mul_ $suffix>]() {
                $crate::vector::tests::test_strided_view_component_mul::<<$V as $crate::DefaultDenseMatrix>::M>($ctx2);
            }
            #[test]
            fn [<test_strided_view_component_div_ $suffix>]() {
                $crate::vector::tests::test_strided_view_component_div::<<$V as $crate::DefaultDenseMatrix>::M>($ctx2);
            }
            #[test]
            fn [<test_strided_view_mul_scalar_ $suffix>]() {
                $crate::vector::tests::test_strided_view_mul_scalar::<<$V as $crate::DefaultDenseMatrix>::M>($ctx2);
            }
            #[test]
            fn [<test_strided_view_fill_ $suffix>]() {
                $crate::vector::tests::test_strided_view_fill::<<$V as $crate::DefaultDenseMatrix>::M>($ctx2);
            }
            #[test]
            fn [<test_strided_view_assign_at_indices_ $suffix>]() {
                $crate::vector::tests::test_strided_view_assign_at_indices::<<$V as $crate::DefaultDenseMatrix>::M>($ctx2);
            }
            #[test]
            fn [<test_strided_view_copy_from_indices_ $suffix>]() {
                $crate::vector::tests::test_strided_view_copy_from_indices::<<$V as $crate::DefaultDenseMatrix>::M>($ctx2);
            }
            #[test]
            fn [<test_strided_view_gather_ $suffix>]() {
                $crate::vector::tests::test_strided_view_gather::<<$V as $crate::DefaultDenseMatrix>::M>($ctx2);
            }
            #[test]
            fn [<test_strided_view_scatter_ $suffix>]() {
                $crate::vector::tests::test_strided_view_scatter::<<$V as $crate::DefaultDenseMatrix>::M>($ctx2);
            }
        }
    };
}
#[cfg(test)]
pub(crate) use generate_vector_tests_batched;
#[cfg(test)]
pub(crate) use generate_vector_tests_nonbatched;
