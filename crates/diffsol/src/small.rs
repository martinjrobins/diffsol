//! Fixed-capacity host-side containers for the solvers' small coefficient arrays.
//!
//! Runge-Kutta tableaux and the BDF difference-table coefficients are only ever a handful of
//! values, are computed from host scalars, and are consumed as plain `&[T]` slices by
//! [`DenseMatrix::gemv_cols`] and [`DenseMatrix::mul_cols_by`]. Holding them in backend
//! vectors costs an allocation and — on CUDA — a device-to-host copy per element read. These
//! two types hold them inline instead, so they need no allocation and reading one is just an
//! index.
//!
//! Both are capacity-`N` with a runtime shape. That is deliberate rather than
//! `SmallMat<T, ROWS, COLS>`: the interpolation `beta` matrix is not square (7x4 for `tsit45`),
//! and the BDF `R`/`U` blocks are *packed at the current order*, so their column stride changes
//! from step to step. A fixed-stride type could express neither.

use std::fmt::Debug;
use std::ops::{Index, IndexMut};

use crate::scalar::Scalar;

/// A fixed-capacity list of coefficients; the leading [`len`](SmallVec::len) entries are live.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmallVec<T, const N: usize> {
    data: [T; N],
    len: usize,
}

impl<T: Scalar, const N: usize> SmallVec<T, N> {
    /// A list of `len` zeros.
    ///
    /// Panics if `len > N`.
    #[inline]
    pub fn zeros(len: usize) -> Self {
        assert!(len <= N, "SmallVec: length {len} exceeds capacity {N}");
        Self {
            data: [T::zero(); N],
            len,
        }
    }

    /// Copies `values` in.
    ///
    /// Panics if `values.len() > N`.
    #[inline]
    pub fn from_slice(values: &[T]) -> Self {
        let mut out = Self::zeros(values.len());
        out.data[..values.len()].copy_from_slice(values);
        out
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The live entries, ready to hand to a kernel.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data[..self.len]
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data[..self.len]
    }
}

/// Panics if the index is past the live entries.
///
/// Indexes `data` directly rather than going through [`as_slice`](SmallVec::as_slice): the
/// bound is then the constant `N`, which the optimiser can discharge inside a loop, and the
/// explicit check keeps the "only the live entries" contract.
impl<T: Scalar, const N: usize> Index<usize> for SmallVec<T, N> {
    type Output = T;

    #[inline]
    fn index(&self, i: usize) -> &T {
        assert!(i < self.len, "SmallVec: index {i} past length {}", self.len);
        &self.data[i]
    }
}

impl<T: Scalar, const N: usize> IndexMut<usize> for SmallVec<T, N> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        assert!(i < self.len, "SmallVec: index {i} past length {}", self.len);
        &mut self.data[i]
    }
}

/// A fixed-capacity column-major matrix.
///
/// Columns are densely packed with stride [`nrows`](SmallMat::nrows), so a column is a
/// contiguous slice ([`as_col_slice`](SmallMat::as_col_slice)) and the logical shape can change
/// without restriding. A *row* is therefore strided — code that needs contiguous rows should
/// store the transpose (see [`transposed`](SmallMat::transposed)).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmallMat<T, const N: usize> {
    data: [T; N],
    nrows: usize,
    ncols: usize,
}

impl<T: Scalar, const N: usize> SmallMat<T, N> {
    /// An `nrows x ncols` block of zeros.
    ///
    /// Panics if `nrows * ncols > N`.
    #[inline]
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        assert!(
            nrows * ncols <= N,
            "SmallMat: {nrows}x{ncols} exceeds capacity {N}"
        );
        Self {
            data: [T::zero(); N],
            nrows,
            ncols,
        }
    }

    /// Copies `values` in, interpreted column-major.
    ///
    /// Panics unless `values.len() == nrows * ncols`.
    #[inline]
    pub fn from_slice(nrows: usize, ncols: usize, values: &[T]) -> Self {
        assert_eq!(
            values.len(),
            nrows * ncols,
            "SmallMat: expected {nrows} * {ncols} values"
        );
        let mut out = Self::zeros(nrows, ncols);
        out.data[..values.len()].copy_from_slice(values);
        out
    }

    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// The live entries, column-major.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data[..self.nrows * self.ncols]
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data[..self.nrows * self.ncols]
    }

    /// Column `j`, a contiguous slice of `nrows` entries.
    #[inline]
    pub fn as_col_slice(&self, j: usize) -> &[T] {
        assert!(j < self.ncols, "SmallMat: column {j} out of bounds");
        &self.data[j * self.nrows..][..self.nrows]
    }

    #[inline]
    pub fn as_col_slice_mut(&mut self, j: usize) -> &mut [T] {
        assert!(j < self.ncols, "SmallMat: column {j} out of bounds");
        &mut self.data[j * self.nrows..][..self.nrows]
    }

    /// `self` with rows and columns swapped.
    ///
    /// Used to store a matrix whose *rows* are the coefficient runs a kernel consumes, so that
    /// each run comes out of [`as_col_slice`](SmallMat::as_col_slice) contiguously.
    #[inline]
    pub fn transposed(&self) -> Self {
        let mut out = Self::zeros(self.ncols, self.nrows);
        for j in 0..self.ncols {
            for i in 0..self.nrows {
                out[(j, i)] = self[(i, j)];
            }
        }
        out
    }

    /// `self * rhs`.
    ///
    /// Panics unless `self.ncols == rhs.nrows`, or if the result would exceed capacity.
    /// Accumulates over the inner index for each output entry, matching a column-major `gemm`.
    ///
    /// The three shapes are read into locals up front: the loops then index flat slices at a
    /// stride the optimiser can keep in a register, rather than re-loading `nrows` from the
    /// struct on every element.
    #[inline]
    pub fn mat_mul(&self, rhs: &Self) -> Self {
        assert_eq!(
            self.ncols, rhs.nrows,
            "SmallMat: cannot multiply {}x{} by {}x{}",
            self.nrows, self.ncols, rhs.nrows, rhs.ncols
        );
        let (m, k, n) = (self.nrows, self.ncols, rhs.ncols);
        let mut out = Self::zeros(m, n);
        let a = &self.data[..m * k];
        let b = &rhs.data[..k * n];
        let c = &mut out.data[..m * n];
        for j in 0..n {
            for i in 0..m {
                let mut acc = T::zero();
                for l in 0..k {
                    acc += a[l * m + i] * b[j * k + l];
                }
                c[j * m + i] = acc;
            }
        }
        out
    }
}

/// Row-then-column indexing, `m[(i, j)]`, in the natural orientation regardless of the
/// column-major storage. Panics if `(i, j)` is outside the live block.
///
/// Indexes `data` directly rather than reslicing a column, so the only bound left for the
/// optimiser is the constant `N` — reslicing per element is what made the element-wise loops
/// in the BDF coefficient blocks measurably slower than raw arrays.
impl<T: Scalar, const N: usize> Index<(usize, usize)> for SmallMat<T, N> {
    type Output = T;

    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &T {
        assert!(
            i < self.nrows && j < self.ncols,
            "SmallMat: ({i}, {j}) out of bounds"
        );
        &self.data[j * self.nrows + i]
    }
}

impl<T: Scalar, const N: usize> IndexMut<(usize, usize)> for SmallMat<T, N> {
    #[inline]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut T {
        assert!(
            i < self.nrows && j < self.ncols,
            "SmallMat: ({i}, {j}) out of bounds"
        );
        &mut self.data[j * self.nrows + i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_vec_slices_only_the_live_entries() {
        let v = SmallVec::<f64, 8>::from_slice(&[1.0, 2.0, 3.0]);
        assert_eq!(v.len(), 3);
        assert_eq!(v.as_slice(), &[1.0, 2.0, 3.0]);
        assert_eq!(v[2], 3.0);

        let mut z = SmallVec::<f64, 8>::zeros(2);
        z[1] = 5.0;
        assert_eq!(z.as_slice(), &[0.0, 5.0]);
    }

    #[test]
    #[should_panic(expected = "exceeds capacity")]
    fn small_vec_rejects_over_capacity() {
        SmallVec::<f64, 2>::from_slice(&[1.0, 2.0, 3.0]);
    }

    #[test]
    fn small_mat_columns_are_contiguous_when_not_square() {
        // 3x2, column-major: col0 = [1,2,3], col1 = [4,5,6]
        let m = SmallMat::<f64, 32>::from_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(m.nrows(), 3);
        assert_eq!(m.ncols(), 2);
        assert_eq!(m.as_col_slice(0), &[1.0, 2.0, 3.0]);
        assert_eq!(m.as_col_slice(1), &[4.0, 5.0, 6.0]);
        assert_eq!(m[(2, 1)], 6.0);
        assert_eq!(m.as_slice().len(), 6);
    }

    #[test]
    fn small_mat_packs_to_the_current_shape() {
        // the column stride follows nrows, so a reshape repacks rather than restrides:
        // this is what the BDF R/U blocks rely on as the order changes
        let wide = SmallMat::<f64, 36>::from_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(wide.as_col_slice(1), &[3.0, 4.0]);
        let tall = SmallMat::<f64, 36>::from_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(tall.as_col_slice(1), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn small_mat_transposed_makes_rows_contiguous() {
        // 3x2 -> 2x3; row i of the original is column i of the transpose
        let m = SmallMat::<f64, 32>::from_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = m.transposed();
        assert_eq!(t.nrows(), 2);
        assert_eq!(t.ncols(), 3);
        // original row 0 is [1, 4]
        assert_eq!(t.as_col_slice(0), &[1.0, 4.0]);
        assert_eq!(t.as_col_slice(2), &[3.0, 6.0]);
        assert_eq!(t.transposed(), m);
    }

    #[test]
    fn small_mat_mat_mul() {
        // a = [[1, 3], [2, 4]] column-major [1,2,3,4]; b = [[5, 7], [6, 8]]
        let a = SmallMat::<f64, 36>::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let b = SmallMat::<f64, 36>::from_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]);
        // a * b = [[1*5+3*6, 1*7+3*8], [2*5+4*6, 2*7+4*8]] = [[23, 31], [34, 46]]
        let c = a.mat_mul(&b);
        assert_eq!(c[(0, 0)], 23.0);
        assert_eq!(c[(1, 0)], 34.0);
        assert_eq!(c[(0, 1)], 31.0);
        assert_eq!(c[(1, 1)], 46.0);

        // identity leaves it alone
        let eye = SmallMat::<f64, 36>::from_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        assert_eq!(a.mat_mul(&eye), a);
        assert_eq!(eye.mat_mul(&a), a);
    }

    #[test]
    fn small_mat_mat_mul_non_square() {
        // (2x3) * (3x2) -> 2x2
        let a = SmallMat::<f64, 36>::from_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = SmallMat::<f64, 36>::from_slice(3, 2, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let c = a.mat_mul(&b);
        assert_eq!(c.nrows(), 2);
        assert_eq!(c.ncols(), 2);
        // b's col0 selects a's col0; b's col1 selects a's col1
        assert_eq!(c.as_col_slice(0), a.as_col_slice(0));
        assert_eq!(c.as_col_slice(1), a.as_col_slice(1));
    }

    #[test]
    #[should_panic(expected = "cannot multiply")]
    fn small_mat_mat_mul_rejects_mismatched_shapes() {
        let a = SmallMat::<f64, 36>::from_slice(2, 3, &[1.0; 6]);
        let b = SmallMat::<f64, 36>::from_slice(2, 2, &[1.0; 4]);
        let _ = a.mat_mul(&b);
    }

    #[test]
    #[should_panic(expected = "exceeds capacity")]
    fn small_mat_rejects_over_capacity() {
        SmallMat::<f64, 4>::zeros(3, 3);
    }
}
