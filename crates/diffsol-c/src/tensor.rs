use crate::{
    error::DiffsolRtError,
    scalar_type::{Scalar, ScalarType},
};
use diffsol::{
    error::DiffsolError, Context, DenseMatrix, FaerScalar, Matrix, MatrixCommon, NalgebraScalar,
    Vector, VectorCommon,
};
use num_traits::FromPrimitive;
use std::any::Any;
use std::marker::PhantomData;
use std::sync::Arc;

/// The maximum number of axes of a [Tensor] or a [TensorView].
pub const MAX_NDIM: usize = 3;

/// Convert a value into a [Tensor] that owns it, with no copy of the data.
pub(crate) trait ToTensor {
    fn to_tensor(self) -> Tensor;
}

/// Convert a [TensorView] to a diffsol vector or dense matrix. The scalar type of the view can be
/// `f32` or `f64`.
pub(crate) trait FromTensor: Sized {
    /// The diffsol context of the new value.
    type Context;
    fn from_tensor(tensor: TensorView<'_>, ctx: &Self::Context) -> Result<Self, DiffsolRtError>;
}

/// All the values of `tensor` in column-major order (the first axis changes fastest), as `T`
/// values.
pub(crate) fn column_major_values<T: FromPrimitive>(
    tensor: TensorView<'_>,
) -> Result<Vec<T>, DiffsolRtError> {
    Ok(match tensor.dtype() {
        ScalarType::F32 => tensor
            .columns::<f32>()?
            .flatten()
            .map(|&x| T::from_f32(x).unwrap())
            .collect(),
        ScalarType::F64 => tensor
            .columns::<f64>()?
            .flatten()
            .map(|&x| T::from_f64(x).unwrap())
            .collect(),
    })
}

/// Read the values of `tensor` for a diffsol value with `nbatch` lanes.
///
/// The column-major order of a tensor with the lane axis last is the storage order of diffsol:
/// a vector stores lane `b`, value `i` at `b * n + i`, and a dense matrix stores lane `b`, row
/// `i`, column `j` at `(b * ncols + j) * nrows + i`. `tensor` must have `nbatch` lanes or 1 lane
/// on its last axis. The values of one lane are copied to all `nbatch` lanes, so the result
/// always has a value for each lane.
fn lane_values<T: FromPrimitive + Clone>(
    tensor: TensorView<'_>,
    nbatch: usize,
) -> Result<Vec<T>, DiffsolRtError> {
    let nlanes = *tensor.shape().last().unwrap();
    if nlanes != nbatch && nlanes != 1 {
        return Err(
            DiffsolError::Other(format!("Expected 1 or {nbatch} lanes, got {nlanes}")).into(),
        );
    }
    let values: Vec<T> = column_major_values(tensor)?;
    if nlanes == nbatch {
        return Ok(values);
    }
    let mut lanes = Vec::with_capacity(values.len() * nbatch);
    for _ in 0..nbatch {
        lanes.extend_from_slice(&values);
    }
    Ok(lanes)
}

/// Convert an `[n, nlanes]` array to a diffsol vector with the context `ctx`.
fn vector_from<V: Vector<T: Scalar>>(
    tensor: TensorView<'_>,
    ctx: &V::C,
) -> Result<V, DiffsolRtError> {
    tensor.expect_ndim(2)?;
    Ok(V::from_vec(lane_values(tensor, ctx.nbatch())?, ctx.clone()))
}

/// Convert an `[nrows, ncols, nlanes]` array to a diffsol dense matrix with the context `ctx`.
fn dense_matrix_from<M: DenseMatrix<T: Scalar>>(
    tensor: TensorView<'_>,
    ctx: &M::C,
) -> Result<M, DiffsolRtError> {
    tensor.expect_ndim(3)?;
    let (nrows, ncols) = (tensor.shape()[0], tensor.shape()[1]);
    let data = lane_values(tensor, ctx.nbatch())?;
    Ok(M::from_vec(nrows, ncols, data, ctx.clone()))
}

// A diffsol vector converts from an `[n, nlanes]` array, and a diffsol dense matrix from an
// `[nrows, ncols, nlanes]` array. `nlanes` must be the number of lanes of the context, or 1. If
// it is 1, all the lanes get the values of the one lane.

impl<T: Scalar + FaerScalar> FromTensor for diffsol::FaerVec<T> {
    type Context = diffsol::FaerContext;
    fn from_tensor(tensor: TensorView<'_>, ctx: &Self::Context) -> Result<Self, DiffsolRtError> {
        vector_from(tensor, ctx)
    }
}

impl<T: Scalar + NalgebraScalar> FromTensor for diffsol::NalgebraVec<T> {
    type Context = diffsol::NalgebraContext;
    fn from_tensor(tensor: TensorView<'_>, ctx: &Self::Context) -> Result<Self, DiffsolRtError> {
        vector_from(tensor, ctx)
    }
}

impl<T: Scalar + FaerScalar> FromTensor for diffsol::FaerMat<T> {
    type Context = diffsol::FaerContext;
    fn from_tensor(tensor: TensorView<'_>, ctx: &Self::Context) -> Result<Self, DiffsolRtError> {
        dense_matrix_from(tensor, ctx)
    }
}

impl<T: Scalar + NalgebraScalar> FromTensor for diffsol::NalgebraMat<T> {
    type Context = diffsol::NalgebraContext;
    fn from_tensor(tensor: TensorView<'_>, ctx: &Self::Context) -> Result<Self, DiffsolRtError> {
        dense_matrix_from(tensor, ctx)
    }
}

/// The layout of a faer matrix, with the data pointer `ptr` of `m`.
fn faer_layout<T: Scalar + FaerScalar>(m: &faer::Mat<T>, ptr: *const T) -> RawLayout {
    // faer gives the strides in values; a matrix that owns its data has positive strides
    let size = std::mem::size_of::<T>();
    let strides = [
        m.row_stride() as usize * size,
        m.col_stride() as usize * size,
    ];
    let shape = [m.nrows(), m.ncols()];
    RawLayout::valid(ptr as *const u8, &shape, &strides, T::scalar_type())
}

/// The layout of a nalgebra matrix, with the data pointer `ptr` of `m`.
fn nalgebra_layout<T: Scalar + NalgebraScalar>(
    m: &nalgebra::DMatrix<T>,
    ptr: *const T,
) -> RawLayout {
    // nalgebra gives the strides in values
    let size = std::mem::size_of::<T>();
    let (rs, cs) = m.strides();
    let shape = [m.nrows(), m.ncols()];
    RawLayout::valid(
        ptr as *const u8,
        &shape,
        &[rs * size, cs * size],
        T::scalar_type(),
    )
}

/// Give the storage of a diffsol dense matrix, an `nrows × (nbatch * ncols)` matrix where the
/// columns of lane `b` are the column block `b`, the shape `[nrows, ncols, nbatch]`.
fn matrix_lanes(storage: RawLayout, nbatch: usize) -> RawLayout {
    let (nrows, ncols) = (storage.shape[0], storage.shape[1] / nbatch);
    let (row_stride, col_stride) = (storage.strides[0], storage.strides[1]);
    storage.with_layout(
        &[nrows, ncols, nbatch],
        &[row_stride, col_stride, ncols * col_stride],
    )
}

/// A value that a [Tensor] can show with no copy of the data.
///
/// A diffsol vector or dense matrix shows its storage with the lane axis last. The storage of a
/// diffsol vector is an `n × nbatch` matrix with one column for each lane, which is already the
/// shape `[n, nbatch]`. A dense matrix has the shape `[nrows, ncols, nbatch]`. A `Vec<T>` has the
/// shape `[len]`.
///
/// # Safety
/// [TensorLayout::tensor_layout] must give the layout of data that the value holds on the heap,
/// so that the data does not move when the value moves. The data must not change while the value
/// is borrowed (no interior mutability).
pub unsafe trait TensorLayout {
    /// The layout of the data.
    fn tensor_layout(&self) -> RawLayout;
}

/// A diffsol vector that a [TensorViewMut] can show, so that a C callback can write its values
/// with no copy. The layout is `[n, nbatch]`, the same as [TensorLayout::tensor_layout].
///
/// # Safety
/// As for [TensorLayout]. [TensorLayoutMut::tensor_layout_mut] must give the same layout as
/// [TensorLayout::tensor_layout], with a pointer that allows writes to each value of the layout
/// while the value is mutably borrowed.
pub unsafe trait TensorLayoutMut: TensorLayout {
    /// The layout of the data, with a pointer that allows writes.
    fn tensor_layout_mut(&mut self) -> RawLayout;
}

// SAFETY: faer and nalgebra keep the values of a vector or a matrix on the heap, and change them
// only through `&mut`
unsafe impl<T: Scalar + FaerScalar> TensorLayout for diffsol::FaerVec<T> {
    fn tensor_layout(&self) -> RawLayout {
        faer_layout(self.inner(), self.inner().as_ptr())
    }
}

// SAFETY: the pointer comes from `&mut` of the faer matrix, so writes through it are allowed
unsafe impl<T: Scalar + FaerScalar> TensorLayoutMut for diffsol::FaerVec<T> {
    fn tensor_layout_mut(&mut self) -> RawLayout {
        let ptr = self.inner_mut().as_ptr_mut();
        faer_layout(self.inner(), ptr)
    }
}

// SAFETY: as for `FaerVec`
unsafe impl<T: Scalar + NalgebraScalar> TensorLayout for diffsol::NalgebraVec<T> {
    fn tensor_layout(&self) -> RawLayout {
        nalgebra_layout(self.inner(), self.inner().as_ptr())
    }
}

// SAFETY: the pointer comes from `&mut` of the nalgebra matrix, so writes through it are allowed
unsafe impl<T: Scalar + NalgebraScalar> TensorLayoutMut for diffsol::NalgebraVec<T> {
    fn tensor_layout_mut(&mut self) -> RawLayout {
        let ptr = self.inner_mut().as_mut_ptr();
        nalgebra_layout(self.inner(), ptr)
    }
}

// SAFETY: as for `FaerVec`
unsafe impl<T: Scalar + FaerScalar> TensorLayout for diffsol::FaerMat<T> {
    fn tensor_layout(&self) -> RawLayout {
        let layout = faer_layout(self.inner(), self.inner().as_ptr());
        matrix_lanes(layout, self.context().nbatch())
    }
}

// SAFETY: as for `FaerVec`
unsafe impl<T: Scalar + NalgebraScalar> TensorLayout for diffsol::NalgebraMat<T> {
    fn tensor_layout(&self) -> RawLayout {
        let layout = nalgebra_layout(self.inner(), self.inner().as_ptr());
        matrix_lanes(layout, self.context().nbatch())
    }
}

// SAFETY: a `Vec` keeps its values on the heap, and changes them only through `&mut`
unsafe impl<T: Scalar> TensorLayout for Vec<T> {
    fn tensor_layout(&self) -> RawLayout {
        RawLayout::valid_col_major(self.as_ptr() as *const u8, &[self.len()], T::scalar_type())
    }
}

impl<V: TensorLayout + 'static> ToTensor for V {
    fn to_tensor(self) -> Tensor {
        let layout = self.tensor_layout();
        // SAFETY: `TensorLayout` guarantees that the data is on the heap of `self`, so it does
        // not move when `self` moves into the tensor, and that it does not change
        unsafe { Tensor::from_layout(self, layout) }
    }
}

/// A tensor of `part(&owner)`, a value inside `owner`, with no copy of the data.
///
/// The tensor holds a clone of `owner`, so the data stays valid after all the other clones of
/// `owner` are dropped. An `Arc` gives only shared access, so the data does not change while the
/// tensor exists.
pub(crate) fn shared_tensor<O, V>(owner: Arc<O>, part: impl FnOnce(&O) -> &V) -> Tensor
where
    O: ?Sized + 'static,
    V: TensorLayout,
{
    let layout = part(&owner).tensor_layout();
    // SAFETY: `part` gives a value that `owner` holds (or a static value), and the `Arc`
    // allocation does not move when `owner` moves into the tensor. `TensorLayout` guarantees that
    // the data of that value is on the heap and does not change while it is borrowed
    unsafe { Tensor::from_layout(owner, layout) }
}

fn scalar_size(dtype: ScalarType) -> usize {
    match dtype {
        ScalarType::F32 => std::mem::size_of::<f32>(),
        ScalarType::F64 => std::mem::size_of::<f64>(),
    }
}

fn layout_error(message: String) -> DiffsolRtError {
    DiffsolError::Other(message).into()
}

/// Check the layout rules of [Tensor] and [TensorView] (the [Tensor] docs give the rules).
fn check_layout(
    ptr: *const u8,
    shape: &[usize],
    strides: &[usize],
    dtype: ScalarType,
) -> Result<(), DiffsolRtError> {
    let ndim = shape.len();
    if ndim == 0 || ndim > MAX_NDIM {
        return Err(layout_error(format!(
            "A tensor has 1 to {MAX_NDIM} axes, got {ndim}"
        )));
    }
    if strides.len() != ndim {
        return Err(layout_error(format!(
            "Expected {ndim} strides, got {}",
            strides.len()
        )));
    }
    let size = scalar_size(dtype);
    if shape[0] > 1 && strides[0] != size {
        return Err(layout_error(format!(
            "The values of a column are not adjacent: strides[0] is {} bytes, expected {size}",
            strides[0]
        )));
    }
    // the stride of an axis of length 1 is never used
    let strides_aligned = shape
        .iter()
        .zip(strides)
        .all(|(&n, &stride)| n <= 1 || stride.is_multiple_of(size));
    // the alignment of `f32` and `f64` is their size
    if !(ptr as usize).is_multiple_of(size) || !strides_aligned {
        return Err(layout_error("The data is not aligned".to_string()));
    }
    Ok(())
}

/// Check that the scalar type of the values is `T`.
fn check_dtype<T: Scalar>(dtype: ScalarType) -> Result<(), DiffsolRtError> {
    if dtype != T::scalar_type() {
        return Err(layout_error("Data type mismatch".to_string()));
    }
    Ok(())
}

/// The byte offset from the data pointer of the value at `index`, on the axes that `index`
/// covers. `shape` and `strides` are those of the same axes.
///
/// # Panics
/// If `index` does not have one value for each axis, or a value is out of range.
fn byte_offset(shape: &[usize], strides: &[usize], index: &[usize]) -> usize {
    assert_eq!(index.len(), shape.len(), "wrong number of indices");
    index
        .iter()
        .zip(shape)
        .zip(strides)
        .map(|((&i, &n), &stride)| {
            assert!(i < n, "index {i} is out of range for an axis of length {n}");
            i * stride
        })
        .sum()
}

/// The data pointer, the scalar type, the shape and the byte strides of a [Tensor], a
/// [TensorView] or a [TensorViewMut]. A `RawLayout` always obeys the layout rules (the [Tensor]
/// docs give the rules): each constructor checks them. It does not own or borrow the data, so the
/// reads are `unsafe`.
///
/// It has the C layout, so a C callback can read the fields of a [TensorView] directly. The value
/// at the index `(i0, i1, ...)` is at `ptr + i0 * strides[0] + i1 * strides[1] + ...` (in bytes).
/// Only the first `ndim` values of `shape` and `strides` are used.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RawLayout {
    /// The pointer to the first value.
    ptr: *const u8,
    /// The scalar type of the values.
    dtype: ScalarType,
    /// The number of axes, 1 to [MAX_NDIM].
    ndim: usize,
    /// The length of each axis. Only the first `ndim` values are used.
    shape: [usize; MAX_NDIM],
    /// The stride of each axis, in bytes. Only the first `ndim` values are used.
    strides: [usize; MAX_NDIM],
}

impl RawLayout {
    /// A layout with a shape and strides (in bytes), or an error if they do not obey the layout
    /// rules.
    fn new(
        ptr: *const u8,
        shape: &[usize],
        strides: &[usize],
        dtype: ScalarType,
    ) -> Result<Self, DiffsolRtError> {
        check_layout(ptr, shape, strides, dtype)?;
        let ndim = shape.len();
        let mut layout = Self {
            ptr,
            dtype,
            ndim,
            shape: [0; MAX_NDIM],
            strides: [0; MAX_NDIM],
        };
        layout.shape[..ndim].copy_from_slice(shape);
        layout.strides[..ndim].copy_from_slice(strides);
        Ok(layout)
    }
    /// A column-major (Fortran) layout of `shape`, or an error if it does not obey the layout
    /// rules (the number of axes, or the alignment of `ptr`).
    fn col_major(
        ptr: *const u8,
        shape: &[usize],
        dtype: ScalarType,
    ) -> Result<Self, DiffsolRtError> {
        let mut strides = [0; MAX_NDIM];
        let mut stride = scalar_size(dtype);
        for (s, &n) in strides.iter_mut().zip(shape) {
            *s = stride;
            stride *= n;
        }
        Self::new(ptr, shape, &strides[..shape.len().min(MAX_NDIM)], dtype)
    }
    /// As [RawLayout::new], for a layout that the caller knows obeys the layout rules.
    ///
    /// # Panics
    /// If the layout does not obey the layout rules.
    fn valid(ptr: *const u8, shape: &[usize], strides: &[usize], dtype: ScalarType) -> Self {
        Self::new(ptr, shape, strides, dtype)
            .unwrap_or_else(|err| panic!("invalid tensor layout: {err}"))
    }
    /// As [RawLayout::col_major], for a layout that the caller knows obeys the layout rules.
    ///
    /// # Panics
    /// If the layout does not obey the layout rules.
    fn valid_col_major(ptr: *const u8, shape: &[usize], dtype: ScalarType) -> Self {
        Self::col_major(ptr, shape, dtype)
            .unwrap_or_else(|err| panic!("invalid tensor layout: {err}"))
    }
    fn shape(&self) -> &[usize] {
        &self.shape[..self.ndim]
    }
    fn strides(&self) -> &[usize] {
        &self.strides[..self.ndim]
    }
    /// The same data with a new shape and new strides (in bytes).
    ///
    /// # Panics
    /// If the new layout has a different number of values, can reach a byte past the last value
    /// of the old layout, or does not obey the layout rules.
    fn with_layout(self, shape: &[usize], strides: &[usize]) -> Self {
        assert_eq!(
            shape.iter().product::<usize>(),
            self.shape().iter().product::<usize>()
        );
        let last_byte = |shape: &[usize], strides: &[usize]| -> usize {
            shape
                .iter()
                .zip(strides)
                .map(|(n, s)| n.saturating_sub(1) * s)
                .sum()
        };
        // a layout that reads past the old data would make the reads unsound
        assert!(last_byte(shape, strides) <= last_byte(self.shape(), self.strides()));
        Self::valid(self.ptr, shape, strides, self.dtype)
    }
    /// The `[n, ninner, nouter]` layout of a diffsol vector with the `[n, nlanes]` layout `self`,
    /// in an operation that writes `ninner * nouter` lanes.
    ///
    /// diffsol broadcasts an operand with fewer lanes in groups: each lane of an operand with
    /// `nouter` lanes repeats over `ninner` adjacent output lanes. The new layout shows this with
    /// strides of 0, so lane `(k, b)` of each operand is at `k * strides[1] + b * strides[2]`:
    /// - `nlanes == ninner * nouter`: the strides `[s, c, ninner * c]`;
    /// - `nlanes == nouter`: the strides `[s, 0, c]`;
    /// - `nlanes == 1`: the strides `[s, 0, 0]`.
    ///
    /// `s` is the size of one value, and `c` is the column stride of `self`.
    ///
    /// # Panics
    /// If `self` does not have 2 axes, or `nlanes` is not one of the values above.
    pub(crate) fn batch_lanes(self, ninner: usize, nouter: usize) -> Self {
        assert_eq!(self.ndim, 2, "a vector layout has 2 axes");
        let (n, nlanes) = (self.shape[0], self.shape[1]);
        let (s, c) = (self.strides[0], self.strides[1]);
        let strides = if nlanes == ninner * nouter {
            [s, c, ninner * c]
        } else if nlanes == nouter {
            [s, 0, c]
        } else if nlanes == 1 {
            [s, 0, 0]
        } else {
            panic!("cannot broadcast {nlanes} lanes into {ninner} x {nouter} lanes");
        };
        // each value of the new layout is a value of `self`, so the reads stay in the data
        Self::valid(self.ptr, &[n, ninner, nouter], &strides, self.dtype)
    }
    /// Check that no two indices of the layout give the same value, so that a mutable view can
    /// give each value one time. In column-major order, the stride of each axis must be at least
    /// the extent of the axes before it.
    fn check_no_overlap(&self) -> Result<(), DiffsolRtError> {
        let mut extent = scalar_size(self.dtype);
        for (&n, &stride) in self.shape().iter().zip(self.strides()) {
            if n > 1 {
                if stride < extent {
                    return Err(layout_error(
                        "The values of a mutable array overlap".to_string(),
                    ));
                }
                extent += (n - 1) * stride;
            }
        }
        Ok(())
    }
    /// Keep only the first `len` indices of the axis `axis`. The strides do not change, so each
    /// index that stays gives the same value.
    ///
    /// # Panics
    /// If `len` is larger than the length of the axis.
    fn truncate_axis(mut self, axis: usize, len: usize) -> Self {
        let n = self.shape()[axis];
        assert!(
            len <= n,
            "cannot keep {len} indices of an axis of length {n}"
        );
        self.shape[axis] = len;
        self
    }
    /// The pointer to the value at `index`, which is aligned (from the layout rules).
    ///
    /// # Panics
    /// If `index` does not have one value for each axis, or a value is out of range.
    fn index_ptr<T: Scalar>(&self, index: &[usize]) -> Result<*const T, DiffsolRtError> {
        check_dtype::<T>(self.dtype)?;
        let offset = byte_offset(self.shape(), self.strides(), index);
        // the index is in range, so the offset stays in the data of the layout
        Ok(self.ptr.wrapping_add(offset) as *const T)
    }
    /// The pointer to the first value of the column `[:, index...]`, and its length. The layout
    /// rules make the values of the column adjacent and aligned.
    ///
    /// # Panics
    /// If `index` does not have one value for each axis after the first, or a value is out of
    /// range.
    fn column_ptr<T: Scalar>(&self, index: &[usize]) -> Result<(*const T, usize), DiffsolRtError> {
        check_dtype::<T>(self.dtype)?;
        let offset = byte_offset(&self.shape()[1..], &self.strides()[1..], index);
        // the index is in range, so the offset stays in the data of the layout
        Ok((self.ptr.wrapping_add(offset) as *const T, self.shape[0]))
    }
    /// A reference to the value at `index`.
    ///
    /// # Safety
    /// For each index in the shape, the data must have a valid value of the type `dtype`, which
    /// stays valid and does not change during `'a`.
    unsafe fn index<'a, T: Scalar>(&self, index: &[usize]) -> Result<&'a T, DiffsolRtError> {
        let ptr = self.index_ptr::<T>(index)?;
        // SAFETY: the index is in range, so the caller guarantees a valid value of type `T` there
        // during `'a`, and the layout rules make it aligned
        Ok(unsafe { &*ptr })
    }
    /// The column `[:, index...]`, for a scalar type `T` that the caller checked.
    ///
    /// # Safety
    /// As for [RawLayout::index], and the scalar type must be `T`.
    unsafe fn column_of_type<'a, T: Scalar>(&self, index: &[usize]) -> &'a [T] {
        let offset = byte_offset(&self.shape()[1..], &self.strides()[1..], index);
        let len = self.shape[0];
        if len == 0 {
            return &[];
        }
        // SAFETY: the index is in range, and the layout rules make the `len` values of the column
        // adjacent and aligned. The caller guarantees that they are valid values of type `T`
        // during `'a`
        unsafe { std::slice::from_raw_parts(self.ptr.add(offset) as *const T, len) }
    }
    /// The column `[:, index...]`, one slice of `shape[0]` values.
    ///
    /// # Safety
    /// As for [RawLayout::index].
    unsafe fn column<'a, T: Scalar>(&self, index: &[usize]) -> Result<&'a [T], DiffsolRtError> {
        check_dtype::<T>(self.dtype)?;
        // SAFETY: the scalar type is `T`, and the caller guarantees the data
        Ok(unsafe { self.column_of_type(index) })
    }
    /// All the columns, in column-major order (the index on axis 1 changes fastest).
    ///
    /// # Safety
    /// As for [RawLayout::index].
    unsafe fn columns<'a, T: Scalar>(
        &self,
    ) -> Result<impl Iterator<Item = &'a [T]> + 'a, DiffsolRtError> {
        check_dtype::<T>(self.dtype)?;
        let layout = *self;
        let ncolumns: usize = layout.shape()[1..].iter().product();
        Ok((0..ncolumns).map(move |mut c| {
            // the index of column `c` on the axes `1..ndim`, in column-major order
            let mut index = [0; MAX_NDIM];
            for (i, &n) in index.iter_mut().zip(&layout.shape()[1..]) {
                *i = c % n;
                c /= n;
            }
            // SAFETY: the scalar type is `T`, and the caller guarantees the data
            unsafe { layout.column_of_type(&index[..layout.ndim - 1]) }
        }))
    }
}

/// An N-D array of `f32` or `f64` values that Rust owns, and that the host language (C, Python)
/// reads without a copy. The outputs of this crate are `Tensor` values. The inputs are
/// [TensorView] values.
///
/// # Data layout
/// A `Tensor` has a data pointer, a scalar type (`dtype`), a `shape` and a `strides` value for
/// each axis. The strides are in bytes. The value at the index `(i0, i1, ...)` is at the byte
/// offset `i0 * strides[0] + i1 * strides[1] + ...` from the data pointer.
///
/// The arrays use the column-major (Fortran) order of diffsol: the first axis changes fastest, and
/// the lane axis is the last axis. The arrays that this crate makes from diffsol values have these
/// layouts (`s` is the size of one value, and `c` is the column stride):
/// - A diffsol vector is `[n, nbatch]`, with the strides `[s, c]`. Each lane is one column.
/// - A diffsol dense matrix is `[nrows, ncols, nbatch]`, with the strides `[s, c, ncols * c]`.
///   Each lane is a column-major matrix.
/// - A Rust `Vec<T>`, for example `ts`, is `[n]`, with the stride `[s]`.
///
/// `c` is `nrows * s` for nalgebra. It can be larger for faer, which pads its columns. The padding
/// is not part of the array, and its memory is not initialized. A solution can also have spare
/// columns after the last time of each lane, which are not part of the array.
///
/// # Layout rules
/// Each `Tensor` and [TensorView] obeys these rules, which the constructors check:
/// - It has 1 to [MAX_NDIM] axes.
/// - The values of a column, `[:, j, k]`, are adjacent: `strides[0]` is the size of one value
///   (or the first axis has 1 value or none).
/// - The data pointer is aligned, and each stride is a multiple of the size of one value.
///
/// So each column is one slice, which [Tensor::column] and [Tensor::columns] give with no copy.
/// The padding is only after a column, so a column never contains it. [Tensor::index] gives one
/// value.
///
/// # Ownership
/// A `Tensor` always owns the value that holds its data (the `owner`), so the data stays valid
/// while the array exists.
pub struct Tensor {
    layout: RawLayout,
    /// The value that holds the data. It is only kept alive, never read.
    _owner: Box<dyn Any>,
}

impl Tensor {
    /// Make an array that owns `owner`, with a shape and strides (in bytes) that obey the layout
    /// rules. The [Tensor] docs give the data layout and the rules.
    ///
    /// # Panics
    /// If the layout does not obey the layout rules.
    ///
    /// # Safety
    /// `ptr` must point into data that `owner` holds on the heap, so that the data does not move
    /// when `owner` moves into the array. For each index in `shape`, `ptr` plus the byte offset
    /// from `strides` must point to a valid value of the type `dtype`. The data must not change
    /// while the array exists.
    pub unsafe fn new<O: 'static>(
        owner: O,
        ptr: *const u8,
        shape: &[usize],
        strides: &[usize],
        dtype: ScalarType,
    ) -> Self {
        let layout = RawLayout::valid(ptr, shape, strides, dtype);
        // SAFETY: the caller guarantees the data of `layout`
        unsafe { Self::from_layout(owner, layout) }
    }
    /// Make an array that owns `owner`, with the layout `layout`.
    ///
    /// # Safety
    /// As for [Tensor::new], for the data of `layout`.
    unsafe fn from_layout<O: 'static>(owner: O, layout: RawLayout) -> Self {
        Self {
            layout,
            _owner: Box::new(owner),
        }
    }
    /// Make an array that owns `values`, with the shape `shape` in column-major (Fortran) order.
    ///
    /// # Panics
    /// If the number of values is not the product of `shape`, or `shape` does not have 1 to
    /// [MAX_NDIM] axes.
    pub fn from_vec<T: Scalar + 'static>(values: Vec<T>, shape: &[usize]) -> Self {
        assert_eq!(
            values.len(),
            shape.iter().product::<usize>(),
            "the number of values must be the product of the shape"
        );
        let layout =
            RawLayout::valid_col_major(values.as_ptr() as *const u8, shape, T::scalar_type());
        // SAFETY: `values` has one value for each index of `shape`, and its data is on the heap,
        // so it does not move when `values` moves into the array
        unsafe { Self::from_layout(values, layout) }
    }
    /// A view of the array, with no copy.
    pub fn view(&self) -> TensorView<'_> {
        // the data stays valid and does not change while `self` is borrowed
        TensorView {
            layout: self.layout,
            _data: PhantomData,
        }
    }
    /// The length of each axis.
    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }
    /// The stride of each axis, in bytes. The [Tensor] docs give the data layout.
    pub fn strides(&self) -> &[usize] {
        self.layout.strides()
    }
    /// The pointer to the first value. It is valid while `self` exists, and the data must not
    /// be changed through it.
    pub fn data_ptr(&self) -> *const u8 {
        self.layout.ptr
    }
    /// The scalar type of the values.
    pub fn dtype(&self) -> ScalarType {
        self.layout.dtype
    }
    /// The same data with a new shape and new strides (in bytes). There is no copy.
    ///
    /// # Panics
    /// If the new layout has a different number of values, can reach a byte past the last value
    /// of the old layout, or does not obey the layout rules.
    pub(crate) fn with_layout(mut self, shape: &[usize], strides: &[usize]) -> Self {
        self.layout = self.layout.with_layout(shape, strides);
        self
    }
    /// Keep only the first `len` indices of the axis `axis`. There is no copy: the strides do not
    /// change, so each index that stays gives the same value.
    ///
    /// # Panics
    /// If `len` is larger than the length of the axis.
    pub(crate) fn truncate_axis(mut self, axis: usize, len: usize) -> Self {
        self.layout = self.layout.truncate_axis(axis, len);
        self
    }
    /// A reference to the value at `index`, with one value for each axis, for example
    /// `tensor.index::<f64>([0, i, 0])`. The scalar type must be `T`.
    ///
    /// # Panics
    /// If `index` does not have one value for each axis, or a value is out of range.
    pub fn index<T: Scalar>(&self, index: impl AsRef<[usize]>) -> Result<&T, DiffsolRtError> {
        self.view().index(index)
    }
    /// The column at `index`: the values `[:, index[0], index[1], ...]`, as one slice of
    /// `shape[0]` values, with no copy. `index` has one value for each axis after the first. The
    /// scalar type must be `T`.
    ///
    /// For a vector `[n, nbatch]`, `column(&[b])` is lane `b`. For a matrix
    /// `[nrows, ncols, nbatch]`, `column(&[j, b])` is column `j` of lane `b`. For a 1D array,
    /// `column(&[])` is all the values.
    ///
    /// # Panics
    /// If `index` does not have one value for each axis after the first, or a value is out of
    /// range.
    pub fn column<T: Scalar>(&self, index: &[usize]) -> Result<&[T], DiffsolRtError> {
        self.view().column(index)
    }
    /// All the columns (see [Tensor::column]), in column-major order: the index on axis 1
    /// changes fastest. Together, they give all the values in column-major order. The scalar
    /// type must be `T`.
    pub fn columns<T: Scalar>(&self) -> Result<impl Iterator<Item = &[T]> + '_, DiffsolRtError> {
        self.view().columns()
    }
}

impl<'a> From<&'a Tensor> for TensorView<'a> {
    fn from(tensor: &'a Tensor) -> Self {
        tensor.view()
    }
}

/// A view of an N-D array of `f32` or `f64` values that the view does not own. The inputs of
/// this crate are `TensorView` values.
///
/// A `TensorView` is small and `Copy`, and it is only on the stack: the shape and the strides are
/// fixed-size arrays. It has the same data layout and the same layout rules as [Tensor] (strides
/// in bytes, column-major order, lane axis last, adjacent values in each column). So a view of a
/// row-major array is not possible: make a column-major (Fortran-ordered) copy first. The lifetime
/// `'a` is the lifetime of the data.
///
/// A `TensorView` has the C layout of [RawLayout], so a C callback that gets a
/// `const TensorView *` can read the fields `ptr`, `dtype`, `ndim`, `shape` and `strides`.
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct TensorView<'a> {
    layout: RawLayout,
    _data: PhantomData<&'a [u8]>,
}

impl<'a> TensorView<'a> {
    /// Make a view of data, with a shape and strides (in bytes) that obey the layout rules. The
    /// [Tensor] docs give the data layout and the rules.
    ///
    /// # Errors
    /// If the layout does not obey the layout rules: for example, the strides of a row-major
    /// array, more than [MAX_NDIM] axes, or data that is not aligned.
    ///
    /// # Safety
    /// For each index in `shape`, `ptr` plus the byte offset from `strides` must point to a valid
    /// value of the type `dtype`. The data must stay valid, and must not change, during `'a`.
    pub unsafe fn new(
        ptr: *const u8,
        shape: &[usize],
        strides: &[usize],
        dtype: ScalarType,
    ) -> Result<Self, DiffsolRtError> {
        Ok(Self {
            layout: RawLayout::new(ptr, shape, strides, dtype)?,
            _data: PhantomData,
        })
    }
    /// Make a view of data with the shape `shape` in column-major (Fortran) order.
    ///
    /// # Panics
    /// If `shape` does not have 1 to [MAX_NDIM] axes, or `ptr` is not aligned for `dtype`.
    ///
    /// # Safety
    /// `ptr` must point to `product(shape)` valid, contiguous values of the type `dtype`. The data
    /// must stay valid, and must not change, during `'a`.
    pub unsafe fn new_col_major(ptr: *const u8, shape: &[usize], dtype: ScalarType) -> Self {
        Self {
            layout: RawLayout::valid_col_major(ptr, shape, dtype),
            _data: PhantomData,
        }
    }
    /// Make a view of `values`, with the shape `shape` in column-major (Fortran) order.
    ///
    /// # Panics
    /// If the number of values is not the product of `shape`, or `shape` does not have 1 to
    /// [MAX_NDIM] axes.
    pub fn from_slice<T: Scalar>(values: &'a [T], shape: &[usize]) -> Self {
        assert_eq!(
            values.len(),
            shape.iter().product::<usize>(),
            "the number of values must be the product of the shape"
        );
        // SAFETY: `values` has one aligned value for each index of `shape`, and the borrow keeps
        // it valid and unchanged during `'a`
        unsafe { Self::new_col_major(values.as_ptr() as *const u8, shape, T::scalar_type()) }
    }
    /// The length of each axis.
    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }
    /// The stride of each axis, in bytes. The [Tensor] docs give the data layout.
    pub fn strides(&self) -> &[usize] {
        self.layout.strides()
    }
    /// The pointer to the first value. It is valid during `'a`, and the data must not be changed
    /// through it.
    pub fn data_ptr(&self) -> *const u8 {
        self.layout.ptr
    }
    /// The scalar type of the values.
    pub fn dtype(&self) -> ScalarType {
        self.layout.dtype
    }
    /// Check that the view has `expected` axes.
    pub(crate) fn expect_ndim(&self, expected: usize) -> Result<(), DiffsolRtError> {
        let ndim = self.layout.ndim;
        if ndim != expected {
            return Err(layout_error(format!(
                "Expected a {expected}D array, got a {ndim}D array"
            )));
        }
        Ok(())
    }
    /// A reference to the value at `index`, with one value for each axis. The scalar type must be
    /// `T`.
    ///
    /// # Panics
    /// If `index` does not have one value for each axis, or a value is out of range.
    pub fn index<T: Scalar>(&self, index: impl AsRef<[usize]>) -> Result<&'a T, DiffsolRtError> {
        // SAFETY: the constructor guarantees the data during `'a`
        unsafe { self.layout.index(index.as_ref()) }
    }
    /// The column at `index`: the values `[:, index[0], index[1], ...]`, as one slice of
    /// `shape[0]` values, with no copy. [Tensor::column] gives the details.
    ///
    /// # Panics
    /// If `index` does not have one value for each axis after the first, or a value is out of
    /// range.
    pub fn column<T: Scalar>(&self, index: &[usize]) -> Result<&'a [T], DiffsolRtError> {
        // SAFETY: the constructor guarantees the data during `'a`
        unsafe { self.layout.column(index) }
    }
    /// All the columns, in column-major order. [Tensor::columns] gives the details.
    pub fn columns<T: Scalar>(&self) -> Result<impl Iterator<Item = &'a [T]> + 'a, DiffsolRtError> {
        // SAFETY: the constructor guarantees the data during `'a`
        unsafe { self.layout.columns() }
    }
}

impl TensorView<'_> {
    /// A view of the layout `layout`.
    ///
    /// # Safety
    /// As for [TensorView::new], for the data of `layout`.
    pub(crate) unsafe fn from_layout(layout: RawLayout) -> Self {
        Self {
            layout,
            _data: PhantomData,
        }
    }
}

/// A mutable view of an N-D array of `f32` or `f64` values that the view does not own.
///
/// It has the same data layout and the same layout rules as [TensorView]. Also, no two indices
/// give the same value (there is no stride of 0 on an axis with more than one value), so each
/// value can be changed through the view. The C callbacks of [crate::ode_callbacks] write their
/// outputs through a `TensorViewMut`.
///
/// A `TensorViewMut` has the C layout of [RawLayout]. The C type of `ptr` is `const uint8_t *`:
/// a C callback casts it to `double *` (or `float *`) to write the values.
#[repr(transparent)]
#[derive(Debug)]
pub struct TensorViewMut<'a> {
    layout: RawLayout,
    _data: PhantomData<&'a mut [u8]>,
}

impl<'a> TensorViewMut<'a> {
    /// Make a mutable view of data, with a shape and strides (in bytes) that obey the layout
    /// rules, and no two indices that give the same value.
    ///
    /// # Errors
    /// If the layout does not obey the rules.
    ///
    /// # Safety
    /// For each index in `shape`, `ptr` plus the byte offset from `strides` must point to a valid
    /// value of the type `dtype`. The data must stay valid during `'a`, and nothing else can read
    /// or change it during `'a`.
    pub unsafe fn new(
        ptr: *mut u8,
        shape: &[usize],
        strides: &[usize],
        dtype: ScalarType,
    ) -> Result<Self, DiffsolRtError> {
        let layout = RawLayout::new(ptr, shape, strides, dtype)?;
        layout.check_no_overlap()?;
        Ok(Self {
            layout,
            _data: PhantomData,
        })
    }
    /// A mutable view of the layout `layout`.
    ///
    /// # Panics
    /// If two indices of `layout` give the same value.
    ///
    /// # Safety
    /// As for [TensorViewMut::new], for the data of `layout`, which must allow writes.
    pub(crate) unsafe fn from_layout(layout: RawLayout) -> Self {
        layout
            .check_no_overlap()
            .unwrap_or_else(|err| panic!("invalid mutable tensor layout: {err}"));
        Self {
            layout,
            _data: PhantomData,
        }
    }
    /// Make a mutable view of `values`, with the shape `shape` in column-major (Fortran) order.
    ///
    /// # Panics
    /// If the number of values is not the product of `shape`, or `shape` does not have 1 to
    /// [MAX_NDIM] axes.
    pub fn from_slice_mut<T: Scalar>(values: &'a mut [T], shape: &[usize]) -> Self {
        assert_eq!(
            values.len(),
            shape.iter().product::<usize>(),
            "the number of values must be the product of the shape"
        );
        let layout =
            RawLayout::valid_col_major(values.as_mut_ptr() as *const u8, shape, T::scalar_type());
        // SAFETY: `values` has one aligned value for each index of `shape`, a column-major layout
        // has no overlap, and the mutable borrow gives the only access during `'a`
        unsafe { Self::from_layout(layout) }
    }
    /// A view of the same data, which borrows `self`.
    pub fn as_view(&self) -> TensorView<'_> {
        TensorView {
            layout: self.layout,
            _data: PhantomData,
        }
    }
    /// The length of each axis.
    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }
    /// The stride of each axis, in bytes. The [Tensor] docs give the data layout.
    pub fn strides(&self) -> &[usize] {
        self.layout.strides()
    }
    /// The pointer to the first value. It is valid during `'a`.
    pub fn data_ptr(&mut self) -> *mut u8 {
        self.layout.ptr as *mut u8
    }
    /// The scalar type of the values.
    pub fn dtype(&self) -> ScalarType {
        self.layout.dtype
    }
    /// A reference to the value at `index`. [TensorView::index] gives the details.
    ///
    /// # Panics
    /// If `index` does not have one value for each axis, or a value is out of range.
    pub fn index<T: Scalar>(&self, index: impl AsRef<[usize]>) -> Result<&T, DiffsolRtError> {
        // SAFETY: the constructor guarantees the data while `self` is borrowed
        unsafe { self.layout.index(index.as_ref()) }
    }
    /// The column at `index`. [Tensor::column] gives the details.
    ///
    /// # Panics
    /// If `index` does not have one value for each axis after the first, or a value is out of
    /// range.
    pub fn column<T: Scalar>(&self, index: &[usize]) -> Result<&[T], DiffsolRtError> {
        // SAFETY: the constructor guarantees the data while `self` is borrowed
        unsafe { self.layout.column(index) }
    }
    /// A mutable reference to the value at `index`, with one value for each axis. The scalar
    /// type must be `T`.
    ///
    /// # Panics
    /// If `index` does not have one value for each axis, or a value is out of range.
    pub fn index_mut<T: Scalar>(
        &mut self,
        index: impl AsRef<[usize]>,
    ) -> Result<&mut T, DiffsolRtError> {
        let ptr = self.layout.index_ptr::<T>(index.as_ref())? as *mut T;
        // SAFETY: the constructor guarantees a valid value that allows writes, and the only
        // access during `'a`. The mutable borrow of `self` gives one reference at a time
        Ok(unsafe { &mut *ptr })
    }
    /// The column at `index`, as one mutable slice of `shape[0]` values. [Tensor::column] gives
    /// the details.
    ///
    /// # Panics
    /// If `index` does not have one value for each axis after the first, or a value is out of
    /// range.
    pub fn column_mut<T: Scalar>(&mut self, index: &[usize]) -> Result<&mut [T], DiffsolRtError> {
        let (ptr, len) = self.layout.column_ptr::<T>(index)?;
        if len == 0 {
            return Ok(&mut []);
        }
        // SAFETY: as for `index_mut`, for the `len` adjacent, aligned values of the column
        Ok(unsafe { std::slice::from_raw_parts_mut(ptr as *mut T, len) })
    }
}

#[cfg(test)]
mod tests {
    use super::{FromTensor, Tensor, TensorLayout, TensorView, TensorViewMut, ToTensor};

    use diffsol::{
        Context, DenseMatrix, FaerContext, FaerMat, FaerVec, NalgebraContext, NalgebraMat,
        NalgebraVec, Vector,
    };

    /// Lane `b`, element `i` of a test vector.
    fn vector_value(b: usize, i: usize) -> f64 {
        10.0 * b as f64 + i as f64
    }

    /// The `f64` value of `tensor` at `index`.
    fn at(tensor: &Tensor, index: &[usize]) -> f64 {
        *tensor.index::<f64>(index).unwrap()
    }

    /// All the `f64` values of `tensor`, in column-major order of its shape (the first axis
    /// changes fastest).
    fn all_values(tensor: &Tensor) -> Vec<f64> {
        let mut indices = vec![vec![]];
        for &n in tensor.shape().iter().rev() {
            indices = indices
                .into_iter()
                .flat_map(|index: Vec<usize>| {
                    (0..n).map(move |i| [vec![i], index.clone()].concat())
                })
                .collect();
        }
        indices.iter().map(|index| at(tensor, index)).collect()
    }

    /// Lane `b`, row `i`, column `j` of a test matrix.
    fn matrix_value(b: usize, i: usize, j: usize) -> f64 {
        100.0 * b as f64 + 10.0 * i as f64 + j as f64
    }

    fn check_vector_round_trip<V>(ctx: V::C, nbatch: usize)
    where
        V: Vector<T = f64> + ToTensor + FromTensor<Context = V::C>,
    {
        let n = 3;
        let ctx = ctx.clone_with_nbatch(nbatch).unwrap();
        let values = (0..nbatch)
            .flat_map(|b| (0..n).map(move |i| vector_value(b, i)))
            .collect();
        let v = V::from_vec(values, ctx.clone());
        let tensor = v.clone().to_tensor();
        assert_eq!(tensor.shape(), &[n, nbatch]);
        for b in 0..nbatch {
            let lane: Vec<f64> = (0..n).map(|i| vector_value(b, i)).collect();
            assert_eq!(
                (0..n).map(|i| at(&tensor, &[i, b])).collect::<Vec<_>>(),
                lane
            );
        }
        let back = V::from_tensor(tensor.view(), &ctx).unwrap();
        assert_eq!(back.context().nbatch(), nbatch);
        assert_eq!(back.clone_as_vec(), v.clone_as_vec());
    }

    fn check_matrix_round_trip<M>(ctx: M::C, nbatch: usize)
    where
        M: DenseMatrix<T = f64> + ToTensor + FromTensor<Context = M::C>,
    {
        let (nrows, ncols) = (2, 3);
        let ctx = ctx.clone_with_nbatch(nbatch).unwrap();
        // diffsol stores lane `b`, row `i`, column `j` at `(b * ncols + j) * nrows + i`
        let mut values = vec![0.0; nbatch * nrows * ncols];
        for b in 0..nbatch {
            for i in 0..nrows {
                for j in 0..ncols {
                    values[(b * ncols + j) * nrows + i] = matrix_value(b, i, j);
                }
            }
        }
        let m = M::from_vec(nrows, ncols, values, ctx.clone());
        let tensor = m.to_tensor();
        assert_eq!(tensor.shape(), &[nrows, ncols, nbatch]);
        for b in 0..nbatch {
            for i in 0..nrows {
                for j in 0..ncols {
                    assert_eq!(at(&tensor, &[i, j, b]), matrix_value(b, i, j));
                }
            }
        }
        let back = M::from_tensor(tensor.view(), &ctx).unwrap();
        let again = back.to_tensor();
        assert_eq!(all_values(&again), all_values(&tensor));
    }

    #[test]
    fn vectors_round_trip_with_a_lane_axis() {
        for nbatch in [1, 3] {
            check_vector_round_trip::<NalgebraVec<f64>>(NalgebraContext::default(), nbatch);
            check_vector_round_trip::<FaerVec<f64>>(FaerContext::default(), nbatch);
        }
    }

    #[test]
    fn dense_matrices_round_trip_with_a_lane_axis() {
        for nbatch in [1, 3] {
            check_matrix_round_trip::<NalgebraMat<f64>>(NalgebraContext::default(), nbatch);
            check_matrix_round_trip::<FaerMat<f64>>(FaerContext::default(), nbatch);
        }
    }

    #[test]
    fn one_lane_broadcasts_and_other_lane_counts_are_rejected() {
        let ctx = NalgebraContext::default().clone_with_nbatch(3).unwrap();
        let one_lane = Tensor::from_vec(vec![1.0, 2.0], &[2, 1]);
        let v = NalgebraVec::<f64>::from_tensor(one_lane.view(), &ctx).unwrap();
        assert_eq!(v.context().nbatch(), 3);
        assert_eq!(v.clone_as_vec(), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);

        // a matrix with one lane also gets a copy of its values in each lane
        let one_lane = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2, 1]);
        let m = NalgebraMat::<f64>::from_tensor(one_lane.view(), &ctx).unwrap();
        assert_eq!(diffsol::Matrix::context(&m).nbatch(), 3);
        let all = all_values(&m.to_tensor());
        assert_eq!(all, [1.0, 2.0, 3.0, 4.0].repeat(3));

        let two_lanes = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let err = NalgebraVec::<f64>::from_tensor(two_lanes.view(), &ctx)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Expected 1 or 3 lanes, got 2"), "{err}");

        let no_lane = TensorView::from_slice(&[1.0, 2.0], &[2]);
        let err = NalgebraVec::<f64>::from_tensor(no_lane, &ctx)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Expected a 2D array, got a 1D array"), "{err}");
    }

    #[test]
    fn shared_tensors_keep_the_owner_alive() {
        struct Owner {
            values: Vec<f64>,
        }
        let owner = std::sync::Arc::new(Owner {
            values: vec![1.0, 2.0, 3.0],
        });
        let tensor = super::shared_tensor(owner.clone(), |o| &o.values);
        assert_eq!(tensor.data_ptr(), owner.values.as_ptr() as *const u8);
        assert_eq!(std::sync::Arc::strong_count(&owner), 2);
        drop(owner);
        assert_eq!(all_values(&tensor), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn truncate_axis_keeps_the_first_indices() {
        // `[2, 3, 2]`: keep the first 2 columns of each lane
        let values: Vec<f64> = (0..12).map(f64::from).collect();
        let tensor = Tensor::from_vec(values, &[2, 3, 2]).truncate_axis(1, 2);
        assert_eq!(tensor.shape(), &[2, 2, 2]);
        assert_eq!(
            all_values(&tensor),
            vec![0.0, 1.0, 2.0, 3.0, 6.0, 7.0, 8.0, 9.0]
        );
    }

    #[test]
    #[should_panic(expected = "cannot keep 4 indices of an axis of length 3")]
    fn truncate_axis_cannot_make_an_axis_longer() {
        Tensor::from_vec(vec![0.0f64; 3], &[3]).truncate_axis(0, 4);
    }

    #[test]
    fn with_layout_keeps_the_values() {
        // `[6, 2]` column-major, with the first axis split into `[3, 2]`
        let values: Vec<f64> = (0..12).map(f64::from).collect();
        let s = std::mem::size_of::<f64>();
        let tensor =
            Tensor::from_vec(values.clone(), &[6, 2]).with_layout(&[3, 2, 2], &[s, 3 * s, 6 * s]);
        assert_eq!(tensor.shape(), &[3, 2, 2]);
        assert_eq!(all_values(&tensor), values);
    }

    #[test]
    fn column_gives_contiguous_columns_and_index_uses_the_strides() {
        // a column-major `[2, 2]` view with a gap of one value after each column
        let data = [1.0f64, 2.0, -1.0, 3.0, 4.0, -1.0];
        let s = std::mem::size_of::<f64>();
        // SAFETY: `data` has a value at each index of the layout, and lives longer than `view`
        let view = unsafe {
            TensorView::new(
                data.as_ptr() as *const u8,
                &[2, 2],
                &[s, 3 * s],
                super::ScalarType::F64,
            )
        }
        .unwrap();
        assert_eq!(view.column::<f64>(&[0]).unwrap(), &[1.0, 2.0]);
        assert_eq!(view.column::<f64>(&[1]).unwrap(), &[3.0, 4.0]);
        let columns: Vec<&[f64]> = view.columns::<f64>().unwrap().collect();
        assert_eq!(columns, vec![&[1.0, 2.0][..], &[3.0, 4.0][..]]);
        assert_eq!(*view.index::<f64>([1, 1]).unwrap(), 4.0);
        let error = view.column::<f32>(&[0]).unwrap_err().to_string();
        assert!(error.contains("Data type mismatch"), "{error}");
        assert!(view.columns::<f32>().is_err());
        assert!(view.index::<f32>([0, 0]).is_err());
    }

    #[test]
    fn views_reject_layouts_without_contiguous_columns() {
        let data = [1.0f64, 2.0, 3.0, 4.0];
        let ptr = data.as_ptr() as *const u8;
        let s = std::mem::size_of::<f64>();
        let f64_type = super::ScalarType::F64;
        let error = |shape: &[usize], strides: &[usize], ptr: *const u8| {
            // SAFETY: the layouts are rejected, so no view reads the data
            unsafe { TensorView::new(ptr, shape, strides, f64_type) }
                .unwrap_err()
                .to_string()
        };
        // a row-major `[2, 2]` view
        let err = error(&[2, 2], &[2 * s, s], ptr);
        assert!(
            err.contains("The values of a column are not adjacent"),
            "{err}"
        );
        // a pointer that is not aligned, and a stride that is not a multiple of the value size
        let err = error(&[2], &[s], ptr.wrapping_add(1));
        assert!(err.contains("The data is not aligned"), "{err}");
        let err = error(&[1, 2], &[s, s + 1], ptr);
        assert!(err.contains("The data is not aligned"), "{err}");
        // 4 axes, or a stride count that does not agree with the shape
        let err = error(&[1, 1, 1, 1], &[s; 4], ptr);
        assert!(err.contains("A tensor has 1 to 3 axes, got 4"), "{err}");
        let err = error(&[2, 2], &[s], ptr);
        assert!(err.contains("Expected 2 strides, got 1"), "{err}");
        // the stride of an axis of length 1 is never used, so it can have any value
        // SAFETY: `data` has a value at each index of the layout, and lives longer than `view`
        let view = unsafe { TensorView::new(ptr, &[1, 2], &[3, 2 * s], f64_type) }.unwrap();
        assert_eq!(view.column::<f64>(&[1]).unwrap(), &[3.0]);
    }

    #[test]
    fn views_from_slices_have_1_to_3_axes() {
        let values: Vec<f64> = (0..12).map(f64::from).collect();
        let s = std::mem::size_of::<f64>();
        let v1 = TensorView::from_slice(&values, &[12]);
        assert_eq!(v1.shape(), &[12]);
        assert_eq!(v1.strides(), &[s]);
        let v2 = TensorView::from_slice(&values, &[3, 4]);
        assert_eq!(v2.shape(), &[3, 4]);
        assert_eq!(v2.strides(), &[s, 3 * s]);
        assert_eq!(*v2.index::<f64>([2, 1]).unwrap(), 5.0);
        let v3 = TensorView::from_slice(&values, &[2, 3, 2]);
        assert_eq!(v3.shape(), &[2, 3, 2]);
        assert_eq!(v3.strides(), &[s, 2 * s, 6 * s]);
        assert_eq!(*v3.index::<f64>([1, 2, 1]).unwrap(), 11.0);
        v3.expect_ndim(3).unwrap();
        assert!(v3.expect_ndim(2).is_err());
    }

    #[test]
    #[should_panic(expected = "A tensor has 1 to 3 axes, got 4")]
    fn views_from_slices_with_4_axes_panic() {
        let values = [0.0f64; 16];
        TensorView::from_slice(&values, &[2, 2, 2, 2]);
    }

    #[test]
    fn view_of_an_array_has_the_same_layout_and_values() {
        let ctx = FaerContext::default().clone_with_nbatch(2).unwrap();
        let m = FaerMat::<f64>::from_vec(2, 3, (0..12).map(f64::from).collect(), ctx);
        let tensor = m.to_tensor();
        let view = tensor.view();
        assert_eq!(view.shape(), tensor.shape());
        assert_eq!(view.strides(), tensor.strides());
        assert_eq!(view.dtype(), tensor.dtype());
        assert_eq!(view.data_ptr(), tensor.data_ptr());
        // the columns do not contain the padding of faer, which is not initialized
        let view_columns: Vec<&[f64]> = view.columns().unwrap().collect();
        let tensor_columns: Vec<&[f64]> = tensor.columns().unwrap().collect();
        assert_eq!(view_columns, tensor_columns);
        assert_eq!(
            all_values(&tensor),
            (0..12).map(f64::from).collect::<Vec<_>>()
        );
        let from: TensorView<'_> = (&tensor).into();
        assert_eq!(from.shape(), tensor.shape());
    }

    #[test]
    fn vectors_convert_from_views_with_padded_lanes() {
        // a `[2, 3]` view with a gap of two values after each lane
        let data = [1.0f64, 2.0, 0.0, 0.0, 10.0, 20.0, 0.0, 0.0, 100.0, 200.0];
        let s = std::mem::size_of::<f64>();
        // SAFETY: `data` has a value at each index of the layout, and lives longer than `view`
        let view = unsafe {
            TensorView::new(
                data.as_ptr() as *const u8,
                &[2, 3],
                &[s, 4 * s],
                super::ScalarType::F64,
            )
        }
        .unwrap();
        let ctx = NalgebraContext::default().clone_with_nbatch(3).unwrap();
        let v = NalgebraVec::<f64>::from_tensor(view, &ctx).unwrap();
        assert_eq!(v.clone_as_vec(), vec![1.0, 2.0, 10.0, 20.0, 100.0, 200.0]);
    }

    #[test]
    fn f32_views_convert_to_f64_vectors() {
        let data = [1.0f32, 2.0];
        let view = TensorView::from_slice(&data, &[2, 1]);
        let v = NalgebraVec::<f64>::from_tensor(view, &NalgebraContext::default()).unwrap();
        assert_eq!(v.clone_as_vec(), vec![1.0, 2.0]);
    }

    #[test]
    fn columns_skip_the_padding_of_faer_storage() {
        // faer pads the columns of its storage, so the lanes of a faer vector are not adjacent
        let ctx = FaerContext::default().clone_with_nbatch(2).unwrap();
        let v = FaerVec::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], ctx);
        let tensor = v.to_tensor();
        assert!(tensor.strides()[1] >= 3 * std::mem::size_of::<f64>());
        let columns: Vec<&[f64]> = tensor.columns().unwrap().collect();
        assert_eq!(columns, vec![&[1.0, 2.0, 3.0][..], &[4.0, 5.0, 6.0][..]]);
        assert_eq!(tensor.column::<f64>(&[1]).unwrap(), &[4.0, 5.0, 6.0]);

        // a `[2, 3, 2]` matrix: column `j` of lane `b` is number `b * 3 + j`
        let m = FaerMat::<f64>::from_vec(2, 3, (0..12).map(f64::from).collect(), ctx);
        let tensor = m.to_tensor();
        let columns: Vec<Vec<f64>> = tensor.columns().unwrap().map(<[f64]>::to_vec).collect();
        let expected: Vec<Vec<f64>> = (0..6)
            .map(|c| vec![2.0 * c as f64, 2.0 * c as f64 + 1.0])
            .collect();
        assert_eq!(columns, expected);
        assert_eq!(tensor.column::<f64>(&[2, 1]).unwrap(), &[10.0, 11.0]);
    }

    #[test]
    fn empty_axes_give_empty_columns_or_no_columns() {
        // no rows: each column is empty
        let no_rows = Tensor::from_vec(Vec::<f64>::new(), &[0, 2]);
        let columns: Vec<&[f64]> = no_rows.columns().unwrap().collect();
        assert_eq!(columns, vec![&[] as &[f64]; 2]);
        assert_eq!(no_rows.column::<f64>(&[1]).unwrap(), &[] as &[f64]);
        // no lanes: there are no columns
        let no_lanes = Tensor::from_vec(Vec::<f64>::new(), &[3, 0]);
        assert_eq!(no_lanes.columns::<f64>().unwrap().count(), 0);
    }

    #[test]
    fn metadata_gives_the_layout() {
        let tensor = Tensor::from_vec((0..6).map(f64::from).collect(), &[2, 3]);
        let s = std::mem::size_of::<f64>();
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.strides(), &[s, 2 * s]);
        assert_eq!(tensor.dtype(), super::ScalarType::F64);
        // the value at (1, 2) is at the byte offset 1 * s + 2 * 2s from the data pointer
        let value = unsafe { *(tensor.data_ptr().add(s + 4 * s) as *const f64) };
        assert_eq!(value, 5.0);
    }

    #[test]
    fn batch_lanes_broadcast_with_strides_of_zero() {
        let s = std::mem::size_of::<f64>();
        let ctx = |nbatch| FaerContext::default().clone_with_nbatch(nbatch).unwrap();
        let lanes = |nbatch: usize| -> Vec<f64> {
            (0..nbatch)
                .flat_map(|b| (0..2).map(move |i| vector_value(b, i)))
                .collect()
        };
        // 2 outer lanes with 3 inner lanes each: 6 output lanes
        let (ninner, nouter) = (3, 2);
        let full = FaerVec::<f64>::from_vec(lanes(6), ctx(6));
        let outer = FaerVec::<f64>::from_vec(lanes(2), ctx(2));
        let one = FaerVec::<f64>::from_vec(lanes(1), ctx(1));
        for (v, expected_lane) in [
            (
                &full,
                Box::new(|k, b| b * ninner + k) as Box<dyn Fn(usize, usize) -> usize>,
            ),
            (&outer, Box::new(|_, b| b)),
            (&one, Box::new(|_, _| 0)),
        ] {
            let layout = v.tensor_layout().batch_lanes(ninner, nouter);
            // SAFETY: the layout shows the data of `v`, which lives longer than `view`
            let view = unsafe { TensorView::from_layout(layout) };
            assert_eq!(view.shape(), &[2, ninner, nouter]);
            assert_eq!(view.strides()[0], s);
            for b in 0..nouter {
                for k in 0..ninner {
                    let lane = expected_lane(k, b);
                    let expected = [vector_value(lane, 0), vector_value(lane, 1)];
                    assert_eq!(view.column::<f64>(&[k, b]).unwrap(), &expected);
                }
            }
        }
        let c = full.tensor_layout().strides()[1];
        let strides = |v: &FaerVec<f64>| v.tensor_layout().batch_lanes(ninner, nouter).strides;
        assert_eq!(strides(&full), [s, c, ninner * c]);
        assert_eq!(
            strides(&outer)[1..],
            [0, outer.tensor_layout().strides()[1]]
        );
        assert_eq!(strides(&one)[1..], [0, 0]);
    }

    #[test]
    #[should_panic(expected = "cannot broadcast 4 lanes into 3 x 2 lanes")]
    fn batch_lanes_reject_other_lane_counts() {
        let ctx = NalgebraContext::default().clone_with_nbatch(4).unwrap();
        let v = NalgebraVec::<f64>::zeros(2, ctx);
        v.tensor_layout().batch_lanes(3, 2);
    }

    #[test]
    fn mutable_views_write_each_lane_of_a_padded_vector() {
        use super::TensorLayoutMut;
        let ctx = FaerContext::default().clone_with_nbatch(4).unwrap();
        let mut v = FaerVec::<f64>::zeros(3, ctx);
        let layout = v.tensor_layout_mut().batch_lanes(2, 2);
        // SAFETY: the layout shows the data of `v`, which is mutably borrowed while `view` lives
        let mut view = unsafe { TensorViewMut::from_layout(layout) };
        for b in 0..2 {
            for k in 0..2 {
                let lane = (b * 2 + k) as f64;
                view.column_mut::<f64>(&[k, b]).unwrap().copy_from_slice(&[
                    lane,
                    lane + 0.5,
                    lane + 0.25,
                ]);
            }
        }
        *view.index_mut::<f64>([2, 1, 1]).unwrap() = -1.0;
        assert_eq!(*view.as_view().index::<f64>([2, 1, 1]).unwrap(), -1.0);
        let expected: Vec<f64> = (0..4)
            .flat_map(|l| [l as f64, l as f64 + 0.5, l as f64 + 0.25])
            .collect::<Vec<_>>()
            .into_iter()
            .enumerate()
            .map(|(i, x)| if i == 11 { -1.0 } else { x })
            .collect();
        assert_eq!(v.clone_as_vec(), expected);
    }

    #[test]
    fn mutable_views_reject_overlapping_layouts() {
        let mut data = [0.0f64; 4];
        let s = std::mem::size_of::<f64>();
        let ptr = data.as_mut_ptr() as *mut u8;
        let f64_type = super::ScalarType::F64;
        // SAFETY: the layouts are rejected, so no view reads or writes the data
        let err = unsafe { TensorViewMut::new(ptr, &[2, 2], &[s, 0], f64_type) }.unwrap_err();
        assert!(err.to_string().contains("overlap"), "{err}");
        let err = unsafe { TensorViewMut::new(ptr, &[2, 2], &[s, s], f64_type) }.unwrap_err();
        assert!(err.to_string().contains("overlap"), "{err}");
        // a stride of 0 on an axis of length 1 is never used
        // SAFETY: `data` has a value at each index of the layout, and lives longer than `view`
        let mut view =
            unsafe { TensorViewMut::new(ptr, &[2, 1, 2], &[s, 0, 2 * s], f64_type) }.unwrap();
        view.column_mut::<f64>(&[0, 1]).unwrap().fill(3.0);
        let view = TensorViewMut::from_slice_mut(&mut data, &[4]);
        assert_eq!(view.column::<f64>(&[]).unwrap(), &[0.0, 0.0, 3.0, 3.0]);
    }

    #[test]
    fn vec_converts_to_a_1d_array() {
        let tensor = vec![1.0f64, 2.0, 3.0].to_tensor();
        assert_eq!(tensor.shape(), &[3]);
        assert_eq!(tensor.column::<f64>(&[]).unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(tensor.columns::<f64>().unwrap().count(), 1);
    }
}
