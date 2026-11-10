use crate::{
    error::DiffsolRtError,
    scalar_type::{Scalar, ScalarType, ToScalarType},
};
use diffsol::{
    error::DiffsolError, Context, DenseMatrix, FaerScalar, Matrix, MatrixCommon, NalgebraScalar,
    Vector, VectorCommon,
};
use num_traits::FromPrimitive;
use std::any::Any;
use std::marker::PhantomData;
use std::sync::Arc;

/// The maximum number of axes of an [TensorView].
pub const MAX_NDIM: usize = 3;

/// Convert a value into an [Tensor] that owns it, with no copy of the data.
pub(crate) trait ToTensor {
    fn to_tensor(self) -> Tensor;
}

/// Convert an [TensorView] to a diffsol vector or dense matrix. The view can have any strides,
/// and the scalar type `f32` or `f64`.
pub(crate) trait FromTensor: Sized {
    /// The diffsol context of the new value.
    type Context;
    fn from_tensor(tensor: TensorView<'_>, ctx: &Self::Context) -> Result<Self, DiffsolRtError>;
}

/// Read the values of `tensor` in lane-major order, with `nbatch` lanes.
///
/// `lane_index(b, k)` is the index of value `k` of lane `b`, and each lane has `len` values.
/// `tensor` must have `nbatch` lanes or 1 lane on its last axis. The values of one lane are
/// copied to all `nbatch` lanes, so the result always has a value for each lane.
fn lane_values<T: FromPrimitive + Clone>(
    tensor: TensorView<'_>,
    nbatch: usize,
    len: usize,
    lane_index: impl Fn(usize, usize) -> [usize; MAX_NDIM],
) -> Result<Vec<T>, DiffsolRtError> {
    let ndim = tensor.shape().len();
    let nlanes = *tensor.shape().last().unwrap();
    if nlanes != nbatch && nlanes != 1 {
        return Err(
            DiffsolError::Other(format!("Expected 1 or {nbatch} lanes, got {nlanes}")).into(),
        );
    }
    let offsets = (0..nlanes)
        .flat_map(|b| (0..len).map(move |k| (b, k)))
        .map(|(b, k)| tensor.offset(&lane_index(b, k)[..ndim]));
    let values: Vec<T> = match tensor.dtype() {
        ScalarType::F32 => {
            let data = tensor.as_slice::<f32>()?;
            offsets.map(|i| T::from_f32(data[i]).unwrap()).collect()
        }
        ScalarType::F64 => {
            let data = tensor.as_slice::<f64>()?;
            offsets.map(|i| T::from_f64(data[i]).unwrap()).collect()
        }
    };
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
    // diffsol stores lane `b`, value `i` at `b * n + i`
    let data = lane_values(tensor, ctx.nbatch(), tensor.shape()[0], |b, i| [i, b, 0])?;
    Ok(V::from_vec(data, ctx.clone()))
}

/// Convert an `[nrows, ncols, nlanes]` array to a diffsol dense matrix with the context `ctx`.
fn dense_matrix_from<M: DenseMatrix<T: Scalar>>(
    tensor: TensorView<'_>,
    ctx: &M::C,
) -> Result<M, DiffsolRtError> {
    tensor.expect_ndim(3)?;
    // diffsol stores lane `b`, row `i`, column `j` at `(b * ncols + j) * nrows + i`
    let (nrows, ncols) = (tensor.shape()[0], tensor.shape()[1]);
    let data = lane_values(tensor, ctx.nbatch(), nrows * ncols, |b, k| {
        [k % nrows, k / nrows, b]
    })?;
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

/// The data pointer, the shape and the byte strides of a tensor.
pub(crate) type Layout = (*const u8, Vec<usize>, Vec<usize>);

/// The layout of a faer matrix.
fn faer_layout<T: Scalar + FaerScalar>(m: &faer::Mat<T>) -> Layout {
    // faer gives the strides in values; a matrix that owns its data has positive strides
    let size = std::mem::size_of::<T>();
    let strides = vec![
        m.row_stride() as usize * size,
        m.col_stride() as usize * size,
    ];
    (m.as_ptr() as *const u8, vec![m.nrows(), m.ncols()], strides)
}

/// The layout of a nalgebra matrix.
fn nalgebra_layout<T: Scalar + NalgebraScalar>(m: &nalgebra::DMatrix<T>) -> Layout {
    // nalgebra gives the strides in values
    let size = std::mem::size_of::<T>();
    let (rs, cs) = m.strides();
    (
        m.as_ptr() as *const u8,
        vec![m.nrows(), m.ncols()],
        vec![rs * size, cs * size],
    )
}

/// Give the storage of a diffsol dense matrix, an `nrows × (nbatch * ncols)` matrix where the
/// columns of lane `b` are the column block `b`, the shape `[nrows, ncols, nbatch]`.
fn matrix_lanes((ptr, shape, strides): Layout, nbatch: usize) -> Layout {
    let (nrows, ncols) = (shape[0], shape[1] / nbatch);
    let (row_stride, col_stride) = (strides[0], strides[1]);
    (
        ptr,
        vec![nrows, ncols, nbatch],
        vec![row_stride, col_stride, ncols * col_stride],
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
pub(crate) unsafe trait TensorLayout {
    /// The scalar type of the values.
    type T: Scalar;
    /// The layout of the data.
    fn tensor_layout(&self) -> Layout;
}

// SAFETY: faer and nalgebra keep the values of a vector or a matrix on the heap, and change them
// only through `&mut`
unsafe impl<T: Scalar + FaerScalar> TensorLayout for diffsol::FaerVec<T> {
    type T = T;
    fn tensor_layout(&self) -> Layout {
        faer_layout(self.inner())
    }
}

// SAFETY: as for `FaerVec`
unsafe impl<T: Scalar + NalgebraScalar> TensorLayout for diffsol::NalgebraVec<T> {
    type T = T;
    fn tensor_layout(&self) -> Layout {
        nalgebra_layout(self.inner())
    }
}

// SAFETY: as for `FaerVec`
unsafe impl<T: Scalar + FaerScalar> TensorLayout for diffsol::FaerMat<T> {
    type T = T;
    fn tensor_layout(&self) -> Layout {
        matrix_lanes(faer_layout(self.inner()), self.context().nbatch())
    }
}

// SAFETY: as for `FaerVec`
unsafe impl<T: Scalar + NalgebraScalar> TensorLayout for diffsol::NalgebraMat<T> {
    type T = T;
    fn tensor_layout(&self) -> Layout {
        matrix_lanes(nalgebra_layout(self.inner()), self.context().nbatch())
    }
}

// SAFETY: a `Vec` keeps its values on the heap, and changes them only through `&mut`
unsafe impl<T: Scalar> TensorLayout for Vec<T> {
    type T = T;
    fn tensor_layout(&self) -> Layout {
        let size = std::mem::size_of::<T>();
        (self.as_ptr() as *const u8, vec![self.len()], vec![size])
    }
}

impl<V: TensorLayout + 'static> ToTensor for V {
    fn to_tensor(self) -> Tensor {
        let (ptr, shape, strides) = self.tensor_layout();
        // SAFETY: `TensorLayout` guarantees that the data is on the heap of `self`, so it does
        // not move when `self` moves into the tensor, and that it does not change
        unsafe { Tensor::new(self, ptr, shape, strides, V::T::scalar_type()) }
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
    let (ptr, shape, strides) = part(&owner).tensor_layout();
    // SAFETY: `part` gives a value that `owner` holds (or a static value), and the `Arc`
    // allocation does not move when `owner` moves into the tensor. `TensorLayout` guarantees that
    // the data of that value is on the heap and does not change while it is borrowed
    unsafe { Tensor::new(owner, ptr, shape, strides, V::T::scalar_type()) }
}

fn scalar_size(dtype: ScalarType) -> usize {
    match dtype {
        ScalarType::F32 => std::mem::size_of::<f32>(),
        ScalarType::F64 => std::mem::size_of::<f64>(),
    }
}

/// Write the byte strides of a column-major (Fortran) array with the shape `shape` to `strides`.
fn col_major_strides(shape: &[usize], dtype: ScalarType, strides: &mut [usize]) {
    let mut stride = scalar_size(dtype);
    for (s, &n) in strides.iter_mut().zip(shape) {
        *s = stride;
        stride *= n;
    }
}

// The functions below give the shared index logic of [Tensor] and [TensorView].

/// The index in the slice of `as_slice` of the value at `index`.
fn layout_offset(shape: &[usize], strides: &[usize], dtype: ScalarType, index: &[usize]) -> usize {
    assert_eq!(index.len(), shape.len(), "wrong number of indices");
    let bytes: usize = index
        .iter()
        .zip(shape)
        .zip(strides)
        .map(|((&i, &n), &stride)| {
            assert!(i < n, "index {i} is out of range for an axis of length {n}");
            i * stride
        })
        .sum();
    bytes / scalar_size(dtype)
}

/// A slice over the data of a layout, from the first value to the last value.
///
/// # Safety
/// For each index in `shape`, `ptr` plus the byte offset from `strides` must point to a valid
/// value of the type `dtype`, which stays valid and does not change during `'a`.
unsafe fn layout_slice<'a, T: Scalar>(
    ptr: *const u8,
    shape: &[usize],
    strides: &[usize],
    dtype: ScalarType,
) -> Result<&'a [T], DiffsolRtError> {
    if dtype != T::scalar_type() {
        return Err(DiffsolError::Other("Data type mismatch".to_string()).into());
    }
    if shape.contains(&0) {
        return Ok(&[]);
    }
    let last_index = shape.iter().map(|n| n - 1).collect::<Vec<_>>();
    let last = layout_offset(shape, strides, dtype, &last_index);
    if !(ptr as usize).is_multiple_of(std::mem::align_of::<T>()) {
        return Err(DiffsolError::Other("The data is not aligned".to_string()).into());
    }
    // SAFETY: the caller guarantees a value of type `T` at each index up to `last`, valid
    // during `'a`
    Ok(unsafe { std::slice::from_raw_parts(ptr as *const T, last + 1) })
}

/// Check that `shape` has `expected` axes.
fn layout_expect_ndim(shape: &[usize], expected: usize) -> Result<(), DiffsolRtError> {
    let ndim = shape.len();
    if ndim != expected {
        return Err(DiffsolError::Other(format!(
            "Expected a {expected}D array, got a {ndim}D array"
        ))
        .into());
    }
    Ok(())
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
/// [Tensor::as_slice] gives a slice over the data, from the first value to the last value.
/// [Tensor::offset] gives the index of a value in that slice. The slice can contain values
/// that are not part of the array: for example, faer pads the columns of a matrix.
///
/// The arrays use the column-major (Fortran) order of diffsol: the first axis changes fastest, and
/// the lane axis is the last axis. The arrays that this crate makes from diffsol values have these
/// layouts (`s` is the size of one value, and `c` is the column stride):
/// - A diffsol vector is `[n, nbatch]`, with the strides `[s, c]`. Lane `b` is at the slice
///   indices `b * c / s .. b * c / s + n`, so each lane is contiguous.
/// - A diffsol dense matrix is `[nrows, ncols, nbatch]`, with the strides `[s, c, ncols * c]`.
///   Each lane is a column-major matrix.
/// - A Rust `Vec<T>`, for example `ts`, is `[n]`, with the stride `[s]`.
///
/// `c` is `nrows * s` for nalgebra. It can be larger for faer, which pads its columns.
///
/// # Ownership
/// A `Tensor` always owns the value that holds its data (the `owner`), so the data stays valid
/// while the array exists.
pub struct Tensor {
    dtype: ScalarType,
    shape: Vec<usize>,
    strides: Vec<usize>,
    ptr: *const u8,
    /// The value that holds the data. It is only kept alive, never read.
    _owner: Box<dyn Any>,
}

impl Tensor {
    /// Make an array that owns `owner`, with any shape and strides (in bytes). The [Tensor] docs
    /// give the data layout.
    ///
    /// # Safety
    /// `ptr` must point into data that `owner` holds on the heap, so that the data does not move
    /// when `owner` moves into the array. For each index in `shape`, `ptr` plus the byte offset
    /// from `strides` must point to a valid, aligned value of the type `dtype`. The data must not
    /// change while the array exists.
    pub unsafe fn new<O: 'static>(
        owner: O,
        ptr: *const u8,
        shape: Vec<usize>,
        strides: Vec<usize>,
        dtype: ScalarType,
    ) -> Self {
        assert_eq!(shape.len(), strides.len());
        Self {
            ptr,
            shape,
            strides,
            dtype,
            _owner: Box::new(owner),
        }
    }
    /// Make an array that owns `owner`, with the shape `shape` in column-major (Fortran) order.
    ///
    /// # Safety
    /// `ptr` must point to `product(shape)` valid, aligned, contiguous values of the type
    /// `dtype`, which `owner` holds on the heap. The data must not change while the array exists.
    pub(crate) unsafe fn new_col_major<O: 'static>(
        owner: O,
        ptr: *const u8,
        shape: Vec<usize>,
        dtype: ScalarType,
    ) -> Self {
        let mut strides = vec![0; shape.len()];
        col_major_strides(&shape, dtype, &mut strides);
        // SAFETY: the caller guarantees the values for a column-major layout of `shape`
        unsafe { Tensor::new(owner, ptr, shape, strides, dtype) }
    }
    /// Make an array that owns `values`, with the shape `shape` in column-major (Fortran) order.
    ///
    /// # Panics
    /// If the number of values is not the product of `shape`.
    pub fn from_vec<T: Scalar + 'static>(values: Vec<T>, shape: Vec<usize>) -> Self {
        assert_eq!(
            values.len(),
            shape.iter().product::<usize>(),
            "the number of values must be the product of the shape"
        );
        let ptr = values.as_ptr() as *const u8;
        // SAFETY: `values` has one value for each index of `shape`, and its data is on the heap,
        // so it does not move when `values` moves into the array
        unsafe { Tensor::new_col_major(values, ptr, shape, T::scalar_type()) }
    }
    /// A view of the array, with no copy.
    ///
    /// # Panics
    /// If the array has more than [MAX_NDIM] axes. The arrays that this crate makes have
    /// [MAX_NDIM] axes or fewer.
    pub fn view(&self) -> TensorView<'_> {
        // SAFETY: the layout is that of `self`, and the data stays valid and does not change
        // while `self` is borrowed
        unsafe { TensorView::new(self.ptr, &self.shape, &self.strides, self.dtype) }
    }
    /// The length of each axis.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    /// The stride of each axis, in bytes. The [Tensor] docs give the data layout.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
    /// The pointer to the first value. It is valid while `self` exists, and the data must not
    /// be changed through it.
    pub fn data_ptr(&self) -> *const u8 {
        self.ptr
    }
    /// The scalar type of the values.
    pub fn dtype(&self) -> ScalarType {
        self.dtype
    }
    /// The same data with a new shape and new strides (in bytes). There is no copy.
    ///
    /// The new layout must give only values that the old layout gives.
    pub(crate) fn with_layout(mut self, shape: Vec<usize>, strides: Vec<usize>) -> Self {
        assert_eq!(shape.len(), strides.len());
        assert_eq!(
            shape.iter().product::<usize>(),
            self.shape.iter().product::<usize>()
        );
        let last_byte = |shape: &[usize], strides: &[usize]| -> usize {
            shape
                .iter()
                .zip(strides)
                .map(|(n, s)| n.saturating_sub(1) * s)
                .sum()
        };
        // a layout that reads past the old data would make `as_slice` unsound
        assert!(last_byte(&shape, &strides) <= last_byte(&self.shape, &self.strides));
        self.shape = shape;
        self.strides = strides;
        self
    }
    /// Keep only the first `len` indices of the axis `axis`. There is no copy.
    ///
    /// # Panics
    /// If `len` is larger than the length of the axis.
    pub(crate) fn truncate_axis(mut self, axis: usize, len: usize) -> Self {
        let n = self.shape[axis];
        assert!(
            len <= n,
            "cannot keep {len} indices of an axis of length {n}"
        );
        self.shape[axis] = len;
        self
    }
    /// The index in [Tensor::as_slice] of the value at `index`.
    ///
    /// # Panics
    /// If `index` does not have one value for each axis, or a value is out of range.
    pub fn offset(&self, index: &[usize]) -> usize {
        layout_offset(&self.shape, &self.strides, self.dtype, index)
    }
    /// A slice over the data of the array, from the first value to the last value. Use
    /// [Tensor::offset] to find a value. The scalar type must be `T`.
    pub fn as_slice<T: Scalar>(&self) -> Result<&[T], DiffsolRtError> {
        // SAFETY: the owner keeps the data valid while `self` is borrowed
        unsafe { layout_slice(self.ptr, &self.shape, &self.strides, self.dtype) }
    }
    /// A reference to the value at `index`, with one value for each axis, for example
    /// `tensor.index::<f64>([0, i, 0])`. The scalar type must be `T`.
    ///
    /// # Panics
    /// If `index` does not have one value for each axis, or a value is out of range.
    pub fn index<T: Scalar>(&self, index: impl AsRef<[usize]>) -> Result<&T, DiffsolRtError> {
        let i = self.offset(index.as_ref());
        Ok(&self.as_slice::<T>()?[i])
    }
}

impl<'a> From<&'a Tensor> for TensorView<'a> {
    fn from(tensor: &'a Tensor) -> Self {
        tensor.view()
    }
}

/// A view of an N-D array of `f32` or `f64` values that the view does not own, with at most
/// [MAX_NDIM] axes. The inputs of this crate are `TensorView` values.
///
/// A `TensorView` is small and `Copy`, and it is only on the stack: the shape and the strides are
/// fixed-size arrays. It has the same data layout as [Tensor] (strides in bytes, column-major order,
/// lane axis last), and an input can have any strides. The lifetime `'a` is the lifetime of the
/// data.
#[derive(Clone, Copy, Debug)]
pub struct TensorView<'a> {
    ptr: *const u8,
    dtype: ScalarType,
    ndim: usize,
    /// Only the first `ndim` values are used.
    shape: [usize; MAX_NDIM],
    /// Only the first `ndim` values are used. In bytes.
    strides: [usize; MAX_NDIM],
    _data: PhantomData<&'a [u8]>,
}

impl<'a> TensorView<'a> {
    /// Make a view of data, with any shape and strides (in bytes). The [Tensor] docs give the data
    /// layout.
    ///
    /// # Panics
    /// If `shape` has more than [MAX_NDIM] axes, or `strides` does not have one value for each
    /// axis.
    ///
    /// # Safety
    /// For each index in `shape`, `ptr` plus the byte offset from `strides` must point to a valid,
    /// aligned value of the type `dtype`. The data must stay valid, and must not change, during
    /// `'a`.
    pub unsafe fn new(
        ptr: *const u8,
        shape: &[usize],
        strides: &[usize],
        dtype: ScalarType,
    ) -> Self {
        let ndim = shape.len();
        assert!(
            ndim <= MAX_NDIM,
            "a TensorView has at most {MAX_NDIM} axes, got {ndim}"
        );
        assert_eq!(ndim, strides.len(), "one stride for each axis is necessary");
        let mut view = Self {
            ptr,
            dtype,
            ndim,
            shape: [0; MAX_NDIM],
            strides: [0; MAX_NDIM],
            _data: PhantomData,
        };
        view.shape[..ndim].copy_from_slice(shape);
        view.strides[..ndim].copy_from_slice(strides);
        view
    }
    /// Make a view of data with the shape `shape` in column-major (Fortran) order.
    ///
    /// # Panics
    /// If `shape` has more than [MAX_NDIM] axes.
    ///
    /// # Safety
    /// `ptr` must point to `product(shape)` valid, aligned, contiguous values of the type
    /// `dtype`. The data must stay valid, and must not change, during `'a`.
    pub unsafe fn new_col_major(ptr: *const u8, shape: &[usize], dtype: ScalarType) -> Self {
        assert!(
            shape.len() <= MAX_NDIM,
            "a TensorView has at most {MAX_NDIM} axes, got {}",
            shape.len()
        );
        let mut strides = [0; MAX_NDIM];
        col_major_strides(shape, dtype, &mut strides);
        // SAFETY: the caller guarantees the values for a column-major layout of `shape`
        unsafe { Self::new(ptr, shape, &strides[..shape.len()], dtype) }
    }
    /// Make a view of `values`, with the shape `shape` in column-major (Fortran) order.
    ///
    /// # Panics
    /// If the number of values is not the product of `shape`, or `shape` has more than
    /// [MAX_NDIM] axes.
    pub fn from_slice<T: Scalar>(values: &'a [T], shape: &[usize]) -> Self {
        assert_eq!(
            values.len(),
            shape.iter().product::<usize>(),
            "the number of values must be the product of the shape"
        );
        // SAFETY: `values` has one value for each index of `shape`, and the borrow keeps it
        // valid and unchanged during `'a`
        unsafe { Self::new_col_major(values.as_ptr() as *const u8, shape, T::scalar_type()) }
    }
    /// The length of each axis.
    pub fn shape(&self) -> &[usize] {
        &self.shape[..self.ndim]
    }
    /// The stride of each axis, in bytes. The [Tensor] docs give the data layout.
    pub fn strides(&self) -> &[usize] {
        &self.strides[..self.ndim]
    }
    /// The pointer to the first value. It is valid during `'a`, and the data must not be changed
    /// through it.
    pub fn data_ptr(&self) -> *const u8 {
        self.ptr
    }
    /// The scalar type of the values.
    pub fn dtype(&self) -> ScalarType {
        self.dtype
    }
    /// Check that the view has `expected` axes.
    pub(crate) fn expect_ndim(&self, expected: usize) -> Result<(), DiffsolRtError> {
        layout_expect_ndim(self.shape(), expected)
    }
    /// The index in [TensorView::as_slice] of the value at `index`.
    ///
    /// # Panics
    /// If `index` does not have one value for each axis, or a value is out of range.
    pub fn offset(&self, index: &[usize]) -> usize {
        layout_offset(self.shape(), self.strides(), self.dtype, index)
    }
    /// A slice over the data of the view, from the first value to the last value. Use
    /// [TensorView::offset] to find a value. The scalar type must be `T`.
    pub fn as_slice<T: Scalar>(&self) -> Result<&'a [T], DiffsolRtError> {
        // SAFETY: the constructor guarantees that the data is valid during `'a`
        unsafe { layout_slice(self.ptr, self.shape(), self.strides(), self.dtype) }
    }
    /// A reference to the value at `index`, with one value for each axis. The scalar type must be
    /// `T`.
    ///
    /// # Panics
    /// If `index` does not have one value for each axis, or a value is out of range.
    pub fn index<T: Scalar>(&self, index: impl AsRef<[usize]>) -> Result<&'a T, DiffsolRtError> {
        let i = self.offset(index.as_ref());
        Ok(&self.as_slice::<T>()?[i])
    }
}

#[cfg(test)]
mod tests {
    use super::{FromTensor, Tensor, TensorView, ToTensor};

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
        let one_lane = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let v = NalgebraVec::<f64>::from_tensor(one_lane.view(), &ctx).unwrap();
        assert_eq!(v.context().nbatch(), 3);
        assert_eq!(v.clone_as_vec(), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);

        // a matrix with one lane also gets a copy of its values in each lane
        let one_lane = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2, 1]);
        let m = NalgebraMat::<f64>::from_tensor(one_lane.view(), &ctx).unwrap();
        assert_eq!(diffsol::Matrix::context(&m).nbatch(), 3);
        let all = all_values(&m.to_tensor());
        assert_eq!(all, [1.0, 2.0, 3.0, 4.0].repeat(3));

        let two_lanes = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
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
        let tensor = Tensor::from_vec(values, vec![2, 3, 2]).truncate_axis(1, 2);
        assert_eq!(tensor.shape(), &[2, 2, 2]);
        assert_eq!(
            all_values(&tensor),
            vec![0.0, 1.0, 2.0, 3.0, 6.0, 7.0, 8.0, 9.0]
        );
    }

    #[test]
    #[should_panic(expected = "cannot keep 4 indices of an axis of length 3")]
    fn truncate_axis_cannot_make_an_axis_longer() {
        Tensor::from_vec(vec![0.0f64; 3], vec![3]).truncate_axis(0, 4);
    }

    #[test]
    fn with_layout_keeps_the_values() {
        // `[6, 2]` column-major, with the first axis split into `[3, 2]`
        let values: Vec<f64> = (0..12).map(f64::from).collect();
        let s = std::mem::size_of::<f64>();
        let tensor = Tensor::from_vec(values.clone(), vec![6, 2])
            .with_layout(vec![3, 2, 2], vec![s, 3 * s, 6 * s]);
        assert_eq!(tensor.shape(), &[3, 2, 2]);
        assert_eq!(all_values(&tensor), values);
    }

    #[test]
    fn as_slice_covers_the_data_and_offset_uses_the_strides() {
        let data = [1.0f64, 2.0, 3.0, 4.0];
        // a row-major `[2, 2]` view
        let s = std::mem::size_of::<f64>();
        // SAFETY: `data` has 4 values, and lives longer than `view`
        let view = unsafe {
            TensorView::new(
                data.as_ptr() as *const u8,
                &[2, 2],
                &[2 * s, s],
                super::ScalarType::F64,
            )
        };
        assert_eq!(view.as_slice::<f64>().unwrap(), &data);
        assert_eq!(view.offset(&[0, 1]), 1);
        assert_eq!(view.offset(&[1, 0]), 2);
        assert_eq!(*view.index::<f64>([1, 0]).unwrap(), 3.0);
        let error = view.as_slice::<f32>().unwrap_err().to_string();
        assert!(error.contains("Data type mismatch"), "{error}");
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
    #[should_panic(expected = "at most 3 axes")]
    fn views_with_4_axes_panic() {
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
        assert_eq!(
            view.as_slice::<f64>().unwrap(),
            tensor.as_slice::<f64>().unwrap()
        );
        let from: TensorView<'_> = (&tensor).into();
        assert_eq!(from.shape(), tensor.shape());
    }

    #[test]
    fn vectors_convert_from_views_with_any_strides() {
        // a `[2, 3]` view of a row-major buffer: lane `b`, value `i` is at `i * 3 + b`
        let data = [1.0f64, 10.0, 100.0, 2.0, 20.0, 200.0];
        let s = std::mem::size_of::<f64>();
        // SAFETY: `data` has 6 values, and lives longer than `view`
        let view = unsafe {
            TensorView::new(
                data.as_ptr() as *const u8,
                &[2, 3],
                &[3 * s, s],
                super::ScalarType::F64,
            )
        };
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
    fn as_slice_includes_the_padding_of_faer_storage() {
        // faer pads the columns of its storage, so the lanes of a faer vector are not adjacent
        let ctx = FaerContext::default().clone_with_nbatch(2).unwrap();
        let v = FaerVec::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], ctx);
        let tensor = v.to_tensor();
        let lane_start = tensor.offset(&[0, 1]);
        assert!(lane_start >= 3);
        assert_eq!(tensor.as_slice::<f64>().unwrap().len(), lane_start + 3);
        assert_eq!(all_values(&tensor), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn metadata_gives_the_layout() {
        let tensor = Tensor::from_vec((0..6).map(f64::from).collect(), vec![2, 3]);
        let s = std::mem::size_of::<f64>();
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.strides(), &[s, 2 * s]);
        assert_eq!(tensor.dtype(), super::ScalarType::F64);
        // the value at (1, 2) is at the byte offset 1 * s + 2 * 2s from the data pointer
        let value = unsafe { *(tensor.data_ptr().add(s + 4 * s) as *const f64) };
        assert_eq!(value, 5.0);
    }

    #[test]
    fn vec_converts_to_a_1d_array() {
        let host = vec![1.0f64, 2.0, 3.0].to_tensor();
        let values = host.as_slice::<f64>().unwrap();
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }
}
