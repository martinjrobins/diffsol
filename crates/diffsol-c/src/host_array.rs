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

/// Convert a value into a HostArray that owns it, with no copy of the data.
pub(crate) trait ToHostArray {
    fn to_host_array(self) -> HostArray;
}

/// Convert a HostArray to a diffsol vector or dense matrix. The array can have any strides, and
/// the scalar type `f32` or `f64`.
pub(crate) trait FromHostArray: Sized {
    /// The diffsol context of the new value.
    type Context;
    fn from_host_array(array: &HostArray, ctx: &Self::Context) -> Result<Self, DiffsolRtError>;
}

/// Read the values of `array` in lane-major order, with `nbatch` lanes.
///
/// `lane_index(b, k)` is the index of value `k` of lane `b`, and each lane has `len` values.
/// `array` must have `nbatch` lanes or 1 lane on its last axis. The values of one lane are
/// copied to all `nbatch` lanes, so the result always has a value for each lane.
fn lane_values<T: FromPrimitive + Clone>(
    array: &HostArray,
    nbatch: usize,
    len: usize,
    lane_index: impl Fn(usize, usize) -> Vec<usize>,
) -> Result<Vec<T>, DiffsolRtError> {
    let nlanes = *array.shape().last().unwrap();
    if nlanes != nbatch && nlanes != 1 {
        return Err(
            DiffsolError::Other(format!("Expected 1 or {nbatch} lanes, got {nlanes}")).into(),
        );
    }
    let offsets = (0..nlanes)
        .flat_map(|b| (0..len).map(move |k| (b, k)))
        .map(|(b, k)| array.offset(&lane_index(b, k)));
    let values: Vec<T> = match array.dtype() {
        ScalarType::F32 => {
            let data = array.as_slice::<f32>()?;
            offsets.map(|i| T::from_f32(data[i]).unwrap()).collect()
        }
        ScalarType::F64 => {
            let data = array.as_slice::<f64>()?;
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
fn vector_from<V: Vector<T: Scalar>>(array: &HostArray, ctx: &V::C) -> Result<V, DiffsolRtError> {
    array.expect_ndim(2)?;
    // diffsol stores lane `b`, value `i` at `b * n + i`
    let data = lane_values(array, ctx.nbatch(), array.shape()[0], |b, i| vec![i, b])?;
    Ok(V::from_vec(data, ctx.clone()))
}

/// Convert an `[nrows, ncols, nlanes]` array to a diffsol dense matrix with the context `ctx`.
fn dense_matrix_from<M: DenseMatrix<T: Scalar>>(
    array: &HostArray,
    ctx: &M::C,
) -> Result<M, DiffsolRtError> {
    array.expect_ndim(3)?;
    // diffsol stores lane `b`, row `i`, column `j` at `(b * ncols + j) * nrows + i`
    let (nrows, ncols) = (array.shape()[0], array.shape()[1]);
    let data = lane_values(array, ctx.nbatch(), nrows * ncols, |b, k| {
        vec![k % nrows, k / nrows, b]
    })?;
    Ok(M::from_vec(nrows, ncols, data, ctx.clone()))
}

// A diffsol vector converts from an `[n, nlanes]` array, and a diffsol dense matrix from an
// `[nrows, ncols, nlanes]` array. `nlanes` must be the number of lanes of the context, or 1. If
// it is 1, all the lanes get the values of the one lane.

impl<T: Scalar + FaerScalar> FromHostArray for diffsol::FaerVec<T> {
    type Context = diffsol::FaerContext;
    fn from_host_array(array: &HostArray, ctx: &Self::Context) -> Result<Self, DiffsolRtError> {
        vector_from(array, ctx)
    }
}

impl<T: Scalar + NalgebraScalar> FromHostArray for diffsol::NalgebraVec<T> {
    type Context = diffsol::NalgebraContext;
    fn from_host_array(array: &HostArray, ctx: &Self::Context) -> Result<Self, DiffsolRtError> {
        vector_from(array, ctx)
    }
}

impl<T: Scalar + FaerScalar> FromHostArray for diffsol::FaerMat<T> {
    type Context = diffsol::FaerContext;
    fn from_host_array(array: &HostArray, ctx: &Self::Context) -> Result<Self, DiffsolRtError> {
        dense_matrix_from(array, ctx)
    }
}

impl<T: Scalar + NalgebraScalar> FromHostArray for diffsol::NalgebraMat<T> {
    type Context = diffsol::NalgebraContext;
    fn from_host_array(array: &HostArray, ctx: &Self::Context) -> Result<Self, DiffsolRtError> {
        dense_matrix_from(array, ctx)
    }
}

// The two view functions below give a view that does not own the data, with no lifetime link
// to it. So they are `unsafe`: the caller must keep the data alive while the view exists. Each
// caller then calls `with_owner`, which makes the view the owner of the value that holds the
// data.

/// A view of a faer matrix that does not own the data.
///
/// # Safety
/// `m` must stay alive, and must not change, while the view exists.
unsafe fn faer_view<T: Scalar + FaerScalar>(m: &faer::Mat<T>) -> HostArray {
    let ptr = m.as_ptr() as *mut u8;
    // faer gives the strides in values; a matrix that owns its data has positive strides
    let size = std::mem::size_of::<T>();
    let strides = vec![
        m.row_stride() as usize * size,
        m.col_stride() as usize * size,
    ];
    // SAFETY: the shape and strides are those of `m`, and the caller keeps `m` alive
    unsafe { HostArray::new(ptr, vec![m.nrows(), m.ncols()], strides, T::scalar_type()) }
}

/// A view of a nalgebra matrix that does not own the data.
///
/// # Safety
/// `m` must stay alive, and must not change, while the view exists.
unsafe fn nalgebra_view<T: Scalar + NalgebraScalar>(m: &nalgebra::DMatrix<T>) -> HostArray {
    let ptr = m.as_ptr() as *mut u8;
    // nalgebra gives the strides in values
    let size = std::mem::size_of::<T>();
    let (rs, cs) = m.strides();
    let strides = vec![rs * size, cs * size];
    // SAFETY: the shape and strides are those of `m`, and the caller keeps `m` alive
    unsafe { HostArray::new(ptr, vec![m.nrows(), m.ncols()], strides, T::scalar_type()) }
}

// A diffsol vector or dense matrix converts to a HostArray that owns it, with no copy of the
// data. The HostArray shows the storage with the lane axis last.
//
// The storage of a diffsol vector is an `n × nbatch` matrix with one column for each lane, which
// is already the shape `[n, nbatch]`.

/// Give the storage of a diffsol dense matrix, an `nrows × (nbatch * ncols)` matrix where the
/// columns of lane `b` are the column block `b`, the shape `[nrows, ncols, nbatch]`.
fn matrix_lanes(storage: HostArray, nbatch: usize) -> HostArray {
    let (nrows, ncols) = (storage.shape()[0], storage.shape()[1] / nbatch);
    let (row_stride, col_stride) = (storage.strides()[0], storage.strides()[1]);
    let shape = vec![nrows, ncols, nbatch];
    let strides = vec![row_stride, col_stride, ncols * col_stride];
    storage.with_layout(shape, strides)
}

impl<T: Scalar + FaerScalar + 'static> ToHostArray for diffsol::FaerVec<T> {
    fn to_host_array(self) -> HostArray {
        // SAFETY: `with_owner` keeps the vector alive while the view exists, and the data is on the
        // heap, so it does not move when `with_owner` moves the vector
        unsafe { faer_view(self.inner()) }.with_owner(self)
    }
}

impl<T: Scalar + NalgebraScalar + 'static> ToHostArray for diffsol::NalgebraVec<T> {
    fn to_host_array(self) -> HostArray {
        // SAFETY: `with_owner` keeps the vector alive while the view exists, and the data is on the
        // heap, so it does not move when `with_owner` moves the vector
        unsafe { nalgebra_view(self.inner()) }.with_owner(self)
    }
}

impl<T: Scalar + FaerScalar + 'static> ToHostArray for diffsol::FaerMat<T> {
    fn to_host_array(self) -> HostArray {
        let nbatch = self.context().nbatch();
        // SAFETY: `with_owner` keeps the matrix alive while the view exists, and the data is on the
        // heap, so it does not move when `with_owner` moves the matrix
        matrix_lanes(unsafe { faer_view(self.inner()) }.with_owner(self), nbatch)
    }
}

impl<T: Scalar + NalgebraScalar + 'static> ToHostArray for diffsol::NalgebraMat<T> {
    fn to_host_array(self) -> HostArray {
        let nbatch = self.context().nbatch();
        // SAFETY: `with_owner` keeps the matrix alive while the view exists, and the data is on the
        // heap, so it does not move when `with_owner` moves the matrix
        matrix_lanes(
            unsafe { nalgebra_view(self.inner()) }.with_owner(self),
            nbatch,
        )
    }
}

impl<T: Scalar + 'static> ToHostArray for Vec<T> {
    fn to_host_array(self) -> HostArray {
        let (ptr, len) = (self.as_ptr() as *mut u8, self.len());
        // SAFETY: there are `len` values at `ptr`. `with_owner` keeps the `Vec` alive, and its
        // data is on the heap, so it does not move when `with_owner` moves the `Vec`
        unsafe { HostArray::new_col_major(ptr, vec![len], T::scalar_type()) }.with_owner(self)
    }
}

/// An N-D array of `f32` or `f64` values that Rust and the host language (C, Python) share
/// without a copy.
///
/// # Data layout
/// A `HostArray` has a data pointer, a scalar type (`dtype`), a `shape` and a `strides` value for
/// each axis. The strides are in bytes. The value at the index `(i0, i1, ...)` is at the byte
/// offset `i0 * strides[0] + i1 * strides[1] + ...` from the data pointer.
///
/// [HostArray::as_slice] gives a slice over the data, from the first value to the last value.
/// [HostArray::offset] gives the index of a value in that slice. The slice can contain values
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
/// `c` is `nrows * s` for nalgebra. It can be larger for faer, which pads its columns. An input
/// array (for example from C) can have any strides.
pub struct HostArray {
    dtype: ScalarType,
    shape: Vec<usize>,
    strides: Vec<usize>,
    ptr: *mut u8,
    owner: Option<Box<dyn Any>>,
}

fn scalar_size(dtype: ScalarType) -> usize {
    match dtype {
        ScalarType::F32 => std::mem::size_of::<f32>(),
        ScalarType::F64 => std::mem::size_of::<f64>(),
    }
}

/// The byte strides of a column-major (Fortran) array with the shape `shape`.
fn col_major_strides(shape: &[usize], dtype: ScalarType) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = scalar_size(dtype);
    for &n in shape {
        strides.push(stride);
        stride *= n;
    }
    strides
}

impl HostArray {
    /// Make an array that does not own its data, with any shape and strides (in bytes). The
    /// [HostArray] docs give the data layout.
    ///
    /// # Safety
    /// For each index in `shape`, `ptr` plus the byte offset from `strides` must point to a valid,
    /// aligned value of the type `dtype`. The data must stay valid, and must not change, while
    /// the array exists.
    pub unsafe fn new(
        ptr: *mut u8,
        shape: Vec<usize>,
        strides: Vec<usize>,
        dtype: ScalarType,
    ) -> Self {
        Self {
            ptr,
            shape,
            strides,
            dtype,
            owner: None,
        }
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
        // SAFETY: `values` has one value for each index of `shape`. `with_owner` keeps it alive,
        // and its data is on the heap, so it does not move when `with_owner` moves `values`
        let ptr = values.as_ptr() as *mut u8;
        unsafe { HostArray::new_col_major(ptr, shape, T::scalar_type()) }.with_owner(values)
    }
    /// Make an array that does not own its data, with the shape `shape` in column-major (Fortran)
    /// order.
    ///
    /// # Safety
    /// `ptr` must point to `product(shape)` valid, aligned, contiguous values of the type
    /// `dtype`. The data must stay valid, and must not change, while the array exists.
    pub(crate) unsafe fn new_col_major(ptr: *mut u8, shape: Vec<usize>, dtype: ScalarType) -> Self {
        let strides = col_major_strides(&shape, dtype);
        // SAFETY: the caller guarantees the values for a column-major layout of `shape`
        unsafe { HostArray::new(ptr, shape, strides, dtype) }
    }
    /// Make the array own `owner`, the value that holds its data, so that the data stays alive
    /// while the array exists. The data must be on the heap, so that it does not move when
    /// `owner` moves into the array.
    fn with_owner<O: 'static>(mut self, owner: O) -> Self {
        self.owner = Some(Box::new(owner));
        self
    }
    /// The length of each axis.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    /// The stride of each axis, in bytes. The [HostArray] docs give the data layout.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
    /// The pointer to the first value. It is valid while `self` exists, and the data must not
    /// be changed through it.
    pub fn data_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }
    /// The scalar type of the values.
    pub fn dtype(&self) -> ScalarType {
        self.dtype
    }
    /// Check that the array has `expected` axes. `name` names the array in the error message.
    pub(crate) fn expect_ndim(&self, expected: usize) -> Result<(), DiffsolRtError> {
        let ndim = self.shape.len();
        if ndim != expected {
            return Err(DiffsolError::Other(format!(
                "Expected a {expected}D array, got a {ndim}D array"
            ))
            .into());
        }
        Ok(())
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
    /// The index in [HostArray::as_slice] of the value at `index`.
    ///
    /// # Panics
    /// If `index` does not have one value for each axis, or a value is out of range.
    pub fn offset(&self, index: &[usize]) -> usize {
        assert_eq!(index.len(), self.shape.len(), "wrong number of indices");
        let bytes: usize = index
            .iter()
            .zip(&self.shape)
            .zip(&self.strides)
            .map(|((&i, &n), &stride)| {
                assert!(i < n, "index {i} is out of range for an axis of length {n}");
                i * stride
            })
            .sum();
        bytes / scalar_size(self.dtype)
    }

    /// A slice over the data of the array, from the first value to the last value. Use
    /// [HostArray::offset] to find a value. The scalar type must be `T`.
    pub fn as_slice<T: Scalar>(&self) -> Result<&[T], DiffsolRtError> {
        if self.dtype != T::scalar_type() {
            return Err(DiffsolError::Other("Data type mismatch".to_string()).into());
        }
        if self.shape.contains(&0) {
            return Ok(&[]);
        }
        let last = self.offset(&self.shape.iter().map(|n| n - 1).collect::<Vec<_>>());
        if !(self.ptr as usize).is_multiple_of(std::mem::align_of::<T>()) {
            return Err(DiffsolError::Other("The data is not aligned".to_string()).into());
        }
        // SAFETY: the data has a value of type `T` at each index up to `last`, and stays valid
        // while `self` exists
        Ok(unsafe { std::slice::from_raw_parts(self.ptr as *const T, last + 1) })
    }
}

#[cfg(test)]
mod tests {
    use super::{FromHostArray, HostArray, ToHostArray};

    use diffsol::{
        Context, DenseMatrix, FaerContext, FaerMat, FaerVec, NalgebraContext, NalgebraMat,
        NalgebraVec, Vector,
    };

    /// Lane `b`, element `i` of a test vector.
    fn vector_value(b: usize, i: usize) -> f64 {
        10.0 * b as f64 + i as f64
    }

    /// The `f64` value of `array` at `index`.
    fn at(array: &HostArray, index: &[usize]) -> f64 {
        array.as_slice::<f64>().unwrap()[array.offset(index)]
    }

    /// All the `f64` values of `array`, in column-major order of its shape (the first axis
    /// changes fastest).
    fn all_values(array: &HostArray) -> Vec<f64> {
        let mut indices = vec![vec![]];
        for &n in array.shape().iter().rev() {
            indices = indices
                .into_iter()
                .flat_map(|index: Vec<usize>| {
                    (0..n).map(move |i| [vec![i], index.clone()].concat())
                })
                .collect();
        }
        indices.iter().map(|index| at(array, index)).collect()
    }

    /// Lane `b`, row `i`, column `j` of a test matrix.
    fn matrix_value(b: usize, i: usize, j: usize) -> f64 {
        100.0 * b as f64 + 10.0 * i as f64 + j as f64
    }

    fn check_vector_round_trip<V>(ctx: V::C, nbatch: usize)
    where
        V: Vector<T = f64> + ToHostArray + FromHostArray<Context = V::C>,
    {
        let n = 3;
        let ctx = ctx.clone_with_nbatch(nbatch).unwrap();
        let values = (0..nbatch)
            .flat_map(|b| (0..n).map(move |i| vector_value(b, i)))
            .collect();
        let v = V::from_vec(values, ctx.clone());
        let array = v.clone().to_host_array();
        assert_eq!(array.shape(), &[n, nbatch]);
        for b in 0..nbatch {
            let lane: Vec<f64> = (0..n).map(|i| vector_value(b, i)).collect();
            assert_eq!(
                (0..n).map(|i| at(&array, &[i, b])).collect::<Vec<_>>(),
                lane
            );
        }
        let back = V::from_host_array(&array, &ctx).unwrap();
        assert_eq!(back.context().nbatch(), nbatch);
        assert_eq!(back.clone_as_vec(), v.clone_as_vec());
    }

    fn check_matrix_round_trip<M>(ctx: M::C, nbatch: usize)
    where
        M: DenseMatrix<T = f64> + ToHostArray + FromHostArray<Context = M::C>,
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
        let array = m.to_host_array();
        assert_eq!(array.shape(), &[nrows, ncols, nbatch]);
        for b in 0..nbatch {
            for i in 0..nrows {
                for j in 0..ncols {
                    assert_eq!(at(&array, &[i, j, b]), matrix_value(b, i, j));
                }
            }
        }
        let back = M::from_host_array(&array, &ctx).unwrap();
        let again = back.to_host_array();
        assert_eq!(all_values(&again), all_values(&array));
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
        let one_lane = HostArray::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let v = NalgebraVec::<f64>::from_host_array(&one_lane, &ctx).unwrap();
        assert_eq!(v.context().nbatch(), 3);
        assert_eq!(v.clone_as_vec(), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);

        // a matrix with one lane also gets a copy of its values in each lane
        let one_lane = HostArray::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2, 1]);
        let m = NalgebraMat::<f64>::from_host_array(&one_lane, &ctx).unwrap();
        assert_eq!(diffsol::Matrix::context(&m).nbatch(), 3);
        let all = all_values(&m.to_host_array());
        assert_eq!(all, [1.0, 2.0, 3.0, 4.0].repeat(3));
        let one_lane = HostArray::from_vec(vec![1.0, 2.0], vec![2, 1]);

        let two_lanes = HostArray::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let err = NalgebraVec::<f64>::from_host_array(&two_lanes, &ctx)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Expected 1 or 3 lanes, got 2"), "{err}");

        let err =
            NalgebraVec::<f64>::from_host_array(&one_lane.with_layout(vec![2], vec![8]), &ctx)
                .unwrap_err()
                .to_string();
        assert!(err.contains("Expected a 2D array, got a 1D array"), "{err}");
    }

    #[test]
    fn with_layout_keeps_the_values() {
        // `[6, 2]` column-major, with the first axis split into `[3, 2]`
        let values: Vec<f64> = (0..12).map(f64::from).collect();
        let s = std::mem::size_of::<f64>();
        let array = HostArray::from_vec(values.clone(), vec![6, 2])
            .with_layout(vec![3, 2, 2], vec![s, 3 * s, 6 * s]);
        assert_eq!(array.shape(), &[3, 2, 2]);
        assert_eq!(all_values(&array), values);
    }

    #[test]
    fn as_slice_covers_the_data_and_offset_uses_the_strides() {
        let data = [1.0f64, 2.0, 3.0, 4.0];
        // a column-major `[2, 2]` array
        let s = std::mem::size_of::<f64>();
        // SAFETY: `data` has 4 values, and lives longer than `host`
        let host = unsafe {
            HostArray::new(
                data.as_ptr() as *mut u8,
                vec![2, 2],
                vec![s, 2 * s],
                super::ScalarType::F64,
            )
        };
        assert_eq!(host.as_slice::<f64>().unwrap(), &data);
        assert_eq!(host.offset(&[0, 1]), 2);
        assert_eq!(host.offset(&[1, 0]), 1);
        let error = host.as_slice::<f32>().unwrap_err().to_string();
        assert!(error.contains("Data type mismatch"), "{error}");
    }

    #[test]
    fn as_slice_includes_the_padding_of_faer_storage() {
        // faer pads the columns of its storage, so the lanes of a faer vector are not adjacent
        let ctx = FaerContext::default().clone_with_nbatch(2).unwrap();
        let v = FaerVec::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], ctx);
        let array = v.to_host_array();
        let lane_start = array.offset(&[0, 1]);
        assert!(lane_start >= 3);
        assert_eq!(array.as_slice::<f64>().unwrap().len(), lane_start + 3);
        assert_eq!(all_values(&array), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn metadata_gives_the_layout() {
        let array = HostArray::from_vec((0..6).map(f64::from).collect(), vec![2, 3]);
        let s = std::mem::size_of::<f64>();
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array.strides(), &[s, 2 * s]);
        assert_eq!(array.dtype(), super::ScalarType::F64);
        // the value at (1, 2) is at the byte offset 1 * s + 2 * 2s from the data pointer
        let value = unsafe { *(array.data_ptr().add(s + 4 * s) as *const f64) };
        assert_eq!(value, 5.0);
    }

    #[test]
    fn vector_round_trips_from_1d_host_array() {
        let host = vec![1.0f64, 2.0, 3.0].to_host_array();
        let values = host.as_slice::<f64>().unwrap();
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }
}
