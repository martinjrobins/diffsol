use crate::{
    error::DiffsolJsError,
    scalar_type::{Scalar, ScalarType},
};
use diffsol::FaerScalar;
use ndarray::{ArrayView2, ShapeBuilder};
use std::any::Any;

pub trait ToHostArray<T> {
    fn to_host_array(self) -> HostArray;
}

pub trait FromHostArray<T> {
    fn from_host_array(array: HostArray) -> Result<T, DiffsolJsError>;
}

impl<T: Scalar + FaerScalar + 'static> ToHostArray<T> for faer::Mat<T> {
    fn to_host_array(self) -> HostArray {
        let owner = Box::new(self);
        let nrows = owner.nrows();
        let ncols = owner.ncols();
        let row_stride = owner.row_stride();
        let col_stride = owner.col_stride();
        let ptr = owner.as_ptr() as *mut u8;
        HostArray::new_col_major(ptr, nrows, ncols, row_stride, col_stride, T::scalar_type())
            .with_owner(owner)
    }
}

impl<T: Scalar + 'static> ToHostArray<T> for nalgebra::DMatrix<T> {
    fn to_host_array(self) -> HostArray {
        let owner = Box::new(self);
        let nrows = owner.nrows();
        let ncols = owner.ncols();
        let (row_stride, col_stride) = owner.strides();
        let row_stride = row_stride as isize;
        let col_stride = col_stride as isize;
        let ptr = owner.as_ptr() as *mut u8;
        HostArray::new_col_major(ptr, nrows, ncols, row_stride, col_stride, T::scalar_type())
            .with_owner(owner)
    }
}

impl<T: Scalar + 'static> ToHostArray<T> for nalgebra::DVector<T> {
    fn to_host_array(self) -> HostArray {
        let owner = Box::new(self);
        let len = owner.len();
        let ptr = owner.as_ptr() as *mut u8;
        HostArray::new_vector(ptr, len, T::scalar_type()).with_owner(owner)
    }
}

impl<T: Scalar + FaerScalar + 'static> ToHostArray<T> for faer::Col<T> {
    fn to_host_array(self) -> HostArray {
        let owner = Box::new(self);
        let len = owner.nrows();
        let ptr = owner.as_ptr() as *mut u8;
        HostArray::new_vector(ptr, len, T::scalar_type()).with_owner(owner)
    }
}

impl<T: Scalar + 'static> ToHostArray<T> for Vec<T> {
    fn to_host_array(self) -> HostArray {
        let owner = Box::new(self);
        let len = owner.len();
        let ptr = owner.as_ptr() as *mut u8;
        HostArray::new_vector(ptr, len, T::scalar_type()).with_owner(owner)
    }
}

impl<'h, T: Scalar> FromHostArray<ArrayView2<'h, T>> for ArrayView2<'h, T> {
    fn from_host_array(array: HostArray) -> Result<Self, DiffsolJsError> {
        array.as_array()
    }
}

impl FromHostArray<Vec<f32>> for Vec<f32> {
    fn from_host_array(array: HostArray) -> Result<Self, DiffsolJsError> {
        array.as_slice::<f32>().map(|slice| slice.to_vec())
    }
}

impl FromHostArray<Vec<f64>> for Vec<f64> {
    fn from_host_array(array: HostArray) -> Result<Self, DiffsolJsError> {
        match array.dtype() {
            ScalarType::F32 => Ok(array
                .as_slice::<f32>()?
                .iter()
                .map(|&value| value as f64)
                .collect()),
            ScalarType::F64 => Ok(array.as_slice::<f64>()?.to_vec()),
        }
    }
}

impl FromHostArray<Vec<Vec<f32>>> for Vec<Vec<f32>> {
    fn from_host_array(array: HostArray) -> Result<Self, DiffsolJsError> {
        array.expect_ndim(2)?;
        let view = array.as_array::<f32>()?;
        Ok((0..view.nrows())
            .map(|row| (0..view.ncols()).map(|col| view[(row, col)]).collect())
            .collect())
    }
}

impl FromHostArray<Vec<Vec<f64>>> for Vec<Vec<f64>> {
    fn from_host_array(array: HostArray) -> Result<Self, DiffsolJsError> {
        array.expect_ndim(2)?;
        match array.dtype() {
            ScalarType::F32 => {
                let view = array.as_array::<f32>()?;
                Ok((0..view.nrows())
                    .map(|row| {
                        (0..view.ncols())
                            .map(|col| view[(row, col)] as f64)
                            .collect()
                    })
                    .collect())
            }
            ScalarType::F64 => {
                let view = array.as_array::<f64>()?;
                Ok((0..view.nrows())
                    .map(|row| (0..view.ncols()).map(|col| view[(row, col)]).collect())
                    .collect())
            }
        }
    }
}

/// a read-only array that is allocated in rust and can be safely accessed in the host language (e.g. Python) without copying
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

impl HostArray {
    pub fn new(ptr: *mut u8, shape: Vec<usize>, strides: Vec<usize>, dtype: ScalarType) -> Self {
        Self {
            ptr,
            shape,
            strides,
            dtype,
            owner: None,
        }
    }
    pub fn new_vector(ptr: *mut u8, len: usize, dtype: ScalarType) -> Self {
        let elem_size = scalar_size(dtype);
        Self {
            ptr,
            shape: vec![len],
            strides: vec![elem_size],
            dtype,
            owner: None,
        }
    }
    pub fn alloc_vector(len: usize, dtype: ScalarType) -> Self {
        match dtype {
            ScalarType::F32 => {
                let mut data = vec![0f32; len];
                let ptr = data.as_mut_ptr() as *mut u8;
                HostArray::new_vector(ptr, len, dtype).with_owner(Box::new(data))
            }
            ScalarType::F64 => {
                let mut data = vec![0f64; len];
                let ptr = data.as_mut_ptr() as *mut u8;
                HostArray::new_vector(ptr, len, dtype).with_owner(Box::new(data))
            }
        }
    }
    pub fn new_col_major(
        ptr: *mut u8,
        rows: usize,
        cols: usize,
        row_stride_elems: isize,
        col_stride_elems: isize,
        dtype: ScalarType,
    ) -> Self {
        let elem_size = scalar_size(dtype);
        Self {
            ptr,
            shape: vec![rows, cols],
            strides: vec![
                elem_size * (row_stride_elems as usize),
                elem_size * (col_stride_elems as usize),
            ],
            dtype,
            owner: None,
        }
    }
    fn with_owner(mut self, owner: Box<dyn Any>) -> Self {
        self.owner = Some(owner);
        self
    }
    pub(crate) fn data_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }
    pub(crate) fn ndim(&self) -> usize {
        self.shape.len()
    }
    pub(crate) fn dim(&self, index: usize) -> usize {
        self.shape.get(index).copied().unwrap_or(0)
    }
    pub(crate) fn stride(&self, index: usize) -> usize {
        self.strides.get(index).copied().unwrap_or(0)
    }
    pub(crate) fn dtype(&self) -> ScalarType {
        self.dtype
    }
    fn expect_ndim(&self, expected: usize) -> Result<(), DiffsolJsError> {
        if self.shape.len() != expected {
            return Err(DiffsolJsError::from(diffsol::error::DiffsolError::Other(
                format!("Expected a {expected}D array"),
            )));
        }
        Ok(())
    }
    pub fn as_array<'h, T: Scalar>(&self) -> Result<ArrayView2<'h, T>, DiffsolJsError> {
        self.expect_ndim(2)?;
        if self.dtype != T::scalar_type() {
            return Err(DiffsolJsError::from(diffsol::error::DiffsolError::Other(
                "Data type mismatch".to_string(),
            )));
        }
        let rows = self.shape[0];
        let cols = self.shape[1];
        let row_stride_bytes = self.strides[0];
        let col_stride_bytes = self.strides[1];
        let row_stride_elems = row_stride_bytes / std::mem::size_of::<T>();
        let col_stride_elems = col_stride_bytes / std::mem::size_of::<T>();
        unsafe {
            Ok(ArrayView2::from_shape_ptr(
                (rows, cols).strides((row_stride_elems, col_stride_elems)),
                self.ptr as *const T,
            ))
        }
    }
    pub fn as_slice<T: Scalar>(&self) -> Result<&[T], DiffsolJsError> {
        self.expect_ndim(1)?;
        if self.dtype != T::scalar_type() {
            return Err(DiffsolJsError::from(diffsol::error::DiffsolError::Other(
                "Data type mismatch".to_string(),
            )));
        }
        let len = self.shape[0];
        Ok(unsafe { std::slice::from_raw_parts(self.ptr as *const T, len) })
    }
}

impl Drop for HostArray {
    fn drop(&mut self) {
        if let Some(owner) = self.owner.take() {
            drop(owner);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{FromHostArray, HostArray, ToHostArray};

    #[test]
    fn vector_from_host_array_rejects_non_1d_input() {
        let array = vec![vec![1.0f64, 2.0], vec![3.0, 4.0]]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let host = HostArray::new_col_major(
            array.as_ptr() as *mut u8,
            2,
            2,
            1,
            2,
            super::ScalarType::F64,
        );
        let error = Vec::<f64>::from_host_array(host).unwrap_err().to_string();
        assert!(error.contains("Expected a 1D array"));
    }

    #[test]
    fn vector_round_trips_from_1d_host_array() {
        let host = vec![1.0f64, 2.0, 3.0].to_host_array();
        let values = Vec::<f64>::from_host_array(host).unwrap();
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn matrix_from_host_array_rejects_non_2d_input() {
        let host = vec![1.0f64, 2.0, 3.0].to_host_array();
        let error = Vec::<Vec<f64>>::from_host_array(host)
            .unwrap_err()
            .to_string();
        assert!(error.contains("Expected a 2D array"));
    }
}
