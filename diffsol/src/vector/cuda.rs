use std::ops::Add;
use std::sync::Arc;

use cudarc::cublas::result as cublas_result;
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, CudaView, CudaViewMut, DeviceRepr, DeviceSlice, ValidAsZeroBits, DevicePtr, DevicePtrMut};
use cudarc::cublas::sys::{cublasHandle_t, lib as cublas};

use crate::{Scalar, Vector, VectorCommon};

enum CudaType {
    F64,
}

trait ScalarCuda: Scalar + ValidAsZeroBits + DeviceRepr {
    fn as_enum() -> CudaType;
}

impl ScalarCuda for f64 {
    fn as_enum() -> CudaType {
        CudaType::F64
    }
}

#[derive(Debug)]
struct VectorCuda<T: ScalarCuda> {
    data: CudaSlice<T>,
    handle: cublasHandle_t,
    stream: Arc<CudaStream>,
}

impl<T: ScalarCuda> VectorCuda<T> {
    pub fn zeros(n: usize) -> Self {
        let device = CudaDevice::new(0).unwrap();
        let stream = device.fork_default_stream().unwrap();
        device.bind_to_thread().unwrap();
        let handle = cublas_result::create_handle().unwrap();
        unsafe { cublas_result::set_stream(handle, stream.stream as *mut _) }.unwrap();
        let data = device.alloc_zeros(n).unwrap();
        Self {
            data,
            handle,
            stream: Arc::new(stream),
        }
    }
}

#[derive(Debug)]
struct VectorCudaView<'a, T: ScalarCuda> {
    data: CudaView<'a, T>,
    parent: &'a VectorCuda<T>,
}

#[derive(Debug)]
struct VectorCudaViewMut<'a, T: ScalarCuda> {
    data: CudaViewMut<'a, T>,
    parent: &'a VectorCuda<T>,
}

macro_rules! impl_vector_common {
    ($vector_type:ty) => {
        impl<'a, T: ScalarCuda> VectorCommon for $vector_type {
            type T = T;
        }
    };
}

impl_vector_common!(VectorCuda<T>);
impl_vector_common!(VectorCudaView<'a, T>);
impl_vector_common!(VectorCudaViewMut<'a, T>);

// impl Add
impl<T: ScalarCuda> Add for VectorCuda<T> {
    type Output = VectorCuda<T>;

    fn add(self, mut rhs: Self) -> Self::Output {
        match T::as_enum() {
            CudaType::F64 => {
                unsafe { cublas().cublasDaxpy_v2(self.handle, self.data.len() as i32, &1.0, *self.data.device_ptr() as *const _, 1, *rhs.data.device_ptr_mut() as *mut _, 1) };
            }
        }
        rhs
    }
}



impl<T: Scalar> Vector for VectorCuda<T> {
    type View<'a> = VectorCudaView<'a, T>;
    type ViewMut<'a> = VectorCudaViewMut<'a, T>;

    fn as_view(&self) -> Self::View<'_> {
        VectorCudaView {
            data: self.data.as_slice(),
            phantom: std::marker::PhantomData,
        }
    }

}