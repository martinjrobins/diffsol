use std::collections::HashMap;
use std::ffi::c_int;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Sub, SubAssign};
use std::sync::{Arc, LazyLock, Mutex};
use std::{env, fs};

use super::{utils::*, VectorIndex, VectorView, VectorViewMut};
use cudarc::cublas::sys::lib as cublas;
use cudarc::cublas::CudaBlas;
use cudarc::driver::{
    CudaContext as CudaDevice, CudaFunction, CudaModule, CudaSlice, CudaStream, CudaView,
    CudaViewMut, DevicePtr, DevicePtrMut, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits,
};
use cudarc::nvrtc::Ptx;

use crate::error::{CudaError, DiffsolError};
use crate::{cuda_error, IndexType, Scalar, Scale, Vector, VectorCommon};

static DEVICES: LazyLock<Mutex<CudaGlobalContext>> =
    LazyLock::new(|| Mutex::new(CudaGlobalContext::new()));

struct CudaGlobalContext {
    devices: HashMap<usize, (Arc<CudaDevice>, Arc<CudaModule>)>,
}

impl CudaGlobalContext {
    fn new() -> Self {
        let devices = HashMap::new();
        Self { devices }
    }
    fn get(&self, ordinal: usize) -> Option<&(Arc<CudaDevice>, Arc<CudaModule>)> {
        self.devices.get(&ordinal)
    }
    fn insert(&mut self, ordinal: usize, device: Arc<CudaDevice>, module: Arc<CudaModule>) {
        self.devices.insert(ordinal, (device, module));
    }
}

#[derive(Clone, Debug)]
struct CudaContext {
    stream: Arc<CudaStream>,
}

extern "C" fn zero(_block_size: std::ffi::c_int) -> usize {
    0
}

impl CudaContext {
    /// Compiles the PTX files for the given device.
    fn compile_ptx(device: &Arc<CudaDevice>) -> Result<Arc<CudaModule>, DiffsolError> {
        let out_dir = env::var("OUT_DIR").unwrap();
        // module in diffsol.ptx
        let ptx_file = format!("{}/diffsol.ptx", out_dir);
        // check if the file exists
        if fs::metadata(&ptx_file).is_err() {
            return Err(cuda_error!(
                Other,
                format!("PTX file not found: {}", ptx_file)
            ));
        }
        // compile the PTX file
        let ptx = Ptx::from_file(&ptx_file);
        let module = device.load_module(ptx).map_err(|e| {
            cuda_error!(
                Other,
                format!("Failed to load module from PTX file {}: {}", ptx_file, e)
            )
        })?;
        Ok(module)
    }

    /// Gets the device for the given ordinal. If the device is not already created, it creates a new one.
    pub fn get_device_and_module(
        ordinal: usize,
    ) -> Result<(Arc<CudaDevice>, Arc<CudaModule>), DiffsolError> {
        let mut devices = DEVICES.lock().unwrap();
        let (device, module) = match devices.get(ordinal) {
            Some(dev_mod) => dev_mod.clone(),
            None => {
                let device = CudaDevice::new(ordinal).unwrap();
                let module = Self::compile_ptx(&device)?;
                devices.insert(ordinal, device.clone(), module.clone());
                (device, module)
            }
        };
        device.bind_to_thread().unwrap();
        Ok((device, module))
    }

    /// Creates a new CudaContext with the given ordinal and creates a new non-default stream.
    pub fn new_with_stream(ordinal: usize) -> Result<Self, DiffsolError> {
        let (device, _module) = Self::get_device_and_module(ordinal)?;
        let stream = device
            .new_stream()
            .map_err(|e| cuda_error!(Other, format!("Failed to create new stream: {}", e)))?;
        Ok(Self { stream })
    }

    /// Creates a new CudaContext with the given ordinal (uses default stream).
    pub fn new(ordinal: usize) -> Result<Self, DiffsolError> {
        let (device, _module) = Self::get_device_and_module(ordinal)?;
        let stream = device.default_stream();
        Ok(Self { stream })
    }

    fn function<T: ScalarCuda>(&self, kernel_name: &str) -> CudaFunction {
        let ordinal = self.stream.context().ordinal();
        let (_device, module) =
            Self::get_device_and_module(ordinal).expect("Failed to get device and module");
        let kernel_name = format!("{}_{}", kernel_name, T::as_str());
        module
            .load_function(kernel_name.as_str())
            .unwrap_or_else(|_| {
                let kernel_name = format!("{}_{}", kernel_name, T::as_str());
                panic!(
                    "Failed to load function {} from module diffsol",
                    kernel_name
                )
            })
    }

    fn launch_config_1d(&self, n: u32, f: &CudaFunction) -> LaunchConfig {
        let (_min_grid_size, block_size) = f
            .occupancy_max_potential_block_size(zero, 0, 0, None)
            .expect("Failed to get occupancy max potential block size");
        let grid_size = n.div_ceil(block_size); // Round up according to array size
        LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    fn axpy<T: ScalarCuda, D1: DevicePtr<T>, D2: DevicePtrMut<T>>(
        &self,
        alpha: T,
        x: &D1,
        y: &mut D2,
    ) {
        let blas = CudaBlas::new(self.stream.clone()).expect("Failed to create CudaBlas");
        let n = x.len() as c_int;
        let (x, _syn_x) = x.device_ptr(&self.stream);
        let (y, _syn_y) = y.device_ptr_mut(&self.stream);
        match T::as_enum() {
            CudaType::F64 => {
                let x = x as *const f64;
                let y = y as *mut f64;
                let alpha = alpha.as_f64();
                unsafe {
                    cublas().cublasDaxpy_v2(*blas.handle(), n, &alpha as *const f64, x, 1, y, 1)
                }
            }
        }
        .result()
        .expect("Failed to call cublasDaxpy_v2");
    }

    fn norm<T: ScalarCuda, D: DevicePtr<T>>(&self, x: &D, k: i32) -> T {
        if k != 2 {
            panic!("Unsupported norm type");
        }
        let blas = CudaBlas::new(self.stream.clone()).expect("Failed to create CudaBlas");
        let n = x.len() as c_int;
        let (x, _syn_x) = x.device_ptr(&self.stream);
        let result: T;
        match T::as_enum() {
            CudaType::F64 => {
                let x = x as *const f64;
                let mut result_f64 = 0.0;
                let status = unsafe {
                    cublas().cublasDnrm2_v2(*blas.handle(), n, x, 1, &mut result_f64 as *mut f64)
                };
                result = result_f64.into();
                status
            }
        }
        .result()
        .expect("Failed to call cublasDnrm2_v2");
        result
    }

    fn squared_norm<T: ScalarCuda>(
        &self,
        y: &CudaSlice<T>,
        y0: &CudaSlice<T>,
        atol: &CudaSlice<T>,
        rtol: T,
    ) -> T {
        let n = y.len() as u32;
        let f = self.function::<T>("vec_squared_norm");
        let config = self.launch_config_1d(n, &f);
        let blocks_per_grid = config.grid_dim.0;
        let mut partial_sums = unsafe {
            self.stream
                .alloc::<T>(blocks_per_grid as usize)
                .expect("Failed to allocate memory for partial sums")
        };
        let mut build = self.stream.launch_builder(&f);
        build
            .arg(y)
            .arg(y0)
            .arg(atol)
            .arg(&rtol)
            .arg(&n)
            .arg(&mut partial_sums);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
        let partial_sums = self
            .stream
            .memcpy_dtov(&partial_sums)
            .expect("Failed to copy data from device to host");
        partial_sums.into_iter().fold(T::zero(), |acc, x| acc + x)
    }

    fn squared_norm_view<T: ScalarCuda>(
        &self,
        y: &CudaView<T>,
        y0: &CudaSlice<T>,
        atol: &CudaSlice<T>,
        rtol: T,
    ) -> T {
        let n = y.len() as u32;
        let f = self.function::<T>("vec_squared_norm");
        let config = self.launch_config_1d(n, &f);
        let blocks_per_grid = config.grid_dim.0;
        let mut partial_sums = unsafe {
            self.stream
                .alloc::<T>(blocks_per_grid as usize)
                .expect("Failed to allocate memory for partial sums")
        };
        let mut build = self.stream.launch_builder(&f);
        build
            .arg(y)
            .arg(y0)
            .arg(atol)
            .arg(&rtol)
            .arg(&n)
            .arg(&mut partial_sums);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
        let partial_sums = self
            .stream
            .memcpy_dtov(&partial_sums)
            .expect("Failed to copy data from device to host");
        partial_sums.into_iter().fold(T::zero(), |acc, x| acc + x)
    }
}

impl Default for CudaContext {
    fn default() -> Self {
        Self::new(0).unwrap()
    }
}

enum CudaType {
    F64,
}

trait ScalarCuda: Scalar + ValidAsZeroBits + DeviceRepr {
    fn as_enum() -> CudaType;
    fn as_f64(self) -> f64 {
        panic!("Unsupported type for as_f64");
    }
    fn as_str() -> &'static str {
        match Self::as_enum() {
            CudaType::F64 => "f64",
        }
    }
}

impl ScalarCuda for f64 {
    fn as_enum() -> CudaType {
        CudaType::F64
    }
    fn as_f64(self) -> f64 {
        self
    }
}

#[derive(Debug, Clone)]
struct CudaVec<T: ScalarCuda> {
    data: CudaSlice<T>,
    context: CudaContext,
}

#[derive(Debug, Clone)]
struct CudaIndex {
    data: CudaSlice<IndexType>,
    context: CudaContext,
}

#[derive(Debug)]
struct CudaVecRef<'a, T: ScalarCuda> {
    data: CudaView<'a, T>,
    context: CudaContext,
}

#[derive(Debug)]
struct CudaVecMut<'a, T: ScalarCuda> {
    data: CudaViewMut<'a, T>,
    context: CudaContext,
}

//impl<T: ScalarCuda> DefaultDenseMatrix for CudaVec<T> {
//    type M = CudaMat<T>;
//}

macro_rules! impl_vector_common {
    ($vec:ty, $con:ty, $in:ty) => {
        impl<T: ScalarCuda> VectorCommon for $vec {
            type T = T;
            type C = $con;
            type Inner = $in;
            fn inner(&self) -> &Self::Inner {
                &self.data
            }
        }
    };
}
pub(crate) use impl_vector_common;

macro_rules! impl_vector_common_ref {
    ($vec:ty, $con:ty, $in:ty) => {
        impl<'a, T: ScalarCuda> VectorCommon for $vec {
            type T = T;
            type C = $con;
            type Inner = $in;
            fn inner(&self) -> &Self::Inner {
                &self.data
            }
        }
    };
}
pub(crate) use impl_vector_common_ref;

impl_vector_common!(CudaVec<T>, CudaContext, CudaSlice<T>);
impl_vector_common_ref!(CudaVecRef<'a, T>, CudaContext, CudaView<'a, T>);
impl_vector_common_ref!(CudaVecMut<'a, T>, CudaContext, CudaViewMut<'a, T>);

macro_rules! impl_mul_scalar {
    ($lhs:ty, $out:ty, $scalar:ty) => {
        impl<T: ScalarCuda> Mul<Scale<T>> for $lhs {
            type Output = $out;
            fn mul(mut self, rhs: Scale<T>) -> Self::Output {
                let f = self.context.function::<T>("vec_mul_assign_scalar");
                let n = self.data.len() as u32;
                let scalar = rhs.value();
                let mut build = self.context.stream.launch_builder(&f);
                build.arg(&mut self.data).arg(&scalar).arg(&n);
                let config = self.context.launch_config_1d(n, &f);
                unsafe { build.launch(config) }.expect("Failed to launch kernel");
                self
            }
        }
    };
}

macro_rules! impl_mul_scalar_alloc {
    ($lhs:ty, $out:ty, $scalar:ty) => {
        impl<T: ScalarCuda> Mul<Scale<T>> for $lhs {
            type Output = $out;
            fn mul(self, rhs: Scale<T>) -> Self::Output {
                let mut ret = Self::Output::zeros(self.data.len(), self.context.clone());
                let f = self.context.function::<T>("vec_mul_scalar");
                let n = self.data.len() as u32;
                let mut build = self.context.stream.launch_builder(&f);
                let scalar = rhs.value();
                build
                    .arg(&self.data)
                    .arg(&scalar)
                    .arg(&mut ret.data)
                    .arg(&n);
                let config = self.context.launch_config_1d(n, &f);
                unsafe { build.launch(config) }.expect("Failed to launch kernel");
                ret
            }
        }
    };
}

macro_rules! impl_mul_scalar_alloc_mut {
    ($lhs:ty, $out:ty, $scalar:ty) => {
        impl<T: ScalarCuda> Mul<Scale<T>> for $lhs {
            type Output = $out;
            fn mul(self, rhs: Scale<T>) -> Self::Output {
                let mut ret = Self::Output::zeros(self.data.len(), self.context.clone());
                let f = self.context.function::<T>("vec_mul_scalar");
                let n = self.data.len() as u32;
                let mut build = self.context.stream.launch_builder(&f);
                let scalar = rhs.value();
                let self_data = self.data.as_view();
                build
                    .arg(&self_data)
                    .arg(&scalar)
                    .arg(&mut ret.data)
                    .arg(&n);
                let config = self.context.launch_config_1d(n, &f);
                unsafe { build.launch(config) }.expect("Failed to launch kernel");
                ret
            }
        }
    };
}

macro_rules! impl_div_scalar {
    ($lhs:ty, $out:ty, $scalar:expr) => {
        impl<'a, T: ScalarCuda> Div<Scale<T>> for $lhs {
            type Output = $out;
            fn div(self, rhs: Scale<T>) -> Self::Output {
                let inv_rhs: T = T::one() / rhs.value();
                self.mul(Scale(inv_rhs))
            }
        }
    };
}

macro_rules! impl_mul_assign_scalar {
    ($col_type:ty, $scalar:ty) => {
        impl<'a, T: ScalarCuda> MulAssign<Scale<T>> for $col_type {
            fn mul_assign(&mut self, rhs: Scale<T>) {
                let f = self.context.function::<T>("vec_mul_assign_scalar");
                let n = self.data.len() as u32;
                let mut build = self.context.stream.launch_builder(&f);
                let scalar = rhs.value();
                build.arg(&mut self.data).arg(&scalar).arg(&n);
                let config = self.context.launch_config_1d(n, &f);
                unsafe { build.launch(config) }.expect("Failed to launch kernel");
            }
        }
    };
}

impl_mul_scalar!(CudaVec<T>, CudaVec<T>, T);
impl_mul_scalar_alloc!(&CudaVec<T>, CudaVec<T>, T);
impl_mul_scalar_alloc!(CudaVecRef<'_, T>, CudaVec<T>, T);
impl_mul_scalar_alloc_mut!(CudaVecMut<'_, T>, CudaVec<T>, T);
impl_div_scalar!(CudaVec<T>, CudaVec<T>, T);
impl_mul_assign_scalar!(CudaVecMut<'a, T>, T);
impl_mul_assign_scalar!(CudaVec<T>, T);

macro_rules! impl_sub_assign {
    ($lhs:ty, $rhs:ty) => {
        impl<T: ScalarCuda> SubAssign<$rhs> for $lhs {
            fn sub_assign(&mut self, rhs: $rhs) {
                let f = self.context.function::<T>("vec_sub_assign");
                let n = self.data.len() as u32;
                let mut build = self.context.stream.launch_builder(&f);
                build.arg(&mut self.data).arg(&rhs.data).arg(&n);
                let config = self.context.launch_config_1d(n, &f);
                unsafe { build.launch(config) }.expect("Failed to launch kernel");
            }
        }
    };
}

macro_rules! impl_add_assign {
    ($lhs:ty, $rhs:ty) => {
        impl<T: ScalarCuda> AddAssign<$rhs> for $lhs {
            fn add_assign(&mut self, rhs: $rhs) {
                let f = self.context.function::<T>("vec_add_assign");
                let n = self.data.len() as u32;
                let mut build = self.context.stream.launch_builder(&f);
                build.arg(&mut self.data).arg(&rhs.data).arg(&n);
                let config = self.context.launch_config_1d(n, &f);
                unsafe { build.launch(config) }.expect("Failed to launch kernel");
            }
        }
    };
}

impl_sub_assign!(CudaVec<T>, CudaVec<T>);
impl_sub_assign!(CudaVec<T>, &CudaVec<T>);
impl_sub_assign!(CudaVec<T>, CudaVecRef<'_, T>);
impl_sub_assign!(CudaVec<T>, &CudaVecRef<'_, T>);

impl_sub_assign!(CudaVecMut<'_, T>, CudaVec<T>);
impl_sub_assign!(CudaVecMut<'_, T>, &CudaVec<T>);
impl_sub_assign!(CudaVecMut<'_, T>, CudaVecRef<'_, T>);
impl_sub_assign!(CudaVecMut<'_, T>, &CudaVecRef<'_, T>);

impl_add_assign!(CudaVec<T>, CudaVec<T>);
impl_add_assign!(CudaVec<T>, &CudaVec<T>);
impl_add_assign!(CudaVec<T>, CudaVecRef<'_, T>);
impl_add_assign!(CudaVec<T>, &CudaVecRef<'_, T>);

impl_add_assign!(CudaVecMut<'_, T>, CudaVec<T>);
impl_add_assign!(CudaVecMut<'_, T>, &CudaVec<T>);
impl_add_assign!(CudaVecMut<'_, T>, CudaVecRef<'_, T>);
impl_add_assign!(CudaVecMut<'_, T>, &CudaVecRef<'_, T>);

macro_rules! impl_sub_both_ref {
    ($lhs:ty, $rhs:ty, $out:ty) => {
        impl<T: ScalarCuda> Sub<$rhs> for $lhs {
            type Output = $out;
            fn sub(self, rhs: $rhs) -> Self::Output {
                assert_eq!(
                    self.data.len(),
                    rhs.data.len(),
                    "Vector length mismatch: {} != {}",
                    self.data.len(),
                    rhs.data.len()
                );
                let mut ret = Self::Output::zeros(self.data.len(), self.context.clone());
                let f = self.context.function::<T>("vec_sub");
                let n = self.data.len() as u32;
                let mut build = self.context.stream.launch_builder(&f);
                build
                    .arg(&self.data)
                    .arg(&rhs.data)
                    .arg(&mut ret.data)
                    .arg(&n);
                let config = self.context.launch_config_1d(n, &f);
                unsafe { build.launch(config) }.expect("Failed to launch kernel");
                ret
            }
        }
    };
}

macro_rules! impl_sub_lhs {
    ($lhs:ty, $rhs:ty, $out:ty) => {
        impl<T: ScalarCuda> Sub<$rhs> for $lhs {
            type Output = $out;
            fn sub(mut self, rhs: $rhs) -> Self::Output {
                assert_eq!(
                    self.data.len(),
                    rhs.data.len(),
                    "Vector length mismatch: {} != {}",
                    self.data.len(),
                    rhs.data.len()
                );
                let f = self.context.function::<T>("vec_sub_assign");
                let n = self.data.len() as u32;
                let mut build = self.context.stream.launch_builder(&f);
                build.arg(&mut self.data).arg(&rhs.data).arg(&n);
                let config = self.context.launch_config_1d(n, &f);
                unsafe { build.launch(config) }.expect("Failed to launch kernel");
                self
            }
        }
    };
}

macro_rules! impl_sub_rhs {
    ($lhs:ty, $rhs:ty, $out:ty) => {
        impl<T: ScalarCuda> Sub<$rhs> for $lhs {
            type Output = $out;
            fn sub(self, mut rhs: $rhs) -> Self::Output {
                assert_eq!(
                    self.data.len(),
                    rhs.data.len(),
                    "Vector length mismatch: {} != {}",
                    self.data.len(),
                    rhs.data.len()
                );
                let f = self.context.function::<T>("vec_sub_assign_rhs");
                let n = self.data.len() as u32;
                let mut build = self.context.stream.launch_builder(&f);
                build.arg(&self.data).arg(&mut rhs.data).arg(&n);
                let config = self.context.launch_config_1d(n, &f);
                unsafe { build.launch(config) }.expect("Failed to launch kernel");
                rhs
            }
        }
    };
}

impl_sub_both_ref!(&CudaVec<T>, &CudaVec<T>, CudaVec<T>);
impl_sub_rhs!(&CudaVec<T>, CudaVec<T>, CudaVec<T>);
impl_sub_both_ref!(&CudaVec<T>, CudaVecRef<'_, T>, CudaVec<T>);
impl_sub_both_ref!(&CudaVec<T>, &CudaVecRef<'_, T>, CudaVec<T>);

impl_sub_lhs!(CudaVec<T>, CudaVec<T>, CudaVec<T>);
impl_sub_lhs!(CudaVec<T>, &CudaVec<T>, CudaVec<T>);
impl_sub_lhs!(CudaVec<T>, CudaVecRef<'_, T>, CudaVec<T>);
impl_sub_lhs!(CudaVec<T>, &CudaVecRef<'_, T>, CudaVec<T>);

impl_sub_rhs!(CudaVecRef<'_, T>, CudaVec<T>, CudaVec<T>);
impl_sub_both_ref!(CudaVecRef<'_, T>, &CudaVec<T>, CudaVec<T>);
impl_sub_both_ref!(CudaVecRef<'_, T>, CudaVecRef<'_, T>, CudaVec<T>);
impl_sub_both_ref!(CudaVecRef<'_, T>, &CudaVecRef<'_, T>, CudaVec<T>);

macro_rules! impl_add_both_ref {
    ($lhs:ty, $rhs:ty, $out:ty) => {
        impl<T: ScalarCuda> Add<$rhs> for $lhs {
            type Output = $out;
            fn add(self, rhs: $rhs) -> Self::Output {
                assert_eq!(
                    self.data.len(),
                    rhs.data.len(),
                    "Vector length mismatch: {} != {}",
                    self.data.len(),
                    rhs.data.len()
                );
                let mut ret = Self::Output::zeros(self.data.len(), self.context.clone());
                let f = self.context.function::<T>("vec_add");
                let n = self.data.len() as u32;
                let mut build = self.context.stream.launch_builder(&f);
                build
                    .arg(&self.data)
                    .arg(&rhs.data)
                    .arg(&mut ret.data)
                    .arg(&n);
                let config = self.context.launch_config_1d(n, &f);
                unsafe { build.launch(config) }.expect("Failed to launch kernel");
                ret
            }
        }
    };
}

macro_rules! impl_add_lhs {
    ($lhs:ty, $rhs:ty, $out:ty) => {
        impl<T: ScalarCuda> Add<$rhs> for $lhs {
            type Output = $out;
            fn add(mut self, rhs: $rhs) -> Self::Output {
                assert_eq!(
                    self.data.len(),
                    rhs.data.len(),
                    "Vector length mismatch: {} != {}",
                    self.data.len(),
                    rhs.data.len()
                );
                let f = self.context.function::<T>("vec_add_assign");
                let n = self.data.len() as u32;
                let mut build = self.context.stream.launch_builder(&f);
                build.arg(&mut self.data).arg(&rhs.data).arg(&n);
                let config = self.context.launch_config_1d(n, &f);
                unsafe { build.launch(config) }.expect("Failed to launch kernel");
                self
            }
        }
    };
}

macro_rules! impl_add_rhs {
    ($lhs:ty, $rhs:ty, $out:ty) => {
        impl<T: ScalarCuda> Add<$rhs> for $lhs {
            type Output = $out;
            fn add(self, mut rhs: $rhs) -> Self::Output {
                assert_eq!(
                    self.data.len(),
                    rhs.data.len(),
                    "Vector length mismatch: {} != {}",
                    self.data.len(),
                    rhs.data.len()
                );
                let f = self.context.function::<T>("vec_add_assign_rhs");
                let n = self.data.len() as u32;
                let mut build = self.context.stream.launch_builder(&f);
                build.arg(&self.data).arg(&mut rhs.data).arg(&n);
                let config = self.context.launch_config_1d(n, &f);
                unsafe { build.launch(config) }.expect("Failed to launch kernel");
                rhs
            }
        }
    };
}

impl_add_both_ref!(&CudaVec<T>, &CudaVec<T>, CudaVec<T>);
impl_add_rhs!(&CudaVec<T>, CudaVec<T>, CudaVec<T>);
impl_add_both_ref!(&CudaVec<T>, CudaVecRef<'_, T>, CudaVec<T>);
impl_add_both_ref!(&CudaVec<T>, &CudaVecRef<'_, T>, CudaVec<T>);

impl_add_lhs!(CudaVec<T>, CudaVec<T>, CudaVec<T>);
impl_add_lhs!(CudaVec<T>, &CudaVec<T>, CudaVec<T>);
impl_add_lhs!(CudaVec<T>, CudaVecRef<'_, T>, CudaVec<T>);
impl_add_lhs!(CudaVec<T>, &CudaVecRef<'_, T>, CudaVec<T>);

impl_add_rhs!(CudaVecRef<'_, T>, CudaVec<T>, CudaVec<T>);
impl_add_both_ref!(CudaVecRef<'_, T>, &CudaVec<T>, CudaVec<T>);
impl_add_both_ref!(CudaVecRef<'_, T>, CudaVecRef<'_, T>, CudaVec<T>);
impl_add_both_ref!(CudaVecRef<'_, T>, &CudaVecRef<'_, T>, CudaVec<T>);

impl VectorIndex for CudaIndex {
    type C = CudaContext;
    fn context(&self) -> &Self::C {
        &self.context
    }
    fn len(&self) -> IndexType {
        self.data.len() as IndexType
    }
    fn zeros(len: IndexType, ctx: Self::C) -> Self {
        let data = ctx
            .stream
            .alloc_zeros(len)
            .expect("Failed to allocate memory for CudaVec");
        Self { data, context: ctx }
    }
    fn clone_as_vec(&self) -> Vec<IndexType> {
        self.context
            .stream
            .memcpy_dtov(&self.data)
            .expect("Failed to copy data from device to host")
    }
    fn from_vec(v: Vec<IndexType>, ctx: Self::C) -> Self {
        let mut data = unsafe {
            ctx.stream
                .alloc(v.len())
                .expect("Failed to allocate memory for CudaVec")
        };
        ctx.stream
            .memcpy_htod(&v, &mut data)
            .expect("Failed to copy data from host to device");
        Self { data, context: ctx }
    }
}

impl<T: ScalarCuda> Vector for CudaVec<T> {
    type View<'a> = CudaVecRef<'a, T>;
    type ViewMut<'a> = CudaVecMut<'a, T>;
    type Index = CudaIndex;

    fn context(&self) -> &Self::C {
        &self.context
    }
    fn inner_mut(&mut self) -> &mut Self::Inner {
        &mut self.data
    }
    fn get_index(&self, index: IndexType) -> Self::T {
        self.context
            .stream
            .memcpy_dtov(&self.data.slice(index..index + 1))
            .expect("Failed to copy data from device to host")[0]
    }
    fn set_index(&mut self, index: IndexType, value: Self::T) {
        let data = vec![value];
        self.context
            .stream
            .memcpy_htod(&data, &mut self.data.slice_mut(index..index + 1))
            .expect("Failed to copy data from host to device");
    }
    fn norm(&self, k: i32) -> Self::T {
        self.context.norm(&self.data, k)
    }
    fn squared_norm(&self, y: &Self, atol: &Self, rtol: Self::T) -> Self::T {
        self.context
            .squared_norm(&self.data, &y.data, &atol.data, rtol)
    }
    fn len(&self) -> IndexType {
        self.data.len() as IndexType
    }

    fn from_vec(v: Vec<Self::T>, ctx: Self::C) -> Self {
        let mut data = unsafe {
            ctx.stream
                .alloc(v.len())
                .expect("Failed to allocate memory for CudaVec")
        };
        ctx.stream
            .memcpy_htod(&v, &mut data)
            .expect("Failed to copy data from host to device");
        Self { data, context: ctx }
    }
    fn from_element(nstates: usize, value: Self::T, ctx: Self::C) -> Self {
        let data = unsafe {
            ctx.stream
                .alloc(nstates)
                .expect("Failed to allocate memory for CudaVec")
        };
        let mut ret = Self { data, context: ctx };
        ret.fill(value);
        ret
    }
    fn zeros(nstates: usize, ctx: Self::C) -> Self {
        let data = ctx
            .stream
            .alloc_zeros(nstates)
            .expect("Failed to allocate memory for CudaVec");
        Self { data, context: ctx }
    }

    fn fill(&mut self, value: Self::T) {
        let f = self.context.function::<T>("vec_fill");
        let n = self.len() as u32;
        let mut build = self.context.stream.launch_builder(&f);
        build.arg(&mut self.data).arg(&value).arg(&n);
        let config = self.context.launch_config_1d(n, &f);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }
    fn as_view(&self) -> Self::View<'_> {
        CudaVecRef {
            data: self.data.as_view(),
            context: self.context.clone(),
        }
    }
    fn as_view_mut(&mut self) -> Self::ViewMut<'_> {
        CudaVecMut {
            data: self.data.as_view_mut(),
            context: self.context.clone(),
        }
    }
    fn copy_from(&mut self, other: &Self) {
        let f = self.context.function::<T>("vec_copy");
        let n = self.len() as u32;
        let mut build = self.context.stream.launch_builder(&f);
        build.arg(&mut self.data).arg(&other.data).arg(&n);
        let config = self.context.launch_config_1d(n, &f);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }
    fn copy_from_view(&mut self, other: &Self::View<'_>) {
        let f = self.context.function::<T>("vec_copy");
        let n = self.len() as u32;
        let mut build = self.context.stream.launch_builder(&f);
        build.arg(&mut self.data).arg(&other.data).arg(&n);
        let config = self.context.launch_config_1d(n, &f);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }
    fn axpy(&mut self, alpha: Self::T, x: &Self, beta: Self::T) {
        self.mul_assign(Scale(beta));
        self.context.axpy::<T, _, _>(alpha, &x.data, &mut self.data);
    }
    fn axpy_v(&mut self, alpha: Self::T, x: &Self::View<'_>, beta: Self::T) {
        self.mul_assign(Scale(beta));
        self.context.axpy::<T, _, _>(alpha, &x.data, &mut self.data);
    }
    fn clone_as_vec(&self) -> Vec<Self::T> {
        self.context
            .stream
            .memcpy_dtov(&self.data)
            .expect("Failed to copy data from device to host")
    }
    fn component_mul_assign(&mut self, other: &Self) {
        let f = self.context.function::<T>("vec_mul_assign");
        let n = self.len() as u32;
        let mut build = self.context.stream.launch_builder(&f);
        build.arg(&mut self.data).arg(&other.data).arg(&n);
        let config = self.context.launch_config_1d(n, &f);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }
    fn component_div_assign(&mut self, other: &Self) {
        let f = self.context.function::<T>("vec_div_assign");
        let n = self.len() as u32;
        let mut build = self.context.stream.launch_builder(&f);
        build.arg(&mut self.data).arg(&other.data).arg(&n);
        let config = self.context.launch_config_1d(n, &f);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }
    fn root_finding(&self, g1: &Self) -> (bool, Self::T, i32) {
        let f = self.context.function::<T>("vec_root_finding");
        let n = self.len() as u32;
        let config = self.context.launch_config_1d(n, &f);
        let blocks_per_grid = config.grid_dim.0;
        let mut max_vals = unsafe {
            self.context
                .stream
                .alloc::<T>(blocks_per_grid as usize)
                .expect("Failed to allocate memory for partial sums")
        };
        let mut max_idxs = unsafe {
            self.context
                .stream
                .alloc::<c_int>(blocks_per_grid as usize)
                .expect("Failed to allocate memory for partial sums")
        };
        let mut root_flag = self
            .context
            .stream
            .alloc_zeros::<c_int>(1)
            .expect("Failed to allocate memory for partial sums");
        let mut build = self.context.stream.launch_builder(&f);
        build
            .arg(&self.data)
            .arg(&g1.data)
            .arg(&n)
            .arg(&mut root_flag)
            .arg(&mut max_vals)
            .arg(&mut max_idxs);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
        // Final reduction on CPU
        let h_max_vals = self
            .context
            .stream
            .memcpy_dtov(&max_vals)
            .expect("Failed to copy data from device to host");
        let h_max_idxs = self
            .context
            .stream
            .memcpy_dtov(&max_idxs)
            .expect("Failed to copy data from device to host");
        let h_root_flag = self
            .context
            .stream
            .memcpy_dtov(&root_flag)
            .expect("Failed to copy data from device to host");
        let root_flag = h_root_flag[0];
        let mut max_val = T::zero();
        let mut max_idx = -1;
        for (max_val_i, max_idx_i) in h_max_vals.iter().zip(h_max_idxs.iter()) {
            if *max_val_i > max_val {
                max_val = *max_val_i;
                max_idx = *max_idx_i;
            }
        }
        (root_flag != 0, max_val, max_idx)
    }
    fn assign_at_indices(&mut self, indices: &Self::Index, value: Self::T) {
        let f = self.context.function::<T>("vec_assign_at_indices");
        let n = indices.len() as u32;
        let mut build = self.context.stream.launch_builder(&f);
        build
            .arg(&mut self.data)
            .arg(&indices.data)
            .arg(&value)
            .arg(&n);
        let config = self.context.launch_config_1d(n, &f);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }
    fn copy_from_indices(&mut self, other: &Self, indices: &Self::Index) {
        let f = self.context.function::<T>("vec_copy_from_indices");
        let n = indices.len() as u32;
        let mut build = self.context.stream.launch_builder(&f);
        build
            .arg(&mut self.data)
            .arg(&other.data)
            .arg(&indices.data)
            .arg(&n);
        let config = self.context.launch_config_1d(n, &f);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }
    fn gather(&mut self, other: &Self, indices: &Self::Index) {
        let f = self.context.function::<T>("vec_gather");
        let n = indices.len() as u32;
        let mut build = self.context.stream.launch_builder(&f);
        build
            .arg(&mut self.data)
            .arg(&other.data)
            .arg(&indices.data)
            .arg(&n);
        let config = self.context.launch_config_1d(n, &f);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }
    fn scatter(&self, indices: &Self::Index, other: &mut Self) {
        let f = self.context.function::<T>("vec_scatter");
        let n = indices.len() as u32;
        let mut build = self.context.stream.launch_builder(&f);
        build
            .arg(&self.data)
            .arg(&indices.data)
            .arg(&mut other.data)
            .arg(&n);
        let config = self.context.launch_config_1d(n, &f);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }
}

impl<T: ScalarCuda> VectorView<'_> for CudaVecRef<'_, T> {
    type Owned = CudaVec<T>;
    fn into_owned(self) -> Self::Owned {
        let mut ret = unsafe { self.context.stream.alloc(self.data.len()) }
            .expect("Failed to allocate memory for CudaVec");
        self.context
            .stream
            .memcpy_dtod(&self.data, &mut ret)
            .expect("Failed to copy data from device to device");
        Self::Owned {
            data: ret,
            context: self.context,
        }
    }
    fn squared_norm(&self, y: &Self::Owned, atol: &Self::Owned, rtol: Self::T) -> Self::T {
        self.context
            .squared_norm_view(&self.data, &y.data, &atol.data, rtol)
    }
}

impl<'a, T: ScalarCuda> VectorViewMut<'a> for CudaVecMut<'a, T> {
    type Owned = CudaVec<T>;
    type View = CudaVecRef<'a, T>;
    type Index = CudaIndex;
    fn copy_from(&mut self, other: &Self::Owned) {
        let f = self.context.function::<T>("vec_copy");
        let n = self.data.len() as u32;
        let mut build = self.context.stream.launch_builder(&f);
        build.arg(&mut self.data).arg(&other.data).arg(&n);
        let config = self.context.launch_config_1d(n, &f);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }
    fn copy_from_view(&mut self, other: &Self::View) {
        let f = self.context.function::<T>("vec_copy");
        let n = self.data.len() as u32;
        let mut build = self.context.stream.launch_builder(&f);
        build.arg(&mut self.data).arg(&other.data).arg(&n);
        let config = self.context.launch_config_1d(n, &f);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }
    fn axpy(&mut self, alpha: Self::T, x: &Self::Owned, beta: Self::T) {
        self.mul_assign(Scale(beta));
        self.context.axpy::<T, _, _>(alpha, &x.data, &mut self.data);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Scale, Vector};

    fn setup_cuda_context() -> CudaContext {
        CudaContext::new(0).expect("Failed to create CudaContext")
    }

    #[test]
    fn test_cuda_vec_creation() {
        let ctx = setup_cuda_context();
        let vec = CudaVec::<f64>::zeros(10, ctx.clone());
        assert_eq!(vec.len(), 10);
        assert!(vec.clone_as_vec().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_cuda_vec_fill() {
        let ctx = setup_cuda_context();
        let mut vec = CudaVec::<f64>::zeros(10, ctx.clone());
        vec.fill(3.14);
        assert!(vec.clone_as_vec().iter().all(|&x| x == 3.14));
    }

    #[test]
    fn test_cuda_vec_add_assign() {
        let ctx = setup_cuda_context();
        let mut vec1 = CudaVec::<f64>::from_vec(vec![1.0, 2.0, 3.0], ctx.clone());
        let vec2 = CudaVec::<f64>::from_vec(vec![4.0, 5.0, 6.0], ctx.clone());
        vec1 += vec2;
        assert_eq!(vec1.clone_as_vec(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_cuda_vec_sub_assign() {
        let ctx = setup_cuda_context();
        let mut vec1 = CudaVec::<f64>::from_vec(vec![5.0, 7.0, 9.0], ctx.clone());
        let vec2 = CudaVec::<f64>::from_vec(vec![1.0, 2.0, 3.0], ctx.clone());
        vec1 -= vec2;
        assert_eq!(vec1.clone_as_vec(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_cuda_vec_mul_scalar() {
        let ctx = setup_cuda_context();
        let vec = CudaVec::<f64>::from_vec(vec![1.0, 2.0, 3.0], ctx.clone());
        let result = vec * Scale(2.0);
        assert_eq!(result.clone_as_vec(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_cuda_vec_div_scalar() {
        let ctx = setup_cuda_context();
        let vec = CudaVec::<f64>::from_vec(vec![2.0, 4.0, 6.0], ctx.clone());
        let result = vec / Scale(2.0);
        assert_eq!(result.clone_as_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_cuda_vec_norm() {
        let ctx = setup_cuda_context();
        let vec = CudaVec::<f64>::from_vec(vec![3.0, 4.0], ctx.clone());
        let norm = vec.norm(2);
        assert_eq!(norm, 5.0);
    }

    #[test]
    fn test_cuda_vec_axpy() {
        let ctx = setup_cuda_context();
        let mut y = CudaVec::<f64>::from_vec(vec![1.0, 2.0, 3.0], ctx.clone());
        let x = CudaVec::<f64>::from_vec(vec![4.0, 5.0, 6.0], ctx.clone());
        y.axpy(2.0, &x, 1.0);
        assert_eq!(y.clone_as_vec(), vec![9.0, 12.0, 15.0]);
    }

    #[test]
    fn test_cuda_vec_ref_squared_norm() {
        let ctx = setup_cuda_context();
        let vec1 = CudaVec::<f64>::from_vec(vec![1.0, 2.0, 3.0], ctx.clone());
        let vec2 = CudaVec::<f64>::from_vec(vec![1.0, 2.0, 3.0], ctx.clone());
        let atol = CudaVec::<f64>::from_vec(vec![0.1, 0.1, 0.1], ctx.clone());
        let rtol = 0.1;
        let norm = vec1.as_view().squared_norm(&vec2, &atol, rtol);
        assert_eq!(norm, 0.0);
    }

    #[test]
    fn test_cuda_vec_mut_copy_from() {
        let ctx = setup_cuda_context();
        let mut vec1 = CudaVec::<f64>::zeros(3, ctx.clone());
        let vec2 = CudaVec::<f64>::from_vec(vec![1.0, 2.0, 3.0], ctx.clone());
        vec1.as_view_mut().copy_from(&vec2);
        assert_eq!(vec1.clone_as_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_cuda_vec_gather() {
        let ctx = setup_cuda_context();
        let mut result = CudaVec::<f64>::zeros(3, ctx.clone());
        let vec = CudaVec::<f64>::from_vec(vec![10.0, 20.0, 30.0, 40.0], ctx.clone());
        let indices = CudaIndex::from_vec(vec![3, 0, 2], ctx.clone());
        result.gather(&vec, &indices);
        assert_eq!(result.clone_as_vec(), vec![40.0, 10.0, 30.0]);
    }

    #[test]
    fn test_cuda_vec_scatter() {
        let ctx = setup_cuda_context();
        let vec = CudaVec::<f64>::from_vec(vec![40.0, 10.0, 30.0], ctx.clone());
        let indices = CudaIndex::from_vec(vec![3, 0, 2], ctx.clone());
        let mut result = CudaVec::<f64>::zeros(4, ctx.clone());
        vec.scatter(&indices, &mut result);
        assert_eq!(result.clone_as_vec(), vec![10.0, 0.0, 30.0, 40.0]);
    }

    #[test]
    fn test_cuda_vec_component_mul_assign() {
        let ctx = setup_cuda_context();
        let mut vec1 = CudaVec::<f64>::from_vec(vec![1.0, 2.0, 3.0], ctx.clone());
        let vec2 = CudaVec::<f64>::from_vec(vec![4.0, 5.0, 6.0], ctx.clone());
        vec1.component_mul_assign(&vec2);
        assert_eq!(vec1.clone_as_vec(), vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_cuda_vec_component_div_assign() {
        let ctx = setup_cuda_context();
        let mut vec1 = CudaVec::<f64>::from_vec(vec![4.0, 10.0, 18.0], ctx.clone());
        let vec2 = CudaVec::<f64>::from_vec(vec![4.0, 5.0, 6.0], ctx.clone());
        vec1.component_div_assign(&vec2);
        assert_eq!(vec1.clone_as_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_cuda_vec_root_finding() {
        let ctx = setup_cuda_context();

        // Case 1: No root found
        let g0 = CudaVec::<f64>::from_vec(vec![1.0, -2.0, 3.0], ctx.clone());
        let g1 = CudaVec::<f64>::from_vec(vec![1.0, 2.0, 3.0], ctx.clone());
        let (found_root, max_frac, max_frac_index) = g0.root_finding(&g1);
        assert!(!found_root);
        assert_eq!(max_frac, 0.5);
        assert_eq!(max_frac_index, 1);

        // Case 2: Root found
        let g0 = CudaVec::<f64>::from_vec(vec![1.0, -2.0, 3.0], ctx.clone());
        let g1 = CudaVec::<f64>::from_vec(vec![1.0, 2.0, 0.0], ctx.clone());
        let (found_root, max_frac, max_frac_index) = g0.root_finding(&g1);
        assert!(found_root);
        assert_eq!(max_frac, 0.5);
        assert_eq!(max_frac_index, 1);

        // Case 3: No zero crossing
        let g0 = CudaVec::<f64>::from_vec(vec![1.0, -2.0, 3.0], ctx.clone());
        let g1 = CudaVec::<f64>::from_vec(vec![1.0, -2.0, 3.0], ctx.clone());
        let (found_root, max_frac, max_frac_index) = g0.root_finding(&g1);
        assert!(!found_root);
        assert_eq!(max_frac, 0.0);
        assert_eq!(max_frac_index, -1);
    }

    #[test]
    fn test_cuda_vec_assign_at_indices() {
        let ctx = setup_cuda_context();
        let mut vec = CudaVec::<f64>::zeros(5, ctx.clone());
        let indices = CudaIndex::from_vec(vec![1, 3], ctx.clone());
        vec.assign_at_indices(&indices, 42.0);
        assert_eq!(vec.clone_as_vec(), vec![0.0, 42.0, 0.0, 42.0, 0.0]);
    }

    #[test]
    fn test_cuda_vec_copy_from_indices() {
        let ctx = setup_cuda_context();
        let mut vec1 = CudaVec::<f64>::zeros(5, ctx.clone());
        let vec2 = CudaVec::<f64>::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0], ctx.clone());
        let indices = CudaIndex::from_vec(vec![0, 2, 4], ctx.clone());
        vec1.copy_from_indices(&vec2, &indices);
        assert_eq!(vec1.clone_as_vec(), vec![10.0, 0.0, 30.0, 0.0, 50.0]);
    }

    #[test]
    fn test_cuda_vec_gather() {
        let ctx = setup_cuda_context();
        let mut result = CudaVec::<f64>::zeros(3, ctx.clone());
        let vec = CudaVec::<f64>::from_vec(vec![10.0, 20.0, 30.0, 40.0], ctx.clone());
        let indices = CudaIndex::from_vec(vec![3, 0, 2], ctx.clone());
        result.gather(&vec, &indices);
        assert_eq!(result.clone_as_vec(), vec![40.0, 10.0, 30.0]);
    }

    #[test]
    fn test_cuda_vec_scatter() {
        let ctx = setup_cuda_context();
        let vec = CudaVec::<f64>::from_vec(vec![40.0, 10.0, 30.0], ctx.clone());
        let indices = CudaIndex::from_vec(vec![3, 0, 2], ctx.clone());
        let mut result = CudaVec::<f64>::zeros(4, ctx.clone());
        vec.scatter(&indices, &mut result);
        assert_eq!(result.clone_as_vec(), vec![10.0, 0.0, 30.0, 40.0]);
    }
}
