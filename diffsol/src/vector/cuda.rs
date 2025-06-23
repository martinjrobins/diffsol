use std::ffi::c_int;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Sub, SubAssign};

use super::{utils::*, VectorIndex, VectorView, VectorViewMut};
use cudarc::cublas::sys as cublas;
use cudarc::cublas::CudaBlas;
use cudarc::driver::{
    CudaFunction, CudaSlice, CudaView, CudaViewMut, DevicePtr, DevicePtrMut, LaunchConfig,
    PushKernelArg,
};

use crate::{
    CudaContext, CudaMat, CudaType, DefaultDenseMatrix, IndexType, ScalarCuda, Scale, Vector,
    VectorCommon,
};

extern "C" fn zero(_block_size: std::ffi::c_int) -> usize {
    0
}

extern "C" fn squared_norm_blk_size<T: ScalarCuda>(block_size: std::ffi::c_int) -> usize {
    (block_size * std::mem::size_of::<T>() as c_int) as usize
}

extern "C" fn root_finding_blk_size<T: ScalarCuda>(block_size: std::ffi::c_int) -> usize {
    ((block_size * std::mem::size_of::<T>() as c_int)
        + (block_size * std::mem::size_of::<c_int>() as c_int)) as usize
}

impl CudaContext {
    pub(crate) fn launch_config_1d(&self, n: u32, f: &CudaFunction) -> LaunchConfig {
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

    pub(crate) fn launch_config_1d_reduce(
        &self,
        n: u32,
        f: &CudaFunction,
        smem_size_f: extern "C" fn(block_size: std::ffi::c_int) -> usize,
    ) -> LaunchConfig {
        let (_min_grid_size, block_size) = f
            .occupancy_max_potential_block_size(smem_size_f, 0, 0, None)
            .expect("Failed to get occupancy max potential block size");
        // block_size must be a power of 2, find the previous power of 2
        // https://internals.rust-lang.org/t/add-prev-power-of-two/14281
        // n = 0 gives highest_bit_set_idx = 0.
        let highest_bit_set_idx = 31 - (block_size | 1).leading_zeros();
        // Binary AND of highest bit with n is a no-op, except zero gets wiped.
        let block_size = (1 << highest_bit_set_idx) & block_size;
        let grid_size = n.div_ceil(block_size); // Round up according to array size
        LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: smem_size_f(block_size as i32) as u32,
        }
    }

    pub(crate) fn axpy<T: ScalarCuda, D1: DevicePtr<T>, D2: DevicePtrMut<T>>(
        &self,
        alpha: T,
        x: &D1,
        y: &mut D2,
    ) {
        let n = x.len() as c_int;
        let (x, _syn_x) = x.device_ptr(&self.stream);
        let (y, _syn_y) = y.device_ptr_mut(&self.stream);
        self.axpy_inner(alpha, x, y, n);
    }

    pub(crate) fn axpy_inner<T: ScalarCuda>(&self, alpha: T, x: u64, y: u64, n: c_int) {
        let blas = CudaBlas::new(self.stream.clone()).expect("Failed to create CudaBlas");
        match T::as_enum() {
            CudaType::F64 => {
                let x = x as *const f64;
                let y = y as *mut f64;
                let alpha = alpha.as_f64();
                unsafe {
                    cublas::cublasDaxpy_v2(*blas.handle(), n, &alpha as *const f64, x, 1, y, 1)
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
                    cublas::cublasDnrm2_v2(*blas.handle(), n, x, 1, &mut result_f64 as *mut f64)
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
        assert_eq!(n, y0.len() as u32, "Length mismatch: {} != {}", n, y0.len());
        assert_eq!(
            n,
            atol.len() as u32,
            "Length mismatch: {} != {}",
            n,
            atol.len()
        );
        let f = self.function::<T>("vec_squared_norm");
        let config = self.launch_config_1d_reduce(n, &f, squared_norm_blk_size::<T>);
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
        assert_eq!(n, y0.len() as u32, "Length mismatch: {} != {}", n, y0.len());
        assert_eq!(
            n,
            atol.len() as u32,
            "Length mismatch: {} != {}",
            n,
            atol.len()
        );
        let f = self.function::<T>("vec_squared_norm");
        let config = self.launch_config_1d_reduce(n, &f, squared_norm_blk_size::<T>);
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

#[derive(Debug, Clone)]
pub struct CudaVec<T: ScalarCuda> {
    pub(crate) data: CudaSlice<T>,
    pub(crate) context: CudaContext,
}

#[derive(Debug, Clone)]
pub struct CudaIndex {
    pub(crate) data: CudaSlice<c_int>,
    pub(crate) context: CudaContext,
}

#[derive(Debug)]
pub struct CudaVecRef<'a, T: ScalarCuda> {
    pub(crate) data: CudaView<'a, T>,
    pub(crate) context: CudaContext,
}

#[derive(Debug)]
pub struct CudaVecMut<'a, T: ScalarCuda> {
    pub(crate) data: CudaViewMut<'a, T>,
    pub(crate) context: CudaContext,
}

impl<T: ScalarCuda> DefaultDenseMatrix for CudaVec<T> {
    type M = CudaMat<T>;
}

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
                if n == 0 {
                    return self;
                }
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
                if n == 0 {
                    return ret;
                }
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
                if n == 0 {
                    return ret;
                }
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
                if n == 0 {
                    return;
                }
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
                if n == 0 {
                    return;
                }
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
                if n == 0 {
                    return;
                }
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
                if n == 0 {
                    return ret;
                }
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
                if n == 0 {
                    return self;
                }
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
                if n == 0 {
                    return rhs;
                }
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
                if n == 0 {
                    return ret;
                }
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
                if n == 0 {
                    return self;
                }
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
                if n == 0 {
                    return rhs;
                }
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
            .into_iter()
            .map(|x| x as IndexType)
            .collect()
    }
    fn from_vec(v: Vec<IndexType>, ctx: Self::C) -> Self {
        let mut data = unsafe {
            ctx.stream
                .alloc(v.len())
                .expect("Failed to allocate memory for CudaVec")
        };
        let v = v.into_iter().map(|x| x as c_int).collect::<Vec<_>>();
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
    fn from_slice(slice: &[Self::T], ctx: Self::C) -> Self {
        let mut data = unsafe {
            ctx.stream
                .alloc(slice.len())
                .expect("Failed to allocate memory for CudaVec")
        };
        ctx.stream
            .memcpy_htod(slice, &mut data)
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
        if n == 0 {
            return;
        }
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
        if n == 0 {
            return;
        }
        let mut build = self.context.stream.launch_builder(&f);
        build.arg(&mut self.data).arg(&other.data).arg(&n);
        let config = self.context.launch_config_1d(n, &f);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }
    fn copy_from_view(&mut self, other: &Self::View<'_>) {
        let f = self.context.function::<T>("vec_copy");
        let n = self.len() as u32;
        if n == 0 {
            return;
        }
        let mut build = self.context.stream.launch_builder(&f);
        build.arg(&mut self.data).arg(&other.data).arg(&n);
        let config = self.context.launch_config_1d(n, &f);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }
    fn axpy(&mut self, alpha: Self::T, x: &Self, beta: Self::T) {
        if beta != T::one() {
            self.mul_assign(Scale(beta));
        }
        self.context.axpy::<T, _, _>(alpha, &x.data, &mut self.data);
    }
    fn axpy_v(&mut self, alpha: Self::T, x: &Self::View<'_>, beta: Self::T) {
        if beta != T::one() {
            self.mul_assign(Scale(beta));
        }
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
        if n == 0 {
            return;
        }
        let mut build = self.context.stream.launch_builder(&f);
        build.arg(&mut self.data).arg(&other.data).arg(&n);
        let config = self.context.launch_config_1d(n, &f);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }
    fn component_div_assign(&mut self, other: &Self) {
        let f = self.context.function::<T>("vec_div_assign");
        let n = self.len() as u32;
        if n == 0 {
            return;
        }
        let mut build = self.context.stream.launch_builder(&f);
        build.arg(&mut self.data).arg(&other.data).arg(&n);
        let config = self.context.launch_config_1d(n, &f);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }
    fn root_finding(&self, g1: &Self) -> (bool, Self::T, i32) {
        let f = self.context.function::<T>("vec_root_finding");
        let n = self.len() as u32;
        assert_eq!(
            n,
            g1.len() as u32,
            "Vector length mismatch: {} != {}",
            n,
            g1.len()
        );
        let config = self
            .context
            .launch_config_1d_reduce(n, &f, root_finding_blk_size::<T>);
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
        if n == 0 {
            return; // Skip if zero-length vectors
        }
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
        if n == 0 {
            return; // Skip if zero-length vectors
        }
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
        if n == 0 {
            return; // Skip if zero-length vectors
        }
        assert_eq!(
            indices.len(),
            self.len(),
            "Vector length mismatch: {} != {}",
            indices.len(),
            self.len()
        );
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
        if n == 0 {
            return; // Skip if zero-length vectors
        }
        let mut build = self.context.stream.launch_builder(&f);
        build.arg(&mut self.data).arg(&other.data).arg(&n);
        let config = self.context.launch_config_1d(n, &f);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }
    fn copy_from_view(&mut self, other: &Self::View) {
        let f = self.context.function::<T>("vec_copy");
        let n = self.data.len() as u32;
        if n == 0 {
            return; // Skip if zero-length vectors
        }
        let mut build = self.context.stream.launch_builder(&f);
        build.arg(&mut self.data).arg(&other.data).arg(&n);
        let config = self.context.launch_config_1d(n, &f);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }
    fn axpy(&mut self, alpha: Self::T, x: &Self::Owned, beta: Self::T) {
        if beta != T::one() {
            self.mul_assign(Scale(beta));
        }
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
        vec.fill(2.14);
        assert!(vec.clone_as_vec().iter().all(|&x| x == 2.14));
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
        y.axpy(2.0, &x, 0.0);
        assert_eq!(y.clone_as_vec(), vec![8.0, 10.0, 12.0]);
        y.axpy(0.0, &x, 1.0);
        assert_eq!(y.clone_as_vec(), vec![8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_cuda_vec_ref_squared_norm() {
        let ctx = setup_cuda_context();
        let vec1 = CudaVec::<f64>::from_vec(vec![2.0, 3.0, 4.0], ctx.clone());
        let vec2 = CudaVec::<f64>::from_vec(vec![1.0, 2.0, 3.0], ctx.clone());
        let atol = CudaVec::<f64>::from_vec(vec![0.1, 0.2, 0.3], ctx.clone());
        let rtol = 0.1;
        let norm = vec1.as_view().squared_norm(&vec2, &atol, rtol);
        let expected = [2.0, 3.0, 4.0]
            .iter()
            .zip([1.0, 2.0, 3.0].iter())
            .zip([0.1, 0.2, 0.3].iter())
            .map(|((a, b), c)| (a / (b * rtol + c)).powi(2))
            .sum::<f64>();
        assert!((norm - expected).abs() < 1e-6);
        let norm = vec1.squared_norm(&vec2, &atol, rtol);
        assert!((norm - expected).abs() < 1e-6);
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
        assert_eq!(indices.clone_as_vec(), vec![1, 3]);
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

    #[test]
    fn test_cuda_vec_get_set_index() {
        let ctx = setup_cuda_context();
        let mut vec = CudaVec::<f64>::zeros(5, ctx.clone());
        vec.set_index(2, 42.0);
        assert_eq!(vec.get_index(2), 42.0);
        assert_eq!(vec.clone_as_vec(), vec![0.0, 0.0, 42.0, 0.0, 0.0]);
        assert_eq!(vec.get_index(0), 0.0);
    }

    #[test]
    fn test_from_slice() {
        let ctx = setup_cuda_context();
        let slice = [1.0, 2.0, 3.0];
        let vec = CudaVec::from_slice(&slice, ctx.clone());
        assert_eq!(vec.clone_as_vec(), slice);
    }
}
