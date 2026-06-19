use std::ffi::c_int;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Sub, SubAssign};

use super::{VectorIndex, VectorView, VectorViewMut};
use cudarc::cublas::sys as cublas;
use cudarc::cublas::CudaBlas;
use cudarc::driver::{
    CudaFunction, CudaSlice, CudaView, CudaViewMut, DevicePtr, DevicePtrMut,
    LaunchConfig, PushKernelArg,
};

use crate::{
    CudaContext, CudaMat, CudaType, DefaultDenseMatrix, IndexType, ScalarCuda, Scale, Vector,
    VectorCommon, Context,
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
                result = T::from_f64(result_f64).unwrap();
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
            .clone_dtoh(&partial_sums)
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
            .clone_dtoh(&partial_sums)
            .expect("Failed to copy data from device to host");
        partial_sums.into_iter().fold(T::zero(), |acc, x| acc + x)
    }

    pub(crate) fn squared_norm_all_views<T: ScalarCuda>(
        &self,
        y: &CudaView<T>,
        y0: &CudaView<T>,
        atol: &CudaView<T>,
        rtol: T,
    ) -> T {
        let n = y.len() as u32;
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
            .clone_dtoh(&partial_sums)
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
    pub(crate) nstates: IndexType,
    pub(crate) col_offset: IndexType,
}

#[derive(Debug)]
pub struct CudaVecMut<'a, T: ScalarCuda> {
    pub(crate) data: CudaViewMut<'a, T>,
    pub(crate) context: CudaContext,
    pub(crate) nstates: IndexType,
    pub(crate) col_offset: IndexType,
}

impl<T: ScalarCuda> DefaultDenseMatrix for CudaVec<T> {
    type M = CudaMat<T>;
}

impl<'a, T: ScalarCuda> CudaVecRef<'a, T> {
    pub(crate) fn stride(&self) -> IndexType {
        self.data.len() as IndexType / self.context.nbatch()
    }

    pub(crate) fn new(
        data: CudaView<'a, T>,
        context: CudaContext,
        nstates: IndexType,
        col_offset: IndexType,
    ) -> Self {
        Self {
            data,
            context,
            nstates,
            col_offset,
        }
    }
}

impl<'a, T: ScalarCuda> CudaVecMut<'a, T> {
    pub(crate) fn stride(&self) -> IndexType {
        self.data.len() as IndexType / self.context.nbatch()
    }

    pub(crate) fn new(
        data: CudaViewMut<'a, T>,
        context: CudaContext,
        nstates: IndexType,
        col_offset: IndexType,
    ) -> Self {
        Self {
            data,
            context,
            nstates,
            col_offset,
        }
    }
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
impl_div_scalar!(CudaVec<T>, CudaVec<T>, T);
impl_mul_assign_scalar!(CudaVec<T>, T);

// Explicit batched-aware Mul<Scale> for CudaVecRef -> CudaVec
impl<T: ScalarCuda> Mul<Scale<T>> for CudaVecRef<'_, T> {
    type Output = CudaVec<T>;
    fn mul(self, rhs: Scale<T>) -> Self::Output {
        let nbatch = self.context.nbatch();
        let mut ret = CudaVec::zeros(self.nstates, self.context.clone());
        let f = self.context.function::<T>("vec_mul_scalar");
        let scalar = rhs.value();
        for b in 0..nbatch {
            let self_start = b * self.stride() + self.col_offset;
            let dst_start = b * self.nstates;
            let n = self.nstates as u32;
            if n == 0 {
                continue;
            }
            let config = self.context.launch_config_1d(n, &f);
            let mut build = self.context.stream.launch_builder(&f);
            unsafe {
                let src_slice = self.data.slice(self_start..self_start + self.nstates);
                let mut dst_slice = ret.data.slice_mut(dst_start..dst_start + self.nstates);
                build
                    .arg(&src_slice)
                    .arg(&scalar)
                    .arg(&mut dst_slice)
                    .arg(&n);
                build.launch(config)
            }
            .expect("Failed to launch kernel");
        }
        ret
    }
}

// Stride-aware MulAssign<Scale> for CudaVecMut
impl<T: ScalarCuda> MulAssign<Scale<T>> for CudaVecMut<'_, T> {
    fn mul_assign(&mut self, rhs: Scale<T>) {
        let nbatch = self.context.nbatch();
        let stride = self.stride();
        let f = self.context.function::<T>("vec_mul_assign_scalar");
        let scalar = rhs.value();
        for b in 0..nbatch {
            let start = b * stride + self.col_offset;
            let n = self.nstates as u32;
            if n == 0 {
                continue;
            }
            let config = self.context.launch_config_1d(n, &f);
            let mut build = self.context.stream.launch_builder(&f);
            unsafe {
                let mut slice = self.data.slice_mut(start..start + self.nstates);
                build.arg(&mut slice).arg(&scalar).arg(&n);
                build.launch(config)
            }
            .expect("Failed to launch kernel");
        }
    }
}

// Stride-aware Mul<Scale> for CudaVecMut -> CudaVec
impl<T: ScalarCuda> Mul<Scale<T>> for CudaVecMut<'_, T> {
    type Output = CudaVec<T>;
    fn mul(self, rhs: Scale<T>) -> Self::Output {
        let nbatch = self.context.nbatch();
        let mut ret = CudaVec::zeros(self.nstates, self.context.clone());
        let f = self.context.function::<T>("vec_mul_scalar");
        let scalar = rhs.value();
        for b in 0..nbatch {
            let self_start = b * self.stride() + self.col_offset;
            let dst_start = b * self.nstates;
            let n = self.nstates as u32;
            if n == 0 {
                continue;
            }
            let config = self.context.launch_config_1d(n, &f);
            let mut build = self.context.stream.launch_builder(&f);
            let self_data = self.data.slice(self_start..self_start + self.nstates);
            unsafe {
                let mut dst_slice = ret.data.slice_mut(dst_start..dst_start + self.nstates);
                build
                    .arg(&self_data)
                    .arg(&scalar)
                    .arg(&mut dst_slice)
                    .arg(&n);
                build.launch(config)
            }
            .expect("Failed to launch kernel");
        }
        ret
    }
}

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

// ============================================================
// Per-batch Add/Sub operations on CudaVec (owned, always contiguous)
//
// RHS may be: &CudaVec (potentially broadcast nbatch=1),
//             CudaVecRef (potentially strided with col_offset)
// ============================================================

impl<T: ScalarCuda> SubAssign<CudaVec<T>> for CudaVec<T> {
    fn sub_assign(&mut self, rhs: CudaVec<T>) {
        self.sub_assign(&rhs);
    }
}
impl<T: ScalarCuda> SubAssign<&CudaVec<T>> for CudaVec<T> {
    fn sub_assign(&mut self, rhs: &CudaVec<T>) {
        let self_nbatch = self.context.nbatch();
        let other_nbatch = rhs.context.nbatch();
        self.context.assert_compatible_nbatch(other_nbatch, "sub_assign");
        sub_assign_owned_batched(self, rhs, self_nbatch, other_nbatch);
    }
}
impl<T: ScalarCuda> SubAssign<CudaVecRef<'_, T>> for CudaVec<T> {
    fn sub_assign(&mut self, rhs: CudaVecRef<'_, T>) {
        self.sub_assign(&rhs);
    }
}
impl<T: ScalarCuda> SubAssign<&CudaVecRef<'_, T>> for CudaVec<T> {
    fn sub_assign(&mut self, rhs: &CudaVecRef<'_, T>) {
        let self_nbatch = self.context.nbatch();
        let other_nbatch = rhs.context.nbatch();
        self.context.assert_compatible_nbatch(other_nbatch, "sub_assign");
        sub_assign_owned_view(self, rhs, self_nbatch, other_nbatch);
    }
}

impl<T: ScalarCuda> AddAssign<CudaVec<T>> for CudaVec<T> {
    fn add_assign(&mut self, rhs: CudaVec<T>) {
        self.add_assign(&rhs);
    }
}
impl<T: ScalarCuda> AddAssign<&CudaVec<T>> for CudaVec<T> {
    fn add_assign(&mut self, rhs: &CudaVec<T>) {
        let self_nbatch = self.context.nbatch();
        let other_nbatch = rhs.context.nbatch();
        self.context.assert_compatible_nbatch(other_nbatch, "add_assign");
        add_assign_owned_batched(self, rhs, self_nbatch, other_nbatch);
    }
}
impl<T: ScalarCuda> AddAssign<CudaVecRef<'_, T>> for CudaVec<T> {
    fn add_assign(&mut self, rhs: CudaVecRef<'_, T>) {
        self.add_assign(&rhs);
    }
}
impl<T: ScalarCuda> AddAssign<&CudaVecRef<'_, T>> for CudaVec<T> {
    fn add_assign(&mut self, rhs: &CudaVecRef<'_, T>) {
        let self_nbatch = self.context.nbatch();
        let other_nbatch = rhs.context.nbatch();
        self.context.assert_compatible_nbatch(other_nbatch, "add_assign");
        add_assign_owned_view(self, rhs, self_nbatch, other_nbatch);
    }
}

fn sub_assign_owned_batched<T: ScalarCuda>(
    this: &mut CudaVec<T>,
    rhs: &CudaVec<T>,
    self_nbatch: usize,
    other_nbatch: usize,
) {
    let nstates = this.data.len() as IndexType / self_nbatch;
    let other_nstates = rhs.data.len() as IndexType / other_nbatch;
    let f = this.context.function::<T>("vec_sub_assign");
    for b in 0..self_nbatch {
        let other_b = if other_nbatch == 1 { 0 } else { b };
        let self_start = b * nstates;
        let other_start = other_b * other_nstates;
        let n = nstates as u32;
        if n == 0 { continue; }
        let config = this.context.launch_config_1d(n, &f);
        let mut build = this.context.stream.launch_builder(&f);
        let mut self_slice = this.data.slice_mut(self_start..self_start + nstates);
        let other_slice = rhs.data.slice(other_start..other_start + other_nstates);
        build.arg(&mut self_slice).arg(&other_slice).arg(&n);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }
}

fn add_assign_owned_batched<T: ScalarCuda>(
    this: &mut CudaVec<T>,
    rhs: &CudaVec<T>,
    self_nbatch: usize,
    other_nbatch: usize,
) {
    let nstates = this.data.len() as IndexType / self_nbatch;
    let other_nstates = rhs.data.len() as IndexType / other_nbatch;
    let f = this.context.function::<T>("vec_add_assign");
    for b in 0..self_nbatch {
        let other_b = if other_nbatch == 1 { 0 } else { b };
        let self_start = b * nstates;
        let other_start = other_b * other_nstates;
        let n = nstates as u32;
        if n == 0 { continue; }
        let config = this.context.launch_config_1d(n, &f);
        let mut build = this.context.stream.launch_builder(&f);
        let mut self_slice = this.data.slice_mut(self_start..self_start + nstates);
        let other_slice = rhs.data.slice(other_start..other_start + other_nstates);
        build.arg(&mut self_slice).arg(&other_slice).arg(&n);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }
}

fn sub_assign_owned_view<T: ScalarCuda>(
    this: &mut CudaVec<T>,
    rhs: &CudaVecRef<'_, T>,
    self_nbatch: usize,
    other_nbatch: usize,
) {
    let nstates = this.data.len() as IndexType / self_nbatch;
    let rhs_stride = rhs.stride();
    let rhs_nstates = rhs.nstates;
    let f = this.context.function::<T>("vec_sub_assign");
    for b in 0..self_nbatch {
        let other_b = if other_nbatch == 1 { 0 } else { b };
        let self_start = b * nstates;
        let other_start = other_b * rhs_stride + rhs.col_offset;
        let n = nstates as u32;
        if n == 0 { continue; }
        let config = this.context.launch_config_1d(n, &f);
        let mut build = this.context.stream.launch_builder(&f);
        let mut self_slice = this.data.slice_mut(self_start..self_start + nstates);
        let other_slice = rhs.data.slice(other_start..other_start + rhs_nstates);
        build.arg(&mut self_slice).arg(&other_slice).arg(&n);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }
}

fn add_assign_owned_view<T: ScalarCuda>(
    this: &mut CudaVec<T>,
    rhs: &CudaVecRef<'_, T>,
    self_nbatch: usize,
    other_nbatch: usize,
) {
    let nstates = this.data.len() as IndexType / self_nbatch;
    let rhs_stride = rhs.stride();
    let rhs_nstates = rhs.nstates;
    let f = this.context.function::<T>("vec_add_assign");
    for b in 0..self_nbatch {
        let other_b = if other_nbatch == 1 { 0 } else { b };
        let self_start = b * nstates;
        let other_start = other_b * rhs_stride + rhs.col_offset;
        let n = nstates as u32;
        if n == 0 { continue; }
        let config = this.context.launch_config_1d(n, &f);
        let mut build = this.context.stream.launch_builder(&f);
        let mut self_slice = this.data.slice_mut(self_start..self_start + nstates);
        let other_slice = rhs.data.slice(other_start..other_start + rhs_nstates);
        build.arg(&mut self_slice).arg(&other_slice).arg(&n);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }
}

// Stride-aware SubAssign for CudaVecMut with owned CudaVec
impl<T: ScalarCuda> SubAssign<CudaVec<T>> for CudaVecMut<'_, T> {
    fn sub_assign(&mut self, rhs: CudaVec<T>) {
        self.sub_assign(&rhs);
    }
}
impl<T: ScalarCuda> SubAssign<&CudaVec<T>> for CudaVecMut<'_, T> {
    fn sub_assign(&mut self, rhs: &CudaVec<T>) {
        let nbatch = self.context.nbatch();
        let other_nbatch = rhs.context.nbatch();
        self.context.assert_compatible_nbatch(other_nbatch, "sub_assign");
        let other_nstates = rhs.data.len() as IndexType / other_nbatch;
        let f = self.context.function::<T>("vec_sub_assign");
        for b in 0..nbatch {
            let other_b = if other_nbatch == 1 { 0 } else { b };
            let self_start = b * self.stride() + self.col_offset;
            let other_start = other_b * other_nstates;
            let n = self.nstates as u32;
            if n == 0 {
                continue;
            }
            let config = self.context.launch_config_1d(n, &f);
            let mut build = self.context.stream.launch_builder(&f);
            let mut self_slice = self.data.slice_mut(self_start..self_start + self.nstates);
            let other_slice = rhs.data.slice(other_start..other_start + other_nstates);
            build.arg(&mut self_slice).arg(&other_slice).arg(&n);
            unsafe { build.launch(config) }.expect("Failed to launch kernel");
        }
    }
}

// Stride-aware AddAssign for CudaVecMut with owned CudaVec
impl<T: ScalarCuda> AddAssign<CudaVec<T>> for CudaVecMut<'_, T> {
    fn add_assign(&mut self, rhs: CudaVec<T>) {
        self.add_assign(&rhs);
    }
}
impl<T: ScalarCuda> AddAssign<&CudaVec<T>> for CudaVecMut<'_, T> {
    fn add_assign(&mut self, rhs: &CudaVec<T>) {
        let nbatch = self.context.nbatch();
        let other_nbatch = rhs.context.nbatch();
        self.context.assert_compatible_nbatch(other_nbatch, "add_assign");
        let other_nstates = rhs.data.len() as IndexType / other_nbatch;
        let f = self.context.function::<T>("vec_add_assign");
        for b in 0..nbatch {
            let other_b = if other_nbatch == 1 { 0 } else { b };
            let self_start = b * self.stride() + self.col_offset;
            let other_start = other_b * other_nstates;
            let n = self.nstates as u32;
            if n == 0 {
                continue;
            }
            let config = self.context.launch_config_1d(n, &f);
            let mut build = self.context.stream.launch_builder(&f);
            let mut self_slice = self.data.slice_mut(self_start..self_start + self.nstates);
            let other_slice = rhs.data.slice(other_start..other_start + other_nstates);
            build.arg(&mut self_slice).arg(&other_slice).arg(&n);
            unsafe { build.launch(config) }.expect("Failed to launch kernel");
        }
    }
}

// Stride-aware SubAssign for CudaVecMut with CudaVecRef RHS
impl<T: ScalarCuda> SubAssign<CudaVecRef<'_, T>> for CudaVecMut<'_, T> {
    fn sub_assign(&mut self, rhs: CudaVecRef<'_, T>) {
        self.sub_assign(&rhs);
    }
}
impl<T: ScalarCuda> SubAssign<&CudaVecRef<'_, T>> for CudaVecMut<'_, T> {
    fn sub_assign(&mut self, rhs: &CudaVecRef<'_, T>) {
        let nbatch = self.context.nbatch();
        let other_nbatch = rhs.context.nbatch();
        self.context.assert_compatible_nbatch(other_nbatch, "sub_assign");
        let f = self.context.function::<T>("vec_sub_assign");
        for b in 0..nbatch {
            let other_b = if other_nbatch == 1 { 0 } else { b };
            let self_start = b * self.stride() + self.col_offset;
            let other_stride = rhs.stride();
            let other_start = other_b * other_stride + rhs.col_offset;
            let n = self.nstates as u32;
            if n == 0 {
                continue;
            }
            let config = self.context.launch_config_1d(n, &f);
            let mut build = self.context.stream.launch_builder(&f);
            let mut self_slice = self.data.slice_mut(self_start..self_start + self.nstates);
            let other_slice = rhs.data.slice(other_start..other_start + rhs.nstates);
            build.arg(&mut self_slice).arg(&other_slice).arg(&n);
            unsafe { build.launch(config) }.expect("Failed to launch kernel");
        }
    }
}

// Stride-aware AddAssign for CudaVecMut with CudaVecRef RHS
impl<T: ScalarCuda> AddAssign<CudaVecRef<'_, T>> for CudaVecMut<'_, T> {
    fn add_assign(&mut self, rhs: CudaVecRef<'_, T>) {
        self.add_assign(&rhs);
    }
}
impl<T: ScalarCuda> AddAssign<&CudaVecRef<'_, T>> for CudaVecMut<'_, T> {
    fn add_assign(&mut self, rhs: &CudaVecRef<'_, T>) {
        let nbatch = self.context.nbatch();
        let other_nbatch = rhs.context.nbatch();
        self.context.assert_compatible_nbatch(other_nbatch, "add_assign");
        let f = self.context.function::<T>("vec_add_assign");
        for b in 0..nbatch {
            let other_b = if other_nbatch == 1 { 0 } else { b };
            let self_start = b * self.stride() + self.col_offset;
            let other_stride = rhs.stride();
            let other_start = other_b * other_stride + rhs.col_offset;
            let n = self.nstates as u32;
            if n == 0 {
                continue;
            }
            let config = self.context.launch_config_1d(n, &f);
            let mut build = self.context.stream.launch_builder(&f);
            let mut self_slice = self.data.slice_mut(self_start..self_start + self.nstates);
            let other_slice = rhs.data.slice(other_start..other_start + rhs.nstates);
            build.arg(&mut self_slice).arg(&other_slice).arg(&n);
            unsafe { build.launch(config) }.expect("Failed to launch kernel");
        }
    }
}

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

// ============================================================
// Sub operations: a - b -> CudaVec (new owned)
// ============================================================

// &CudaVec - &CudaVec -> CudaVec
impl<T: ScalarCuda> Sub<&CudaVec<T>> for &CudaVec<T> {
    type Output = CudaVec<T>;
    fn sub(self, rhs: &CudaVec<T>) -> CudaVec<T> {
        let self_nbatch = self.context.nbatch();
        let other_nbatch = rhs.context.nbatch();
        self.context.assert_compatible_nbatch(other_nbatch, "sub");
        let nstates = self.data.len() as IndexType / self_nbatch;
        let other_nstates = rhs.data.len() as IndexType / other_nbatch;
        let mut ret = CudaVec::zeros(nstates, self.context.clone());
        for b in 0..self_nbatch {
            let other_b = if other_nbatch == 1 { 0 } else { b };
            let f = self.context.function::<T>("vec_sub");
            let n = nstates as u32;
            if n == 0 { continue; }
            let config = self.context.launch_config_1d(n, &f);
            let mut build = self.context.stream.launch_builder(&f);
            let self_slice = self.data.slice(b * nstates..(b + 1) * nstates);
            let other_slice = rhs.data.slice(other_b * other_nstates..(other_b + 1) * other_nstates);
            let mut dst_slice = ret.data.slice_mut(b * nstates..(b + 1) * nstates);
            build.arg(&self_slice).arg(&other_slice).arg(&mut dst_slice).arg(&n);
            unsafe { build.launch(config) }.expect("Failed to launch kernel");
        }
        ret
    }
}

// &CudaVec - CudaVecRef -> CudaVec
impl<T: ScalarCuda> Sub<CudaVecRef<'_, T>> for &CudaVec<T> {
    type Output = CudaVec<T>;
    fn sub(self, rhs: CudaVecRef<'_, T>) -> CudaVec<T> { self.sub(&rhs) }
}
impl<T: ScalarCuda> Sub<&CudaVecRef<'_, T>> for &CudaVec<T> {
    type Output = CudaVec<T>;
    fn sub(self, rhs: &CudaVecRef<'_, T>) -> CudaVec<T> {
        let self_nbatch = self.context.nbatch();
        let other_nbatch = rhs.context.nbatch();
        self.context.assert_compatible_nbatch(other_nbatch, "sub");
        let nstates = self.data.len() as IndexType / self_nbatch;
        let rhs_stride = rhs.stride();
        let rhs_nstates = rhs.nstates;
        let mut ret = CudaVec::zeros(nstates, self.context.clone());
        for b in 0..self_nbatch {
            let other_b = if other_nbatch == 1 { 0 } else { b };
            let other_start = other_b * rhs_stride + rhs.col_offset;
            let f = self.context.function::<T>("vec_sub");
            let n = nstates as u32;
            if n == 0 { continue; }
            let config = self.context.launch_config_1d(n, &f);
            let mut build = self.context.stream.launch_builder(&f);
            let self_slice = self.data.slice(b * nstates..(b + 1) * nstates);
            let other_slice = rhs.data.slice(other_start..other_start + rhs_nstates);
            let mut dst_slice = ret.data.slice_mut(b * nstates..(b + 1) * nstates);
            build.arg(&self_slice).arg(&other_slice).arg(&mut dst_slice).arg(&n);
            unsafe { build.launch(config) }.expect("Failed to launch kernel");
        }
        ret
    }
}

// CudaVec - CudaVec -> CudaVec (owned by value)
impl<T: ScalarCuda> Sub<CudaVec<T>> for CudaVec<T> {
    type Output = CudaVec<T>;
    fn sub(mut self, rhs: CudaVec<T>) -> CudaVec<T> { self.sub_assign(&rhs); self }
}

// CudaVec + CudaVec -> CudaVec (owned by value)
impl<T: ScalarCuda> Add<CudaVec<T>> for CudaVec<T> {
    type Output = CudaVec<T>;
    fn add(mut self, rhs: CudaVec<T>) -> CudaVec<T> { self.add_assign(&rhs); self }
}

// CudaVec - &CudaVec -> CudaVec (in-place on self)
impl<T: ScalarCuda> Sub<&CudaVec<T>> for CudaVec<T> {
    type Output = CudaVec<T>;
    fn sub(mut self, rhs: &CudaVec<T>) -> CudaVec<T> {
        self.sub_assign(rhs);
        self
    }
}

// CudaVec - CudaVecRef -> CudaVec (in-place on self)
impl<T: ScalarCuda> Sub<CudaVecRef<'_, T>> for CudaVec<T> {
    type Output = CudaVec<T>;
    fn sub(mut self, rhs: CudaVecRef<'_, T>) -> CudaVec<T> { self.sub_assign(&rhs); self }
}
impl<T: ScalarCuda> Sub<&CudaVecRef<'_, T>> for CudaVec<T> {
    type Output = CudaVec<T>;
    fn sub(mut self, rhs: &CudaVecRef<'_, T>) -> CudaVec<T> { self.sub_assign(rhs); self }
}

// CudaVecRef - CudaVec -> CudaVec
impl<T: ScalarCuda> Sub<CudaVec<T>> for CudaVecRef<'_, T> {
    type Output = CudaVec<T>;
    fn sub(self, rhs: CudaVec<T>) -> CudaVec<T> { &self - &rhs }
}

// CudaVecRef - &CudaVec -> CudaVec
impl<T: ScalarCuda> Sub<&CudaVec<T>> for CudaVecRef<'_, T> {
    type Output = CudaVec<T>;
    fn sub(self, rhs: &CudaVec<T>) -> CudaVec<T> { sub_view_minus_owned(&self, rhs) }
}
impl<T: ScalarCuda> Sub<&CudaVec<T>> for &CudaVecRef<'_, T> {
    type Output = CudaVec<T>;
    fn sub(self, rhs: &CudaVec<T>) -> CudaVec<T> { sub_view_minus_owned(self, rhs) }
}

// CudaVecRef - CudaVecRef -> CudaVec
impl<T: ScalarCuda> Sub<CudaVecRef<'_, T>> for CudaVecRef<'_, T> {
    type Output = CudaVec<T>;
    fn sub(self, rhs: CudaVecRef<'_, T>) -> CudaVec<T> { &self - &rhs }
}
impl<T: ScalarCuda> Sub<&CudaVecRef<'_, T>> for CudaVecRef<'_, T> {
    type Output = CudaVec<T>;
    fn sub(self, rhs: &CudaVecRef<'_, T>) -> CudaVec<T> { sub_view_minus_view(&self, rhs) }
}
impl<T: ScalarCuda> Sub<&CudaVecRef<'_, T>> for &CudaVecRef<'_, T> {
    type Output = CudaVec<T>;
    fn sub(self, rhs: &CudaVecRef<'_, T>) -> CudaVec<T> { sub_view_minus_view(self, rhs) }
}

fn sub_view_minus_owned<T: ScalarCuda>(lhs: &CudaVecRef<'_, T>, rhs: &CudaVec<T>) -> CudaVec<T> {
    let self_nbatch = lhs.context.nbatch();
    let other_nbatch = rhs.context.nbatch();
    lhs.context.assert_compatible_nbatch(other_nbatch, "sub");
    let stride = lhs.stride();
    let nstates = lhs.nstates;
    let other_nstates = rhs.data.len() as IndexType / other_nbatch;
    let mut ret = CudaVec::zeros(nstates, lhs.context.clone());
    for b in 0..self_nbatch {
        let other_b = if other_nbatch == 1 { 0 } else { b };
        let self_start = b * stride + lhs.col_offset;
        let other_start = other_b * other_nstates;
        let f = lhs.context.function::<T>("vec_sub");
        let n = nstates as u32;
        if n == 0 { continue; }
        let config = lhs.context.launch_config_1d(n, &f);
        let mut build = lhs.context.stream.launch_builder(&f);
        let self_slice = lhs.data.slice(self_start..self_start + nstates);
        let other_slice = rhs.data.slice(other_start..other_start + other_nstates);
        let mut dst_slice = ret.data.slice_mut(b * nstates..(b + 1) * nstates);
        build.arg(&self_slice).arg(&other_slice).arg(&mut dst_slice).arg(&n);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }
    ret
}

fn sub_view_minus_view<T: ScalarCuda>(lhs: &CudaVecRef<'_, T>, rhs: &CudaVecRef<'_, T>) -> CudaVec<T> {
    let self_nbatch = lhs.context.nbatch();
    let other_nbatch = rhs.context.nbatch();
    lhs.context.assert_compatible_nbatch(other_nbatch, "sub");
    let self_stride = lhs.stride();
    let rhs_stride = rhs.stride();
    let nstates = lhs.nstates;
    let mut ret = CudaVec::zeros(nstates, lhs.context.clone());
    for b in 0..self_nbatch {
        let other_b = if other_nbatch == 1 { 0 } else { b };
        let self_start = b * self_stride + lhs.col_offset;
        let other_start = other_b * rhs_stride + rhs.col_offset;
        let f = lhs.context.function::<T>("vec_sub");
        let n = nstates as u32;
        if n == 0 { continue; }
        let config = lhs.context.launch_config_1d(n, &f);
        let mut build = lhs.context.stream.launch_builder(&f);
        let self_slice = lhs.data.slice(self_start..self_start + nstates);
        let other_slice = rhs.data.slice(other_start..other_start + rhs.nstates);
        let mut dst_slice = ret.data.slice_mut(b * nstates..(b + 1) * nstates);
        build.arg(&self_slice).arg(&other_slice).arg(&mut dst_slice).arg(&n);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }
    ret
}

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

// ============================================================
// Add operations: a + b -> CudaVec (new owned)
// ============================================================

// &CudaVec + CudaVec -> CudaVec
impl<T: ScalarCuda> Add<CudaVec<T>> for &CudaVec<T> {
    type Output = CudaVec<T>;
    fn add(self, rhs: CudaVec<T>) -> CudaVec<T> { self.add(&rhs) }
}
// &CudaVec - CudaVec -> CudaVec
impl<T: ScalarCuda> Sub<CudaVec<T>> for &CudaVec<T> {
    type Output = CudaVec<T>;
    fn sub(self, rhs: CudaVec<T>) -> CudaVec<T> { self.sub(&rhs) }
}

// &CudaVec + &CudaVec -> CudaVec
impl<T: ScalarCuda> Add<&CudaVec<T>> for &CudaVec<T> {
    type Output = CudaVec<T>;
    fn add(self, rhs: &CudaVec<T>) -> CudaVec<T> {
        let self_nbatch = self.context.nbatch();
        let other_nbatch = rhs.context.nbatch();
        self.context.assert_compatible_nbatch(other_nbatch, "add");
        let nstates = self.data.len() as IndexType / self_nbatch;
        let other_nstates = rhs.data.len() as IndexType / other_nbatch;
        let mut ret = CudaVec::zeros(nstates, self.context.clone());
        for b in 0..self_nbatch {
            let other_b = if other_nbatch == 1 { 0 } else { b };
            let f = self.context.function::<T>("vec_add");
            let n = nstates as u32;
            if n == 0 { continue; }
            let config = self.context.launch_config_1d(n, &f);
            let mut build = self.context.stream.launch_builder(&f);
            let self_slice = self.data.slice(b * nstates..(b + 1) * nstates);
            let other_slice = rhs.data.slice(other_b * other_nstates..(other_b + 1) * other_nstates);
            let mut dst_slice = ret.data.slice_mut(b * nstates..(b + 1) * nstates);
            build.arg(&self_slice).arg(&other_slice).arg(&mut dst_slice).arg(&n);
            unsafe { build.launch(config) }.expect("Failed to launch kernel");
        }
        ret
    }
}

// &CudaVec + CudaVecRef -> CudaVec
impl<T: ScalarCuda> Add<CudaVecRef<'_, T>> for &CudaVec<T> {
    type Output = CudaVec<T>;
    fn add(self, rhs: CudaVecRef<'_, T>) -> CudaVec<T> { self.add(&rhs) }
}
impl<T: ScalarCuda> Add<&CudaVecRef<'_, T>> for &CudaVec<T> {
    type Output = CudaVec<T>;
    fn add(self, rhs: &CudaVecRef<'_, T>) -> CudaVec<T> {
        let self_nbatch = self.context.nbatch();
        let other_nbatch = rhs.context.nbatch();
        self.context.assert_compatible_nbatch(other_nbatch, "add");
        let nstates = self.data.len() as IndexType / self_nbatch;
        let rhs_stride = rhs.stride();
        let rhs_nstates = rhs.nstates;
        let mut ret = CudaVec::zeros(nstates, self.context.clone());
        for b in 0..self_nbatch {
            let other_b = if other_nbatch == 1 { 0 } else { b };
            let other_start = other_b * rhs_stride + rhs.col_offset;
            let f = self.context.function::<T>("vec_add");
            let n = nstates as u32;
            if n == 0 { continue; }
            let config = self.context.launch_config_1d(n, &f);
            let mut build = self.context.stream.launch_builder(&f);
            let self_slice = self.data.slice(b * nstates..(b + 1) * nstates);
            let other_slice = rhs.data.slice(other_start..other_start + rhs_nstates);
            let mut dst_slice = ret.data.slice_mut(b * nstates..(b + 1) * nstates);
            build.arg(&self_slice).arg(&other_slice).arg(&mut dst_slice).arg(&n);
            unsafe { build.launch(config) }.expect("Failed to launch kernel");
        }
        ret
    }
}

// CudaVec + &CudaVec -> CudaVec (in-place on self)
impl<T: ScalarCuda> Add<&CudaVec<T>> for CudaVec<T> {
    type Output = CudaVec<T>;
    fn add(mut self, rhs: &CudaVec<T>) -> CudaVec<T> {
        self.add_assign(rhs);
        self
    }
}

// CudaVec + CudaVecRef -> CudaVec (in-place on self)
impl<T: ScalarCuda> Add<CudaVecRef<'_, T>> for CudaVec<T> {
    type Output = CudaVec<T>;
    fn add(mut self, rhs: CudaVecRef<'_, T>) -> CudaVec<T> { self.add_assign(&rhs); self }
}
impl<T: ScalarCuda> Add<&CudaVecRef<'_, T>> for CudaVec<T> {
    type Output = CudaVec<T>;
    fn add(mut self, rhs: &CudaVecRef<'_, T>) -> CudaVec<T> { self.add_assign(rhs); self }
}

// CudaVecRef + CudaVec -> CudaVec
impl<T: ScalarCuda> Add<CudaVec<T>> for CudaVecRef<'_, T> {
    type Output = CudaVec<T>;
    fn add(self, rhs: CudaVec<T>) -> CudaVec<T> { &self + &rhs }
}

// CudaVecRef + &CudaVec -> CudaVec
impl<T: ScalarCuda> Add<&CudaVec<T>> for CudaVecRef<'_, T> {
    type Output = CudaVec<T>;
    fn add(self, rhs: &CudaVec<T>) -> CudaVec<T> { add_view_plus_owned(&self, rhs) }
}
impl<T: ScalarCuda> Add<&CudaVec<T>> for &CudaVecRef<'_, T> {
    type Output = CudaVec<T>;
    fn add(self, rhs: &CudaVec<T>) -> CudaVec<T> { add_view_plus_owned(self, rhs) }
}

// CudaVecRef + CudaVecRef -> CudaVec
impl<T: ScalarCuda> Add<CudaVecRef<'_, T>> for CudaVecRef<'_, T> {
    type Output = CudaVec<T>;
    fn add(self, rhs: CudaVecRef<'_, T>) -> CudaVec<T> { &self + &rhs }
}
impl<T: ScalarCuda> Add<&CudaVecRef<'_, T>> for CudaVecRef<'_, T> {
    type Output = CudaVec<T>;
    fn add(self, rhs: &CudaVecRef<'_, T>) -> CudaVec<T> { add_view_plus_view(&self, rhs) }
}
impl<T: ScalarCuda> Add<&CudaVecRef<'_, T>> for &CudaVecRef<'_, T> {
    type Output = CudaVec<T>;
    fn add(self, rhs: &CudaVecRef<'_, T>) -> CudaVec<T> { add_view_plus_view(self, rhs) }
}

fn add_view_plus_owned<T: ScalarCuda>(lhs: &CudaVecRef<'_, T>, rhs: &CudaVec<T>) -> CudaVec<T> {
    let self_nbatch = lhs.context.nbatch();
    let other_nbatch = rhs.context.nbatch();
    lhs.context.assert_compatible_nbatch(other_nbatch, "add");
    let stride = lhs.stride();
    let nstates = lhs.nstates;
    let other_nstates = rhs.data.len() as IndexType / other_nbatch;
    let mut ret = CudaVec::zeros(nstates, lhs.context.clone());
    for b in 0..self_nbatch {
        let other_b = if other_nbatch == 1 { 0 } else { b };
        let self_start = b * stride + lhs.col_offset;
        let other_start = other_b * other_nstates;
        let f = lhs.context.function::<T>("vec_add");
        let n = nstates as u32;
        if n == 0 { continue; }
        let config = lhs.context.launch_config_1d(n, &f);
        let mut build = lhs.context.stream.launch_builder(&f);
        let self_slice = lhs.data.slice(self_start..self_start + nstates);
        let other_slice = rhs.data.slice(other_start..other_start + other_nstates);
        let mut dst_slice = ret.data.slice_mut(b * nstates..(b + 1) * nstates);
        build.arg(&self_slice).arg(&other_slice).arg(&mut dst_slice).arg(&n);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }
    ret
}

fn add_view_plus_view<T: ScalarCuda>(lhs: &CudaVecRef<'_, T>, rhs: &CudaVecRef<'_, T>) -> CudaVec<T> {
    let self_nbatch = lhs.context.nbatch();
    let other_nbatch = rhs.context.nbatch();
    lhs.context.assert_compatible_nbatch(other_nbatch, "add");
    let self_stride = lhs.stride();
    let rhs_stride = rhs.stride();
    let nstates = lhs.nstates;
    let mut ret = CudaVec::zeros(nstates, lhs.context.clone());
    for b in 0..self_nbatch {
        let other_b = if other_nbatch == 1 { 0 } else { b };
        let self_start = b * self_stride + lhs.col_offset;
        let other_start = other_b * rhs_stride + rhs.col_offset;
        let f = lhs.context.function::<T>("vec_add");
        let n = nstates as u32;
        if n == 0 { continue; }
        let config = lhs.context.launch_config_1d(n, &f);
        let mut build = lhs.context.stream.launch_builder(&f);
        let self_slice = lhs.data.slice(self_start..self_start + nstates);
        let other_slice = rhs.data.slice(other_start..other_start + rhs.nstates);
        let mut dst_slice = ret.data.slice_mut(b * nstates..(b + 1) * nstates);
        build.arg(&self_slice).arg(&other_slice).arg(&mut dst_slice).arg(&n);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }
    ret
}

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
            .clone_dtoh(&self.data)
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
        let nbatch = self.context.nbatch();
        if nbatch > 1 {
            panic!("get_index not supported for batched vectors");
        }
        self.context
            .stream
            .clone_dtoh(&self.data.slice(index..index + 1))
            .expect("Failed to copy data from device to host")[0]
    }
    fn set_index(&mut self, index: IndexType, value: Self::T) {
        let nbatch = self.context.nbatch();
        let nstates = self.data.len() as IndexType / nbatch;
        assert!(index < nstates, "Index out of bounds");
        let data = vec![value];
        for b in 0..nbatch {
            let idx = b * nstates + index;
            self.context
                .stream
                .memcpy_htod(&data, &mut self.data.slice_mut(idx..idx + 1))
                .expect("Failed to copy data from host to device");
        }
    }
    fn norm(&self, k: i32) -> Self::T {
        let nbatch = self.context.nbatch();
        let nstates = self.data.len() as IndexType / nbatch;
        if nbatch == 1 {
            return self.context.norm(&self.data, k);
        }
        let mut max_norm = T::zero();
        for b in 0..nbatch {
            let start = b * nstates;
            let slice = self.data.slice(start..start + nstates);
            let norm = self.context.norm(&slice, k);
            if norm > max_norm {
                max_norm = norm;
            }
        }
        max_norm
    }
    fn squared_norm(&self, y: &Self, atol: &Self, rtol: Self::T) -> Self::T {
        let nbatch = self.context.nbatch();
        let nstates = self.data.len() as IndexType / nbatch;
        let atol_nbatch = atol.context.nbatch();
        if nbatch == 1 && atol_nbatch == 1 {
            return self
                .context
                .squared_norm(&self.data, &y.data, &atol.data, rtol);
        }
        let mut max_norm = T::zero();
        for b in 0..nbatch {
            let atol_b = if atol_nbatch == 1 { 0 } else { b };
            let self_start = b * nstates;
            let y_start = b * nstates;
            let atol_start = atol_b * (atol.data.len() as IndexType / atol_nbatch);
            let self_slice = self.data.slice(self_start..self_start + nstates);
            let y_slice = y.data.slice(y_start..y_start + nstates);
            let atol_slice = atol
                .data
                .slice(atol_start..atol_start + (atol.data.len() as IndexType / atol_nbatch));
            let sum = self
                .context
                .squared_norm_all_views(&self_slice, &y_slice, &atol_slice, rtol);
            let nstates_t = T::from_f64(nstates as f64).unwrap();
            let norm = sum / nstates_t;
            if norm > max_norm {
                max_norm = norm;
            }
        }
        max_norm
    }
    fn len(&self) -> IndexType {
        self.data.len() as IndexType / self.context.nbatch()
    }

    fn from_vec(v: Vec<Self::T>, ctx: Self::C) -> Self {
        let nbatch = ctx.nbatch();
        assert!(
            v.len() % nbatch == 0,
            "vector length {} must be divisible by nbatch {}",
            v.len(),
            nbatch
        );
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
        let nbatch = ctx.nbatch();
        assert!(
            slice.len() % nbatch == 0,
            "slice length {} must be divisible by nbatch {}",
            slice.len(),
            nbatch
        );
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
        let nbatch = ctx.nbatch();
        let total = nstates * nbatch;
        let data = unsafe {
            ctx.stream
                .alloc(total)
                .expect("Failed to allocate memory for CudaVec")
        };
        let mut ret = Self { data, context: ctx };
        ret.fill(value);
        ret
    }
    fn zeros(nstates: usize, ctx: Self::C) -> Self {
        let nbatch = ctx.nbatch();
        let total = nstates * nbatch;
        let data = ctx
            .stream
            .alloc_zeros(total)
            .expect("Failed to allocate memory for CudaVec");
        Self { data, context: ctx }
    }

    fn fill(&mut self, value: Self::T) {
        let f = self.context.function::<T>("vec_fill");
        let n = self.data.len() as u32;
        if n == 0 {
            return;
        }
        let mut build = self.context.stream.launch_builder(&f);
        build.arg(&mut self.data).arg(&value).arg(&n);
        let config = self.context.launch_config_1d(n, &f);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }
    fn as_view(&self) -> Self::View<'_> {
        let nstates = self.data.len() as IndexType / self.context.nbatch();
        CudaVecRef {
            data: self.data.as_view(),
            context: self.context.clone(),
            nstates,
            col_offset: 0,
        }
    }
    fn as_view_mut(&mut self) -> Self::ViewMut<'_> {
        let nstates = self.data.len() as IndexType / self.context.nbatch();
        CudaVecMut {
            data: self.data.as_view_mut(),
            context: self.context.clone(),
            nstates,
            col_offset: 0,
        }
    }
    fn copy_from(&mut self, other: &Self) {
        let self_nbatch = self.context.nbatch();
        let other_nbatch = other.context.nbatch();
        self.context
            .assert_compatible_nbatch(other_nbatch, "copy_from");
        if self_nbatch == other_nbatch {
            let f = self.context.function::<T>("vec_copy");
            let n = self.data.len() as u32;
            if n == 0 {
                return;
            }
            let mut build = self.context.stream.launch_builder(&f);
            build.arg(&mut self.data).arg(&other.data).arg(&n);
            let config = self.context.launch_config_1d(n, &f);
            unsafe { build.launch(config) }.expect("Failed to launch kernel");
        } else {
            let nstates = self.data.len() as IndexType / self_nbatch;
            let other_nstates = other.data.len() as IndexType / other_nbatch;
            for b in 0..self_nbatch {
                let other_b = if other_nbatch == 1 { 0 } else { b };
                let self_start = b * nstates;
                let other_start = other_b * other_nstates;
                let f = self.context.function::<T>("vec_copy");
                let n = nstates as u32;
                let config = self.context.launch_config_1d(n, &f);
                let mut build = self.context.stream.launch_builder(&f);
                let mut self_slice = self.data.slice_mut(self_start..self_start + nstates);
                let other_slice = other.data.slice(other_start..other_start + other_nstates);
                build.arg(&mut self_slice).arg(&other_slice).arg(&n);
                unsafe { build.launch(config) }.expect("Failed to launch kernel");
            }
        }
    }
    fn copy_from_view(&mut self, other: &Self::View<'_>) {
        let self_nbatch = self.context.nbatch();
        let other_nbatch = other.context.nbatch();
        self.context
            .assert_compatible_nbatch(other_nbatch, "copy_from_view");
        if self_nbatch == other_nbatch && other.col_offset == 0 && other.stride() == other.nstates {
            let f = self.context.function::<T>("vec_copy");
            let n = self.data.len() as u32;
            if n == 0 {
                return;
            }
            let mut build = self.context.stream.launch_builder(&f);
            build.arg(&mut self.data).arg(&other.data).arg(&n);
            let config = self.context.launch_config_1d(n, &f);
            unsafe { build.launch(config) }.expect("Failed to launch kernel");
        } else {
            let nstates = self.data.len() as IndexType / self_nbatch;
            for b in 0..self_nbatch {
                let other_b = if other_nbatch == 1 { 0 } else { b };
                let self_start = b * nstates;
                let other_stride = other.stride();
                let other_start = other_b * other_stride + other.col_offset;
                let other_nstates = other.nstates;
                let f = self.context.function::<T>("vec_copy");
                let n = nstates as u32;
                let config = self.context.launch_config_1d(n, &f);
                let mut build = self.context.stream.launch_builder(&f);
                let mut self_slice = self.data.slice_mut(self_start..self_start + nstates);
                let other_slice = other.data.slice(other_start..other_start + other_nstates);
                build.arg(&mut self_slice).arg(&other_slice).arg(&n);
                unsafe { build.launch(config) }.expect("Failed to launch kernel");
            }
        }
    }
    fn axpy(&mut self, alpha: Self::T, x: &Self, beta: Self::T) {
        let self_nbatch = self.context.nbatch();
        let x_nbatch = x.context.nbatch();
        self.context
            .assert_compatible_nbatch(x_nbatch, "axpy");
        if beta != T::one() {
            self.mul_assign(Scale(beta));
        }
        if self_nbatch == x_nbatch {
            self.context.axpy::<T, _, _>(alpha, &x.data, &mut self.data);
        } else {
            let nstates = self.data.len() as IndexType / self_nbatch;
            let x_nstates = x.data.len() as IndexType / x_nbatch;
            for b in 0..self_nbatch {
                let x_b = if x_nbatch == 1 { 0 } else { b };
                let self_start = b * nstates;
                let x_start = x_b * x_nstates;
                let x_slice = x.data.slice(x_start..x_start + x_nstates);
                let (x_ptr, _) = x_slice.device_ptr(&self.context.stream);
                let mut self_slice = self.data.slice_mut(self_start..self_start + nstates);
                let (y_ptr, _) = self_slice.device_ptr_mut(&self.context.stream);
                self.context
                    .axpy_inner::<T>(alpha, x_ptr, y_ptr, nstates as c_int);
            }
        }
    }
    fn axpy_v(&mut self, alpha: Self::T, x: &Self::View<'_>, beta: Self::T) {
        let self_nbatch = self.context.nbatch();
        let x_nbatch = x.context.nbatch();
        self.context
            .assert_compatible_nbatch(x_nbatch, "axpy_v");
        if beta != T::one() {
            self.mul_assign(Scale(beta));
        }
        let nstates = self.data.len() as IndexType / self_nbatch;
        for b in 0..self_nbatch {
            let x_b = if x_nbatch == 1 { 0 } else { b };
            let self_start = b * nstates;
            let x_stride = x.stride();
            let x_start = x_b * x_stride + x.col_offset;
            let x_nstates = x.nstates;
            let x_slice = x.data.slice(x_start..x_start + x_nstates);
            let (x_ptr, _) = x_slice.device_ptr(&self.context.stream);
            let mut self_slice = self.data.slice_mut(self_start..self_start + nstates);
            let (y_ptr, _) = self_slice.device_ptr_mut(&self.context.stream);
            self.context
                .axpy_inner::<T>(alpha, x_ptr, y_ptr, nstates as c_int);
        }
    }
    fn clone_as_vec(&self) -> Vec<Self::T> {
        self.context
            .stream
            .clone_dtoh(&self.data)
            .expect("Failed to copy data from device to host")
    }
    fn component_mul_assign(&mut self, other: &Self) {
        let self_nbatch = self.context.nbatch();
        let other_nbatch = other.context.nbatch();
        self.context
            .assert_compatible_nbatch(other_nbatch, "component_mul_assign");
        if self_nbatch == other_nbatch {
            let f = self.context.function::<T>("vec_mul_assign");
            let n = self.data.len() as u32;
            if n == 0 {
                return;
            }
            let mut build = self.context.stream.launch_builder(&f);
            build.arg(&mut self.data).arg(&other.data).arg(&n);
            let config = self.context.launch_config_1d(n, &f);
            unsafe { build.launch(config) }.expect("Failed to launch kernel");
        } else {
            let nstates = self.data.len() as IndexType / self_nbatch;
            let other_nstates = other.data.len() as IndexType / other_nbatch;
            for b in 0..self_nbatch {
                let other_b = if other_nbatch == 1 { 0 } else { b };
                let self_start = b * nstates;
                let other_start = other_b * other_nstates;
                let f = self.context.function::<T>("vec_mul_assign");
                let n = nstates as u32;
                let config = self.context.launch_config_1d(n, &f);
                let mut build = self.context.stream.launch_builder(&f);
                let mut self_slice = self.data.slice_mut(self_start..self_start + nstates);
                let other_slice = other.data.slice(other_start..other_start + other_nstates);
                build.arg(&mut self_slice).arg(&other_slice).arg(&n);
                unsafe { build.launch(config) }.expect("Failed to launch kernel");
            }
        }
    }
    fn component_div_assign(&mut self, other: &Self) {
        let self_nbatch = self.context.nbatch();
        let other_nbatch = other.context.nbatch();
        self.context
            .assert_compatible_nbatch(other_nbatch, "component_div_assign");
        if self_nbatch == other_nbatch {
            let f = self.context.function::<T>("vec_div_assign");
            let n = self.data.len() as u32;
            if n == 0 {
                return;
            }
            let mut build = self.context.stream.launch_builder(&f);
            build.arg(&mut self.data).arg(&other.data).arg(&n);
            let config = self.context.launch_config_1d(n, &f);
            unsafe { build.launch(config) }.expect("Failed to launch kernel");
        } else {
            let nstates = self.data.len() as IndexType / self_nbatch;
            let other_nstates = other.data.len() as IndexType / other_nbatch;
            for b in 0..self_nbatch {
                let other_b = if other_nbatch == 1 { 0 } else { b };
                let self_start = b * nstates;
                let other_start = other_b * other_nstates;
                let f = self.context.function::<T>("vec_div_assign");
                let n = nstates as u32;
                let config = self.context.launch_config_1d(n, &f);
                let mut build = self.context.stream.launch_builder(&f);
                let mut self_slice = self.data.slice_mut(self_start..self_start + nstates);
                let other_slice = other.data.slice(other_start..other_start + other_nstates);
                build.arg(&mut self_slice).arg(&other_slice).arg(&n);
                unsafe { build.launch(config) }.expect("Failed to launch kernel");
            }
        }
    }
    fn root_finding(&self, g1: &Self) -> (bool, Self::T, i32) {
        let nbatch = self.context.nbatch();
        let nstates = self.data.len() as IndexType / nbatch;
        let g1_nbatch = g1.context.nbatch();
        let g1_nstates = g1.data.len() as IndexType / g1_nbatch;
        assert_eq!(
            nstates, g1_nstates,
            "Vector length mismatch: {} != {}",
            nstates, g1_nstates
        );
        let f = self.context.function::<T>("vec_root_finding");

        let mut first_result: Option<(bool, T, i32)> = None;
        for b in 0..nbatch {
            let g1_b = if g1_nbatch == 1 { 0 } else { b };
            let self_start = b * nstates;
            let g1_start = g1_b * g1_nstates;
            let self_slice = self.data.slice(self_start..self_start + nstates);
            let g1_slice = g1.data.slice(g1_start..g1_start + g1_nstates);

            let n = nstates as u32;
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
                .arg(&self_slice)
                .arg(&g1_slice)
                .arg(&n)
                .arg(&mut root_flag)
                .arg(&mut max_vals)
                .arg(&mut max_idxs);
            unsafe { build.launch(config) }.expect("Failed to launch kernel");

            let h_max_vals = self
                .context
                .stream
                .clone_dtoh(&max_vals)
                .expect("Failed to copy data from device to host");
            let h_max_idxs = self
                .context
                .stream
                .clone_dtoh(&max_idxs)
                .expect("Failed to copy data from device to host");
            let h_root_flag = self
                .context
                .stream
                .clone_dtoh(&root_flag)
                .expect("Failed to copy data from device to host");
            let found_root = h_root_flag[0] != 0;
            let mut max_val = T::zero();
            let mut max_idx = -1;
            for (max_val_i, max_idx_i) in h_max_vals.iter().zip(h_max_idxs.iter()) {
                if *max_val_i > max_val {
                    max_val = *max_val_i;
                    max_idx = *max_idx_i;
                }
            }
            let result = (found_root, max_val, max_idx);
            if let Some(ref first) = first_result {
                if first.0 != result.0 || first.2 != result.2 {
                    panic!(
                        "Root finding results differ across batches: batch 0 = {:?}, batch {} = {:?}",
                        first, b, result
                    );
                }
            } else {
                first_result = Some(result);
            }
        }
        first_result.unwrap()
    }
    fn assign_at_indices(&mut self, indices: &Self::Index, value: Self::T) {
        let nbatch = self.context.nbatch();
        let nstates = self.data.len() as IndexType / nbatch;
        for b in 0..nbatch {
            let batch_offset = b * nstates;
            let f = self.context.function::<T>("vec_assign_at_indices");
            let n = indices.len() as u32;
            if n == 0 { continue; }
            let config = self.context.launch_config_1d(n, &f);
            let mut build = self.context.stream.launch_builder(&f);
            let mut self_slice = self.data.slice_mut(batch_offset..batch_offset + nstates);
            build
                .arg(&mut self_slice)
                .arg(&indices.data)
                .arg(&value)
                .arg(&n);
            unsafe { build.launch(config) }.expect("Failed to launch kernel");
        }
    }
    fn copy_from_indices(&mut self, other: &Self, indices: &Self::Index) {
        let nbatch = self.context.nbatch();
        let nstates = self.data.len() as IndexType / nbatch;
        for b in 0..nbatch {
            let batch_offset = b * nstates;
            let f = self.context.function::<T>("vec_copy_from_indices");
            let n = indices.len() as u32;
            if n == 0 { continue; }
            let config = self.context.launch_config_1d(n, &f);
            let mut build = self.context.stream.launch_builder(&f);
            let mut self_slice = self.data.slice_mut(batch_offset..batch_offset + nstates);
            let other_slice = other.data.slice(batch_offset..batch_offset + nstates);
            build
                .arg(&mut self_slice)
                .arg(&other_slice)
                .arg(&indices.data)
                .arg(&n);
            unsafe { build.launch(config) }.expect("Failed to launch kernel");
        }
    }
    fn gather(&mut self, other: &Self, indices: &Self::Index) {
        let nbatch = self.context.nbatch();
        let nstates = self.data.len() as IndexType / nbatch;
        for b in 0..nbatch {
            let batch_offset = b * nstates;
            let f = self.context.function::<T>("vec_gather");
            let n = indices.len() as u32;
            if n == 0 { continue; }
            let config = self.context.launch_config_1d(n, &f);
            let mut build = self.context.stream.launch_builder(&f);
            let mut self_slice = self.data.slice_mut(batch_offset..batch_offset + nstates);
            let other_slice = other.data.slice(batch_offset..batch_offset + nstates);
            build
                .arg(&mut self_slice)
                .arg(&other_slice)
                .arg(&indices.data)
                .arg(&n);
            unsafe { build.launch(config) }.expect("Failed to launch kernel");
        }
    }
    fn scatter(&self, indices: &Self::Index, other: &mut Self) {
        let nbatch = self.context.nbatch();
        let nstates = self.data.len() as IndexType / nbatch;
        let other_nbatch = other.context.nbatch();
        let other_nstates = other.data.len() as IndexType / other_nbatch;
        for b in 0..nbatch {
            let other_b = if other_nbatch == 1 { 0 } else { b };
            let self_offset = b * nstates;
            let other_offset = other_b * other_nstates;
            let f = self.context.function::<T>("vec_scatter");
            let n = indices.len() as u32;
            if n == 0 { continue; }
            let config = self.context.launch_config_1d(n, &f);
            let mut build = self.context.stream.launch_builder(&f);
            let self_slice = self.data.slice(self_offset..self_offset + nstates);
            let mut other_slice = other.data.slice_mut(other_offset..other_offset + other_nstates);
            build
                .arg(&self_slice)
                .arg(&indices.data)
                .arg(&mut other_slice)
                .arg(&n);
            unsafe { build.launch(config) }.expect("Failed to launch kernel");
        }
    }
    fn get_batch(&self, batch: usize) -> Self::View<'_> {
        let nbatch = self.context.nbatch();
        let nstates = self.data.len() as IndexType / nbatch;
        assert!(batch < nbatch, "Batch index out of bounds");
        let start = batch * nstates;
        CudaVecRef {
            data: self.data.slice(start..start + nstates),
            context: self.context.clone_with_nbatch(1),
            nstates,
            col_offset: 0,
        }
    }
    fn get_batch_mut(&mut self, batch: usize) -> Self::ViewMut<'_> {
        let nbatch = self.context.nbatch();
        let nstates = self.data.len() as IndexType / nbatch;
        assert!(batch < nbatch, "Batch index out of bounds");
        let start = batch * nstates;
        CudaVecMut {
            data: self.data.slice_mut(start..start + nstates),
            context: self.context.clone_with_nbatch(1),
            nstates,
            col_offset: 0,
        }
    }
}

impl<T: ScalarCuda> VectorView<'_> for CudaVecRef<'_, T> {
    type Owned = CudaVec<T>;
    fn get_index(&self, index: IndexType) -> Self::T {
        let nbatch = self.context.nbatch();
        if nbatch > 1 {
            panic!("get_index not supported for batched views");
        }
        let offset = self.col_offset + index;
        self.context
            .stream
            .clone_dtoh(&self.data.slice(offset..offset + 1))
            .expect("Failed to copy data from device to host")[0]
    }
    fn into_owned(self) -> Self::Owned {
        let nbatch = self.context.nbatch();
        let stride = self.stride();
        let total_valid = self.nstates * nbatch;
        if stride == self.nstates && self.col_offset == 0 {
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
        } else {
            let mut ret = unsafe {
                self.context
                    .stream
                    .alloc(total_valid)
                    .expect("Failed to allocate memory for CudaVec")
            };
            for b in 0..nbatch {
                let src_start = b * stride + self.col_offset;
                let dst_start = b * self.nstates;
                let src_slice = self.data.slice(src_start..src_start + self.nstates);
                let mut dst_slice =
                    ret.slice_mut(dst_start..dst_start + self.nstates);
                self.context
                    .stream
                    .memcpy_dtod(&src_slice, &mut dst_slice)
                    .expect("Failed to copy data from device to device");
            }
            Self::Owned {
                data: ret,
                context: self.context,
            }
        }
    }
    fn squared_norm(&self, y: &Self::Owned, atol: &Self::Owned, rtol: Self::T) -> Self::T {
        let nbatch = self.context.nbatch();
        let stride = self.stride();
        let atol_nbatch = atol.context.nbatch();
        if nbatch == 1 && atol_nbatch == 1 && stride == self.nstates {
            return self
                .context
                .squared_norm_view(&self.data, &y.data, &atol.data, rtol);
        }
        let mut max_norm = T::zero();
        let y_nstates = y.data.len() as IndexType / y.context.nbatch();
        let atol_nstates = atol.data.len() as IndexType / atol_nbatch;
        for b in 0..nbatch {
            let atol_b = if atol_nbatch == 1 { 0 } else { b };
            let self_start = b * stride + self.col_offset;
            let y_start = b * y_nstates;
            let atol_start = atol_b * atol_nstates;
            let self_slice = self.data.slice(self_start..self_start + self.nstates);
            let y_slice = y.data.slice(y_start..y_start + y_nstates);
            let atol_slice = atol.data.slice(atol_start..atol_start + atol_nstates);
            let sum = self
                .context
                .squared_norm_all_views(&self_slice, &y_slice, &atol_slice, rtol);
            let nstates_t = T::from_f64(self.nstates as f64).unwrap();
            let norm = sum / nstates_t;
            if norm > max_norm {
                max_norm = norm;
            }
        }
        max_norm
    }
}

impl<'a, T: ScalarCuda> VectorViewMut<'a> for CudaVecMut<'a, T> {
    type Owned = CudaVec<T>;
    type View = CudaVecRef<'a, T>;
    type Index = CudaIndex;
    fn copy_from(&mut self, other: &Self::Owned) {
        let nbatch = self.context.nbatch();
        let other_nbatch = other.context.nbatch();
        self.context
            .assert_compatible_nbatch(other_nbatch, "copy_from");
        if nbatch == other_nbatch && self.col_offset == 0 && self.stride() == self.nstates {
            let f = self.context.function::<T>("vec_copy");
            let n = self.data.len() as u32;
            if n == 0 {
                return;
            }
            let mut build = self.context.stream.launch_builder(&f);
            build.arg(&mut self.data).arg(&other.data).arg(&n);
            let config = self.context.launch_config_1d(n, &f);
            unsafe { build.launch(config) }.expect("Failed to launch kernel");
        } else {
            let other_nstates = other.data.len() as IndexType / other_nbatch;
            for b in 0..nbatch {
                let other_b = if other_nbatch == 1 { 0 } else { b };
                let self_start = b * self.stride() + self.col_offset;
                let other_start = other_b * other_nstates;
                let f = self.context.function::<T>("vec_copy");
                let n = self.nstates as u32;
                let config = self.context.launch_config_1d(n, &f);
                let mut build = self.context.stream.launch_builder(&f);
                let mut self_slice = self.data.slice_mut(self_start..self_start + self.nstates);
                let other_slice = other.data.slice(other_start..other_start + other_nstates);
                build.arg(&mut self_slice).arg(&other_slice).arg(&n);
                unsafe { build.launch(config) }.expect("Failed to launch kernel");
            }
        }
    }
    fn copy_from_view(&mut self, other: &Self::View) {
        let nbatch = self.context.nbatch();
        let other_nbatch = other.context.nbatch();
        self.context
            .assert_compatible_nbatch(other_nbatch, "copy_from_view");
        if nbatch == other_nbatch
            && self.col_offset == 0
            && self.stride() == self.nstates
            && other.col_offset == 0
            && other.stride() == other.nstates
        {
            let f = self.context.function::<T>("vec_copy");
            let n = self.data.len() as u32;
            if n == 0 {
                return;
            }
            let mut build = self.context.stream.launch_builder(&f);
            build.arg(&mut self.data).arg(&other.data).arg(&n);
            let config = self.context.launch_config_1d(n, &f);
            unsafe { build.launch(config) }.expect("Failed to launch kernel");
        } else {
            for b in 0..nbatch {
                let other_b = if other_nbatch == 1 { 0 } else { b };
                let self_start = b * self.stride() + self.col_offset;
                let other_stride = other.stride();
                let other_start = other_b * other_stride + other.col_offset;
                let f = self.context.function::<T>("vec_copy");
                let n = self.nstates as u32;
                let config = self.context.launch_config_1d(n, &f);
                let mut build = self.context.stream.launch_builder(&f);
                let mut self_slice = self.data.slice_mut(self_start..self_start + self.nstates);
                let other_slice = other.data.slice(other_start..other_start + other.nstates);
                build.arg(&mut self_slice).arg(&other_slice).arg(&n);
                unsafe { build.launch(config) }.expect("Failed to launch kernel");
            }
        }
    }
    fn set_index(&mut self, index: IndexType, value: Self::T) {
        let nbatch = self.context.nbatch();
        assert!(index < self.nstates, "Index out of bounds");
        let data = vec![value];
        for b in 0..nbatch {
            let idx = b * self.stride() + self.col_offset + index;
            self.context
                .stream
                .memcpy_htod(&data, &mut self.data.slice_mut(idx..idx + 1))
                .expect("Failed to copy data from host to device");
        }
    }
    fn axpy(&mut self, alpha: Self::T, x: &Self::Owned, beta: Self::T) {
        let nbatch = self.context.nbatch();
        let x_nbatch = x.context.nbatch();
        self.context
            .assert_compatible_nbatch(x_nbatch, "axpy");
        if beta != T::one() {
            self.mul_assign(Scale(beta));
        }
        let x_nstates = x.data.len() as IndexType / x_nbatch;
        let stride = self.stride();
        for b in 0..nbatch {
            let x_b = if x_nbatch == 1 { 0 } else { b };
            let self_start = b * stride + self.col_offset;
            let x_start = x_b * x_nstates;
            let x_slice = x.data.slice(x_start..x_start + x_nstates);
            let (x_ptr, _) = x_slice.device_ptr(&self.context.stream);
            let mut self_slice = self.data.slice_mut(self_start..self_start + self.nstates);
            let (y_ptr, _) = self_slice.device_ptr_mut(&self.context.stream);
            self.context
                .axpy_inner::<T>(alpha, x_ptr, y_ptr, self.nstates as c_int);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    super::super::generate_vector_tests!(
        cuda,
        CudaVec<f64>,
        CudaContext::default().with_nbatch(2),
        CudaContext::default().with_nbatch(3)
    );
}
