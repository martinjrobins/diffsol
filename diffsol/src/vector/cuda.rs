use std::ffi::c_int;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Sub, SubAssign};

use super::{VectorIndex, VectorView, VectorViewMut};
use cudarc::cublas::sys as cublas;
use cudarc::cublas::CudaBlas;
use cudarc::driver::{
    CudaFunction, CudaSlice, CudaView, CudaViewMut, DevicePtr, LaunchConfig,
    PushKernelArg,
};

use crate::{
    Context, CudaContext, CudaMat, CudaType, DefaultDenseMatrix, IndexType, ScalarCuda, Scale,
    Vector, VectorCommon,
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
    pub(crate) fn launch_config_2d(&self, nstates: u32, nbatch: u32, f: &CudaFunction) -> LaunchConfig {
        let (_min_grid_size, block_size) = f
            .occupancy_max_potential_block_size(zero, 0, 0, None)
            .expect("Failed to get occupancy max potential block size");
        let grid_x = nstates.div_ceil(block_size);
        LaunchConfig {
            grid_dim: (grid_x, nbatch, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    pub(crate) fn launch_config_2d_reduce(
        &self,
        nstates: u32,
        nbatch: u32,
        f: &CudaFunction,
        smem_size_f: extern "C" fn(block_size: std::ffi::c_int) -> usize,
    ) -> LaunchConfig {
        let (_min_grid_size, block_size) = f
            .occupancy_max_potential_block_size(smem_size_f, 0, 0, None)
            .expect("Failed to get occupancy max potential block size");
        let highest_bit_set_idx = 31 - (block_size | 1).leading_zeros();
        let block_size = (1 << highest_bit_set_idx) & block_size;
        let grid_x = nstates.div_ceil(block_size);
        LaunchConfig {
            grid_dim: (grid_x, nbatch, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: smem_size_f(block_size as i32) as u32,
        }
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
}

impl<'a, T: ScalarCuda> CudaVecMut<'a, T> {
    pub(crate) fn stride(&self) -> IndexType {
        self.data.len() as IndexType / self.context.nbatch()
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
                let nbatch = self.context.nbatch();
                let nstates = (self.data.len() / nbatch) as u32;
                if nstates == 0 {
                    return self;
                }
                let nbatch_u32 = nbatch as u32;
                let stride = nstates as i32;
                let scalar = rhs.value();
                let mut build = self.context.stream.launch_builder(&f);
                build.arg(&mut self.data).arg(&scalar).arg(&nstates).arg(&nbatch_u32).arg(&stride);
                let config = self.context.launch_config_2d(nstates, nbatch_u32, &f);
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
                let nbatch = self.context.nbatch();
                let nstates = self.data.len() / nbatch;
                let mut ret = Self::Output::zeros(nstates, self.context.clone());
                let f = self.context.function::<T>("vec_mul_scalar");
                let nstates_u32 = nstates as u32;
                if nstates_u32 == 0 {
                    return ret;
                }
                let nbatch_u32 = nbatch as u32;
                let src_stride = nstates as i32;
                let src_nbatch = nbatch as i32;
                let ret_stride = nstates as i32;
                let mut build = self.context.stream.launch_builder(&f);
                let scalar = rhs.value();
                build
                    .arg(&self.data)
                    .arg(&scalar)
                    .arg(&mut ret.data)
                    .arg(&nstates_u32)
                    .arg(&ret_stride)
                    .arg(&src_stride).arg(&src_nbatch);
                let config = self.context.launch_config_2d(nstates_u32, nbatch_u32, &f);
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
                let nbatch = self.context.nbatch();
                let nstates = (self.data.len() / nbatch) as u32;
                if nstates == 0 {
                    return;
                }
                let nbatch_u32 = nbatch as u32;
                let stride = nstates as i32;
                let mut build = self.context.stream.launch_builder(&f);
                let scalar = rhs.value();
                build.arg(&mut self.data).arg(&scalar).arg(&nstates).arg(&nbatch_u32).arg(&stride);
                let config = self.context.launch_config_2d(nstates, nbatch_u32, &f);
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
        let nstates_u32 = self.nstates as u32;
        if nstates_u32 == 0 {
            return ret;
        }
        let config = self.context.launch_config_2d(nstates_u32, nbatch as u32, &f);
        let mut build = self.context.stream.launch_builder(&f);
        let src_data = self.data.slice(self.col_offset..);
        let src_stride = self.stride() as i32;
        let ret_stride = self.nstates as i32;
        let nbatch_i32 = nbatch as i32;
        build
            .arg(&src_data)
            .arg(&scalar)
            .arg(&mut ret.data)
            .arg(&nstates_u32)
            .arg(&ret_stride)
            .arg(&src_stride)
            .arg(&nbatch_i32);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
        ret
    }
}

// Stride-aware MulAssign<Scale> for CudaVecMut
impl<T: ScalarCuda> MulAssign<Scale<T>> for CudaVecMut<'_, T> {
    fn mul_assign(&mut self, rhs: Scale<T>) {
        let nbatch = self.context.nbatch();
        let f = self.context.function::<T>("vec_mul_assign_scalar");
        let scalar = rhs.value();
        let n = self.nstates as u32;
        if n == 0 {
            return;
        }
        let nbatch_u32 = nbatch as u32;
        let config = self.context.launch_config_2d(n, nbatch_u32, &f);
        let mut build = self.context.stream.launch_builder(&f);
        let stride_i32 = self.stride() as i32;
        let col_offset = self.col_offset;
        let mut self_data = self.data.slice_mut(col_offset..);
        build
            .arg(&mut self_data)
            .arg(&scalar)
            .arg(&n)
            .arg(&nbatch_u32)
            .arg(&stride_i32);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
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
        let nstates_u32 = self.nstates as u32;
        if nstates_u32 == 0 {
            return ret;
        }
        let config = self.context.launch_config_2d(nstates_u32, nbatch as u32, &f);
        let mut build = self.context.stream.launch_builder(&f);
        let self_data = self.data.slice(self.col_offset..);
        let ret_stride = self.nstates as i32;
        let nbatch_i32 = nbatch as i32;
        let stride_i32 = self.stride() as i32;
        build
            .arg(&self_data)
            .arg(&scalar)
            .arg(&mut ret.data)
            .arg(&nstates_u32)
            .arg(&ret_stride)
            .arg(&stride_i32)
            .arg(&nbatch_i32);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
        ret
    }
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
        self.context
            .assert_compatible_nbatch(other_nbatch, "sub_assign");
        let nstates = (self.data.len() as IndexType / self_nbatch) as u32;
        let other_nstates = rhs.data.len() as IndexType / other_nbatch;
        if nstates == 0 {
            return;
        }
        let f = self.context.function::<T>("vec_sub_assign");
        let nbatch = self_nbatch as u32;
        let self_stride = nstates as i32;
        let rhs_stride = other_nstates as i32;
        let other_nbatch_i32 = other_nbatch as i32;
        let config = self.context.launch_config_2d(nstates, nbatch, &f);
        let mut build = self.context.stream.launch_builder(&f);
        build
            .arg(&mut self.data)
            .arg(&rhs.data)
            .arg(&nstates)
            .arg(&self_stride)
            .arg(&rhs_stride)
            .arg(&other_nbatch_i32);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
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
        self.context
            .assert_compatible_nbatch(other_nbatch, "sub_assign");
        let nstates = (self.data.len() as IndexType / self_nbatch) as u32;
        let _rhs_nstates = rhs.nstates;
        if nstates == 0 {
            return;
        }
        let f = self.context.function::<T>("vec_sub_assign");
        let nbatch = self_nbatch as u32;
        let self_stride = nstates as i32;
        let rhs_stride = rhs.stride() as i32;
        let rhs_data = rhs.data.slice(rhs.col_offset..);
        let other_nbatch_i32 = other_nbatch as i32;
        let config = self.context.launch_config_2d(nstates, nbatch, &f);
        let mut build = self.context.stream.launch_builder(&f);
        build
            .arg(&mut self.data)
            .arg(&rhs_data)
            .arg(&nstates)
            .arg(&self_stride)
            .arg(&rhs_stride)
            .arg(&other_nbatch_i32);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
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
        self.context
            .assert_compatible_nbatch(other_nbatch, "add_assign");
        let nstates = (self.data.len() as IndexType / self_nbatch) as u32;
        let other_nstates = rhs.data.len() as IndexType / other_nbatch;
        if nstates == 0 {
            return;
        }
        let f = self.context.function::<T>("vec_add_assign");
        let nbatch = self_nbatch as u32;
        let self_stride = nstates as i32;
        let rhs_stride = other_nstates as i32;
        let other_nbatch_i32 = other_nbatch as i32;
        let config = self.context.launch_config_2d(nstates, nbatch, &f);
        let mut build = self.context.stream.launch_builder(&f);
        build
            .arg(&mut self.data)
            .arg(&rhs.data)
            .arg(&nstates)
            .arg(&self_stride)
            .arg(&rhs_stride)
            .arg(&other_nbatch_i32);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
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
        self.context
            .assert_compatible_nbatch(other_nbatch, "add_assign");
        let nstates = (self.data.len() as IndexType / self_nbatch) as u32;
        if nstates == 0 {
            return;
        }
        let f = self.context.function::<T>("vec_add_assign");
        let nbatch = self_nbatch as u32;
        let self_stride = nstates as i32;
        let rhs_stride = rhs.stride() as i32;
        let other_nbatch_i32 = other_nbatch as i32;
        let rhs_data = rhs.data.slice(rhs.col_offset..);
        let config = self.context.launch_config_2d(nstates, nbatch, &f);
        let mut build = self.context.stream.launch_builder(&f);
        build
            .arg(&mut self.data)
            .arg(&rhs_data)
            .arg(&nstates)
            .arg(&self_stride)
            .arg(&rhs_stride)
            .arg(&other_nbatch_i32);
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
        self.context
            .assert_compatible_nbatch(other_nbatch, "sub_assign");
        let other_nstates = rhs.data.len() as IndexType / other_nbatch;
        let nstates_u32 = self.nstates as u32;
        if nstates_u32 == 0 {
            return;
        }
        let f = self.context.function::<T>("vec_sub_assign");
        let nbatch_u32 = nbatch as u32;
        let other_nbatch_i32 = other_nbatch as i32;
        let config = self.context.launch_config_2d(nstates_u32, nbatch_u32, &f);
        let mut build = self.context.stream.launch_builder(&f);
        let self_stride = self.stride() as i32;
        let col_offset = self.col_offset;
        let mut self_data = self.data.slice_mut(col_offset..);
        let rhs_stride = other_nstates as i32;
        build
            .arg(&mut self_data)
            .arg(&rhs.data)
            .arg(&nstates_u32)
            .arg(&self_stride)
            .arg(&rhs_stride)
            .arg(&other_nbatch_i32);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
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
        self.context
            .assert_compatible_nbatch(other_nbatch, "add_assign");
        let other_nstates = rhs.data.len() as IndexType / other_nbatch;
        let nstates_u32 = self.nstates as u32;
        if nstates_u32 == 0 {
            return;
        }
        let f = self.context.function::<T>("vec_add_assign");
        let nbatch_u32 = nbatch as u32;
        let other_nbatch_i32 = other_nbatch as i32;
        let config = self.context.launch_config_2d(nstates_u32, nbatch_u32, &f);
        let mut build = self.context.stream.launch_builder(&f);
        let self_stride = self.stride() as i32;
        let col_offset = self.col_offset;
        let mut self_data = self.data.slice_mut(col_offset..);
        let rhs_stride = other_nstates as i32;
        build
            .arg(&mut self_data)
            .arg(&rhs.data)
            .arg(&nstates_u32)
            .arg(&self_stride)
            .arg(&rhs_stride)
            .arg(&other_nbatch_i32);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }
}
impl<T: ScalarCuda> SubAssign<CudaVecRef<'_, T>> for CudaVecMut<'_, T> {
    fn sub_assign(&mut self, rhs: CudaVecRef<'_, T>) {
        self.sub_assign(&rhs);
    }
}
impl<T: ScalarCuda> SubAssign<&CudaVecRef<'_, T>> for CudaVecMut<'_, T> {
    fn sub_assign(&mut self, rhs: &CudaVecRef<'_, T>) {
        let nbatch = self.context.nbatch();
        let other_nbatch = rhs.context.nbatch();
        self.context
            .assert_compatible_nbatch(other_nbatch, "sub_assign");
        let nstates_u32 = self.nstates as u32;
        if nstates_u32 == 0 {
            return;
        }
        let f = self.context.function::<T>("vec_sub_assign");
        let nbatch_u32 = nbatch as u32;
        let other_nbatch_i32 = other_nbatch as i32;
        let config = self.context.launch_config_2d(nstates_u32, nbatch_u32, &f);
        let mut build = self.context.stream.launch_builder(&f);
        let self_stride = self.stride() as i32;
        let col_offset = self.col_offset;
        let mut self_data = self.data.slice_mut(col_offset..);
        let rhs_data = rhs.data.slice(rhs.col_offset..);
        let rhs_stride = rhs.stride() as i32;
        build
            .arg(&mut self_data)
            .arg(&rhs_data)
            .arg(&nstates_u32)
            .arg(&self_stride)
            .arg(&rhs_stride)
            .arg(&other_nbatch_i32);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
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
        self.context
            .assert_compatible_nbatch(other_nbatch, "add_assign");
        let nstates_u32 = self.nstates as u32;
        if nstates_u32 == 0 {
            return;
        }
        let f = self.context.function::<T>("vec_add_assign");
        let nbatch_u32 = nbatch as u32;
        let other_nbatch_i32 = other_nbatch as i32;
        let config = self.context.launch_config_2d(nstates_u32, nbatch_u32, &f);
        let mut build = self.context.stream.launch_builder(&f);
        let self_stride = self.stride() as i32;
        let col_offset = self.col_offset;
        let mut self_data = self.data.slice_mut(col_offset..);
        let rhs_data = rhs.data.slice(rhs.col_offset..);
        let rhs_stride = rhs.stride() as i32;
        build
            .arg(&mut self_data)
            .arg(&rhs_data)
            .arg(&nstates_u32)
            .arg(&self_stride)
            .arg(&rhs_stride)
            .arg(&other_nbatch_i32);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }
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
        let nstates_usize = self.data.len() as IndexType / self_nbatch;
        let nstates = nstates_usize as u32;
        let other_nstates = rhs.data.len() as IndexType / other_nbatch;
        let mut ret = CudaVec::zeros(nstates_usize, self.context.clone());
        if nstates == 0 {
            return ret;
        }
        let nbatch = self_nbatch as u32;
        let f = self.context.function::<T>("vec_sub");
        let config = self.context.launch_config_2d(nstates, nbatch, &f);
        let mut build = self.context.stream.launch_builder(&f);
        let self_stride = nstates as i32;
        let rhs_stride = other_nstates as i32;
        let ret_stride = nstates as i32;
        let other_nbatch_i32 = other_nbatch as i32;
        let self_nbatch_i32 = self_nbatch as i32;
        build
            .arg(&self.data)
            .arg(&rhs.data)
            .arg(&mut ret.data)
            .arg(&nstates)
            .arg(&self_stride)
            .arg(&rhs_stride)
            .arg(&other_nbatch_i32)
            .arg(&ret_stride)
            .arg(&self_nbatch_i32);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
        ret
    }
}

// &CudaVec - CudaVecRef -> CudaVec
impl<T: ScalarCuda> Sub<CudaVecRef<'_, T>> for &CudaVec<T> {
    type Output = CudaVec<T>;
    fn sub(self, rhs: CudaVecRef<'_, T>) -> CudaVec<T> {
        self.sub(&rhs)
    }
}
impl<T: ScalarCuda> Sub<&CudaVecRef<'_, T>> for &CudaVec<T> {
    type Output = CudaVec<T>;
    fn sub(self, rhs: &CudaVecRef<'_, T>) -> CudaVec<T> {
        let self_nbatch = self.context.nbatch();
        let other_nbatch = rhs.context.nbatch();
        self.context.assert_compatible_nbatch(other_nbatch, "sub");
        let nstates_usize = self.data.len() as IndexType / self_nbatch;
        let nstates = nstates_usize as u32;
        if nstates == 0 {
            return CudaVec::zeros(nstates_usize, self.context.clone());
        }
        let mut ret = CudaVec::zeros(nstates_usize, self.context.clone());
        let nbatch = self_nbatch as u32;
        let f = self.context.function::<T>("vec_sub");
        let config = self.context.launch_config_2d(nstates, nbatch, &f);
        let mut build = self.context.stream.launch_builder(&f);
        let self_stride = nstates as i32;
        let rhs_stride = rhs.stride() as i32;
        let rhs_data = rhs.data.slice(rhs.col_offset..);
        let ret_stride = nstates as i32;
        let other_nbatch_i32 = other_nbatch as i32;
        let self_nbatch_i32 = self_nbatch as i32;
        build
            .arg(&self.data)
            .arg(&rhs_data)
            .arg(&mut ret.data)
            .arg(&nstates)
            .arg(&self_stride)
            .arg(&rhs_stride)
            .arg(&other_nbatch_i32)
            .arg(&ret_stride)
            .arg(&self_nbatch_i32);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
        ret
    }
}

// CudaVec - CudaVec -> CudaVec (owned by value)
impl<T: ScalarCuda> Sub<CudaVec<T>> for CudaVec<T> {
    type Output = CudaVec<T>;
    fn sub(mut self, rhs: CudaVec<T>) -> CudaVec<T> {
        self.sub_assign(&rhs);
        self
    }
}

// CudaVec + CudaVec -> CudaVec (owned by value)
impl<T: ScalarCuda> Add<CudaVec<T>> for CudaVec<T> {
    type Output = CudaVec<T>;
    fn add(mut self, rhs: CudaVec<T>) -> CudaVec<T> {
        self.add_assign(&rhs);
        self
    }
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
    fn sub(mut self, rhs: CudaVecRef<'_, T>) -> CudaVec<T> {
        self.sub_assign(&rhs);
        self
    }
}
impl<T: ScalarCuda> Sub<&CudaVecRef<'_, T>> for CudaVec<T> {
    type Output = CudaVec<T>;
    fn sub(mut self, rhs: &CudaVecRef<'_, T>) -> CudaVec<T> {
        self.sub_assign(rhs);
        self
    }
}

// CudaVecRef - CudaVec -> CudaVec
impl<T: ScalarCuda> Sub<CudaVec<T>> for CudaVecRef<'_, T> {
    type Output = CudaVec<T>;
    fn sub(self, rhs: CudaVec<T>) -> CudaVec<T> {
        &self - &rhs
    }
}

// CudaVecRef - &CudaVec -> CudaVec
impl<T: ScalarCuda> Sub<&CudaVec<T>> for CudaVecRef<'_, T> {
    type Output = CudaVec<T>;
    fn sub(self, rhs: &CudaVec<T>) -> CudaVec<T> {
        sub_view_minus_owned(&self, rhs)
    }
}
impl<T: ScalarCuda> Sub<&CudaVec<T>> for &CudaVecRef<'_, T> {
    type Output = CudaVec<T>;
    fn sub(self, rhs: &CudaVec<T>) -> CudaVec<T> {
        sub_view_minus_owned(self, rhs)
    }
}

// CudaVecRef - CudaVecRef -> CudaVec
impl<T: ScalarCuda> Sub<CudaVecRef<'_, T>> for CudaVecRef<'_, T> {
    type Output = CudaVec<T>;
    fn sub(self, rhs: CudaVecRef<'_, T>) -> CudaVec<T> {
        &self - &rhs
    }
}
impl<T: ScalarCuda> Sub<&CudaVecRef<'_, T>> for CudaVecRef<'_, T> {
    type Output = CudaVec<T>;
    fn sub(self, rhs: &CudaVecRef<'_, T>) -> CudaVec<T> {
        sub_view_minus_view(&self, rhs)
    }
}
impl<T: ScalarCuda> Sub<&CudaVecRef<'_, T>> for &CudaVecRef<'_, T> {
    type Output = CudaVec<T>;
    fn sub(self, rhs: &CudaVecRef<'_, T>) -> CudaVec<T> {
        sub_view_minus_view(self, rhs)
    }
}

fn sub_view_minus_owned<T: ScalarCuda>(lhs: &CudaVecRef<'_, T>, rhs: &CudaVec<T>) -> CudaVec<T> {
    let self_nbatch = lhs.context.nbatch();
    let other_nbatch = rhs.context.nbatch();
    lhs.context.assert_compatible_nbatch(other_nbatch, "sub");
    let nstates = lhs.nstates;
    let nstates_u32 = nstates as u32;
    let other_nstates = rhs.data.len() as IndexType / other_nbatch;
    let mut ret = CudaVec::zeros(nstates, lhs.context.clone());
    if nstates_u32 == 0 {
        return ret;
    }
    let nbatch = self_nbatch as u32;
    let f = lhs.context.function::<T>("vec_sub");
    let config = lhs.context.launch_config_2d(nstates_u32, nbatch, &f);
    let mut build = lhs.context.stream.launch_builder(&f);
    let self_data = lhs.data.slice(lhs.col_offset..);
    let self_stride = lhs.stride() as i32;
    let rhs_stride = other_nstates as i32;
    let ret_stride = nstates as i32;
    let other_nbatch_i32 = other_nbatch as i32;
    let self_nbatch_i32 = self_nbatch as i32;
    build
        .arg(&self_data)
        .arg(&rhs.data)
        .arg(&mut ret.data)
        .arg(&nstates_u32)
        .arg(&self_stride)
        .arg(&rhs_stride)
        .arg(&other_nbatch_i32)
        .arg(&ret_stride)
        .arg(&self_nbatch_i32);
    unsafe { build.launch(config) }.expect("Failed to launch kernel");
    ret
}

fn sub_view_minus_view<T: ScalarCuda>(
    lhs: &CudaVecRef<'_, T>,
    rhs: &CudaVecRef<'_, T>,
) -> CudaVec<T> {
    let self_nbatch = lhs.context.nbatch();
    let other_nbatch = rhs.context.nbatch();
    lhs.context.assert_compatible_nbatch(other_nbatch, "sub");
    let nstates = lhs.nstates;
    let nstates_u32 = nstates as u32;
    let mut ret = CudaVec::zeros(nstates, lhs.context.clone());
    if nstates_u32 == 0 {
        return ret;
    }
    let nbatch = self_nbatch as u32;
    let f = lhs.context.function::<T>("vec_sub");
    let config = lhs.context.launch_config_2d(nstates_u32, nbatch, &f);
    let mut build = lhs.context.stream.launch_builder(&f);
    let self_data = lhs.data.slice(lhs.col_offset..);
    let rhs_data = rhs.data.slice(rhs.col_offset..);
    let self_stride = lhs.stride() as i32;
    let rhs_stride = rhs.stride() as i32;
    let ret_stride = nstates as i32;
    let other_nbatch_i32 = other_nbatch as i32;
    let self_nbatch_i32 = self_nbatch as i32;
    build
        .arg(&self_data)
        .arg(&rhs_data)
        .arg(&mut ret.data)
        .arg(&nstates_u32)
        .arg(&self_stride)
        .arg(&rhs_stride)
        .arg(&other_nbatch_i32)
        .arg(&ret_stride)
        .arg(&self_nbatch_i32);
    unsafe { build.launch(config) }.expect("Failed to launch kernel");
    ret
}

// ============================================================
// Add operations: a + b -> CudaVec (new owned)
// ============================================================

// &CudaVec + CudaVec -> CudaVec
impl<T: ScalarCuda> Add<CudaVec<T>> for &CudaVec<T> {
    type Output = CudaVec<T>;
    fn add(self, rhs: CudaVec<T>) -> CudaVec<T> {
        self.add(&rhs)
    }
}
// &CudaVec - CudaVec -> CudaVec
impl<T: ScalarCuda> Sub<CudaVec<T>> for &CudaVec<T> {
    type Output = CudaVec<T>;
    fn sub(self, rhs: CudaVec<T>) -> CudaVec<T> {
        self.sub(&rhs)
    }
}

// &CudaVec + &CudaVec -> CudaVec
impl<T: ScalarCuda> Add<&CudaVec<T>> for &CudaVec<T> {
    type Output = CudaVec<T>;
    fn add(self, rhs: &CudaVec<T>) -> CudaVec<T> {
        let self_nbatch = self.context.nbatch();
        let other_nbatch = rhs.context.nbatch();
        self.context.assert_compatible_nbatch(other_nbatch, "add");
        let nstates_usize = self.data.len() as IndexType / self_nbatch;
        let nstates = nstates_usize as u32;
        let other_nstates = rhs.data.len() as IndexType / other_nbatch;
        let mut ret = CudaVec::zeros(nstates_usize, self.context.clone());
        if nstates == 0 {
            return ret;
        }
        let nbatch = self_nbatch as u32;
        let f = self.context.function::<T>("vec_add");
        let config = self.context.launch_config_2d(nstates, nbatch, &f);
        let mut build = self.context.stream.launch_builder(&f);
        let self_stride = nstates as i32;
        let rhs_stride = other_nstates as i32;
        let ret_stride = nstates as i32;
        let other_nbatch_i32 = other_nbatch as i32;
        let self_nbatch_i32 = self_nbatch as i32;
        build
            .arg(&self.data)
            .arg(&rhs.data)
            .arg(&mut ret.data)
            .arg(&nstates)
            .arg(&self_stride)
            .arg(&rhs_stride)
            .arg(&other_nbatch_i32)
            .arg(&ret_stride)
            .arg(&self_nbatch_i32);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
        ret
    }
}

// &CudaVec + CudaVecRef -> CudaVec
impl<T: ScalarCuda> Add<CudaVecRef<'_, T>> for &CudaVec<T> {
    type Output = CudaVec<T>;
    fn add(self, rhs: CudaVecRef<'_, T>) -> CudaVec<T> {
        self.add(&rhs)
    }
}
impl<T: ScalarCuda> Add<&CudaVecRef<'_, T>> for &CudaVec<T> {
    type Output = CudaVec<T>;
    fn add(self, rhs: &CudaVecRef<'_, T>) -> CudaVec<T> {
        let self_nbatch = self.context.nbatch();
        let other_nbatch = rhs.context.nbatch();
        self.context.assert_compatible_nbatch(other_nbatch, "add");
        let nstates_usize = self.data.len() as IndexType / self_nbatch;
        let nstates = nstates_usize as u32;
        if nstates == 0 {
            return CudaVec::zeros(nstates_usize, self.context.clone());
        }
        let mut ret = CudaVec::zeros(nstates_usize, self.context.clone());
        let nbatch = self_nbatch as u32;
        let f = self.context.function::<T>("vec_add");
        let config = self.context.launch_config_2d(nstates, nbatch, &f);
        let mut build = self.context.stream.launch_builder(&f);
        let self_stride = nstates as i32;
        let rhs_stride = rhs.stride() as i32;
        let rhs_data = rhs.data.slice(rhs.col_offset..);
        let ret_stride = nstates as i32;
        let other_nbatch_i32 = other_nbatch as i32;
        let self_nbatch_i32 = self_nbatch as i32;
        build
            .arg(&self.data)
            .arg(&rhs_data)
            .arg(&mut ret.data)
            .arg(&nstates)
            .arg(&self_stride)
            .arg(&rhs_stride)
            .arg(&other_nbatch_i32)
            .arg(&ret_stride)
            .arg(&self_nbatch_i32);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
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
    fn add(mut self, rhs: CudaVecRef<'_, T>) -> CudaVec<T> {
        self.add_assign(&rhs);
        self
    }
}
impl<T: ScalarCuda> Add<&CudaVecRef<'_, T>> for CudaVec<T> {
    type Output = CudaVec<T>;
    fn add(mut self, rhs: &CudaVecRef<'_, T>) -> CudaVec<T> {
        self.add_assign(rhs);
        self
    }
}

// CudaVecRef + CudaVec -> CudaVec
impl<T: ScalarCuda> Add<CudaVec<T>> for CudaVecRef<'_, T> {
    type Output = CudaVec<T>;
    fn add(self, rhs: CudaVec<T>) -> CudaVec<T> {
        &self + &rhs
    }
}

// CudaVecRef + &CudaVec -> CudaVec
impl<T: ScalarCuda> Add<&CudaVec<T>> for CudaVecRef<'_, T> {
    type Output = CudaVec<T>;
    fn add(self, rhs: &CudaVec<T>) -> CudaVec<T> {
        add_view_plus_owned(&self, rhs)
    }
}
impl<T: ScalarCuda> Add<&CudaVec<T>> for &CudaVecRef<'_, T> {
    type Output = CudaVec<T>;
    fn add(self, rhs: &CudaVec<T>) -> CudaVec<T> {
        add_view_plus_owned(self, rhs)
    }
}

// CudaVecRef + CudaVecRef -> CudaVec
impl<T: ScalarCuda> Add<CudaVecRef<'_, T>> for CudaVecRef<'_, T> {
    type Output = CudaVec<T>;
    fn add(self, rhs: CudaVecRef<'_, T>) -> CudaVec<T> {
        &self + &rhs
    }
}
impl<T: ScalarCuda> Add<&CudaVecRef<'_, T>> for CudaVecRef<'_, T> {
    type Output = CudaVec<T>;
    fn add(self, rhs: &CudaVecRef<'_, T>) -> CudaVec<T> {
        add_view_plus_view(&self, rhs)
    }
}
impl<T: ScalarCuda> Add<&CudaVecRef<'_, T>> for &CudaVecRef<'_, T> {
    type Output = CudaVec<T>;
    fn add(self, rhs: &CudaVecRef<'_, T>) -> CudaVec<T> {
        add_view_plus_view(self, rhs)
    }
}

fn add_view_plus_owned<T: ScalarCuda>(lhs: &CudaVecRef<'_, T>, rhs: &CudaVec<T>) -> CudaVec<T> {
    let self_nbatch = lhs.context.nbatch();
    let other_nbatch = rhs.context.nbatch();
    lhs.context.assert_compatible_nbatch(other_nbatch, "add");
    let nstates = lhs.nstates;
    let nstates_u32 = nstates as u32;
    let other_nstates = rhs.data.len() as IndexType / other_nbatch;
    let mut ret = CudaVec::zeros(nstates, lhs.context.clone());
    if nstates_u32 == 0 {
        return ret;
    }
    let nbatch = self_nbatch as u32;
    let f = lhs.context.function::<T>("vec_add");
    let config = lhs.context.launch_config_2d(nstates_u32, nbatch, &f);
    let mut build = lhs.context.stream.launch_builder(&f);
    let self_data = lhs.data.slice(lhs.col_offset..);
    let self_stride = lhs.stride() as i32;
    let rhs_stride = other_nstates as i32;
    let ret_stride = nstates as i32;
    let other_nbatch_i32 = other_nbatch as i32;
    let self_nbatch_i32 = self_nbatch as i32;
    build
        .arg(&self_data)
        .arg(&rhs.data)
        .arg(&mut ret.data)
        .arg(&nstates_u32)
        .arg(&self_stride)
        .arg(&rhs_stride)
        .arg(&other_nbatch_i32)
        .arg(&ret_stride)
        .arg(&self_nbatch_i32);
    unsafe { build.launch(config) }.expect("Failed to launch kernel");
    ret
}

fn add_view_plus_view<T: ScalarCuda>(
    lhs: &CudaVecRef<'_, T>,
    rhs: &CudaVecRef<'_, T>,
) -> CudaVec<T> {
    let self_nbatch = lhs.context.nbatch();
    let other_nbatch = rhs.context.nbatch();
    lhs.context.assert_compatible_nbatch(other_nbatch, "add");
    let nstates = lhs.nstates;
    let nstates_u32 = nstates as u32;
    let mut ret = CudaVec::zeros(nstates, lhs.context.clone());
    if nstates_u32 == 0 {
        return ret;
    }
    let nbatch = self_nbatch as u32;
    let f = lhs.context.function::<T>("vec_add");
    let config = lhs.context.launch_config_2d(nstates_u32, nbatch, &f);
    let mut build = lhs.context.stream.launch_builder(&f);
    let self_data = lhs.data.slice(lhs.col_offset..);
    let rhs_data = rhs.data.slice(rhs.col_offset..);
    let self_stride = lhs.stride() as i32;
    let rhs_stride = rhs.stride() as i32;
    let ret_stride = nstates as i32;
    let other_nbatch_i32 = other_nbatch as i32;
    let self_nbatch_i32 = self_nbatch as i32;
    build
        .arg(&self_data)
        .arg(&rhs_data)
        .arg(&mut ret.data)
        .arg(&nstates_u32)
        .arg(&self_stride)
        .arg(&rhs_stride)
        .arg(&other_nbatch_i32)
        .arg(&ret_stride)
        .arg(&self_nbatch_i32);
    unsafe { build.launch(config) }.expect("Failed to launch kernel");
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
        let y_nbatch = y.context.nbatch();

        let nstates_u32 = nstates as u32;
        let nbatch_u32 = nbatch as u32;

        let f = self.context.function::<T>("vec_squared_norm");
        let config = self.context.launch_config_2d_reduce(nstates_u32, nbatch_u32, &f, squared_norm_blk_size::<T>);
        let blocks_per_batch = config.grid_dim.0 as usize;
        let total_blocks = blocks_per_batch * nbatch;
        let mut partial_sums = unsafe {
            self.context.stream.alloc::<T>(total_blocks)
                .expect("Failed to allocate memory for partial sums")
        };
        let mut build = self.context.stream.launch_builder(&f);

        let y_stride = (self.data.len() as IndexType / nbatch) as i32;
        let y_nbatch_i32 = nbatch as i32;
        let y0_stride = (y.data.len() as IndexType / y_nbatch) as i32;
        let y0_nbatch_i32 = y_nbatch as i32;
        let atol_stride = (atol.data.len() as IndexType / atol_nbatch) as i32;
        let atol_nbatch_i32 = atol_nbatch as i32;
        let rtol_val = rtol;

        build.arg(&self.data).arg(&y.data).arg(&atol.data)
             .arg(&rtol_val)
             .arg(&nstates_u32).arg(&nbatch_u32)
             .arg(&y_stride).arg(&y_nbatch_i32)
             .arg(&y0_stride).arg(&y0_nbatch_i32)
             .arg(&atol_stride).arg(&atol_nbatch_i32)
             .arg(&mut partial_sums);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
        let partial_sums = self.context.stream.clone_dtoh(&partial_sums)
            .expect("Failed to copy data from device to host");
        let nstates_t = T::from_f64(nstates as f64).unwrap();
        let mut max_norm = T::zero();
        for b in 0..nbatch {
            let start = b * blocks_per_batch;
            let sum = partial_sums[start..start + blocks_per_batch].iter().fold(T::zero(), |acc, x| acc + *x);
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
        let nbatch = self.context.nbatch();
        let nstates = self.data.len() as IndexType / nbatch;
        let nstates_u32 = nstates as u32;
        let nbatch_u32 = nbatch as u32;
        if nstates_u32 == 0 {
            return;
        }
        let f = self.context.function::<T>("vec_fill");
        let config = self.context.launch_config_2d(nstates_u32, nbatch_u32, &f);
        let mut build = self.context.stream.launch_builder(&f);
        let nstates_i32 = nstates as i32;
        build
            .arg(&mut self.data)
            .arg(&value)
            .arg(&nstates_u32)
            .arg(&nbatch_u32)
            .arg(&nstates_i32);
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
        let nstates = (self.data.len() as IndexType / self_nbatch) as u32;
        let other_nstates = other.data.len() as IndexType / other_nbatch;
        if nstates == 0 {
            return;
        }
        let nbatch = self_nbatch as u32;
        let f = self.context.function::<T>("vec_copy");
        let config = self.context.launch_config_2d(nstates, nbatch, &f);
        let mut build = self.context.stream.launch_builder(&f);
        let self_stride = nstates as i32;
        let rhs_stride = other_nstates as i32;
        let other_nbatch_i32 = other_nbatch as i32;
        build
            .arg(&mut self.data)
            .arg(&other.data)
            .arg(&nstates)
            .arg(&self_stride)
            .arg(&rhs_stride)
            .arg(&other_nbatch_i32);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }
    fn copy_from_view(&mut self, other: &Self::View<'_>) {
        let self_nbatch = self.context.nbatch();
        let other_nbatch = other.context.nbatch();
        self.context
            .assert_compatible_nbatch(other_nbatch, "copy_from_view");
        let nstates = (self.data.len() as IndexType / self_nbatch) as u32;
        if nstates == 0 {
            return;
        }
        let nbatch = self_nbatch as u32;
        let f = self.context.function::<T>("vec_copy");
        let config = self.context.launch_config_2d(nstates, nbatch, &f);
        let mut build = self.context.stream.launch_builder(&f);
        let self_stride = nstates as i32;
        let rhs_stride = other.stride() as i32;
        let rhs_data = other.data.slice(other.col_offset..);
        let other_nbatch_i32 = other_nbatch as i32;
        build
            .arg(&mut self.data)
            .arg(&rhs_data)
            .arg(&nstates)
            .arg(&self_stride)
            .arg(&rhs_stride)
            .arg(&other_nbatch_i32);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }
    fn axpy(&mut self, alpha: Self::T, x: &Self, beta: Self::T) {
        let self_nbatch = self.context.nbatch();
        let x_nbatch = x.context.nbatch();
        self.context.assert_compatible_nbatch(x_nbatch, "axpy");
        let nstates = self.data.len() as IndexType / self_nbatch;
        let x_nstates = x.data.len() as IndexType / x_nbatch;
        let nstates_u32 = nstates as u32;
        let nbatch_u32 = self_nbatch as u32;
        let f = self.context.function::<T>("vec_axpy");
        let config = self.context.launch_config_2d(nstates_u32, nbatch_u32, &f);
        let mut build = self.context.stream.launch_builder(&f);
        let self_stride = nstates as i32;
        let x_stride = x_nstates as i32;
        let x_nbatch_i32 = x_nbatch as i32;
        let alpha_val = alpha;
        let beta_val = beta;
        build
            .arg(&mut self.data)
            .arg(&x.data)
            .arg(&alpha_val)
            .arg(&beta_val)
            .arg(&nstates_u32)
            .arg(&self_stride)
            .arg(&x_stride)
            .arg(&x_nbatch_i32);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }
    fn axpy_v(&mut self, alpha: Self::T, x: &Self::View<'_>, beta: Self::T) {
        let self_nbatch = self.context.nbatch();
        let x_nbatch = x.context.nbatch();
        self.context.assert_compatible_nbatch(x_nbatch, "axpy_v");
        let nstates = self.data.len() as IndexType / self_nbatch;
        let nstates_u32 = nstates as u32;
        let nbatch_u32 = self_nbatch as u32;
        let f = self.context.function::<T>("vec_axpy");
        let config = self.context.launch_config_2d(nstates_u32, nbatch_u32, &f);
        let mut build = self.context.stream.launch_builder(&f);
        let self_stride = nstates as i32;
        let x_stride = x.stride() as i32;
        let x_nbatch_i32 = x_nbatch as i32;
        let x_data = x.data.slice(x.col_offset..);
        let alpha_val = alpha;
        let beta_val = beta;
        build
            .arg(&mut self.data)
            .arg(&x_data)
            .arg(&alpha_val)
            .arg(&beta_val)
            .arg(&nstates_u32)
            .arg(&self_stride)
            .arg(&x_stride)
            .arg(&x_nbatch_i32);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
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
        let nstates = (self.data.len() as IndexType / self_nbatch) as u32;
        let other_nstates = other.data.len() as IndexType / other_nbatch;
        if nstates == 0 {
            return;
        }
        let nbatch = self_nbatch as u32;
        let f = self.context.function::<T>("vec_mul_assign");
        let config = self.context.launch_config_2d(nstates, nbatch, &f);
        let mut build = self.context.stream.launch_builder(&f);
        let self_stride = nstates as i32;
        let rhs_stride = other_nstates as i32;
        let other_nbatch_i32 = other_nbatch as i32;
        build
            .arg(&mut self.data)
            .arg(&other.data)
            .arg(&nstates)
            .arg(&self_stride)
            .arg(&rhs_stride)
            .arg(&other_nbatch_i32);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }
    fn component_div_assign(&mut self, other: &Self) {
        let self_nbatch = self.context.nbatch();
        let other_nbatch = other.context.nbatch();
        self.context
            .assert_compatible_nbatch(other_nbatch, "component_div_assign");
        let nstates = (self.data.len() as IndexType / self_nbatch) as u32;
        let other_nstates = other.data.len() as IndexType / other_nbatch;
        if nstates == 0 {
            return;
        }
        let nbatch = self_nbatch as u32;
        let f = self.context.function::<T>("vec_div_assign");
        let config = self.context.launch_config_2d(nstates, nbatch, &f);
        let mut build = self.context.stream.launch_builder(&f);
        let self_stride = nstates as i32;
        let rhs_stride = other_nstates as i32;
        let other_nbatch_i32 = other_nbatch as i32;
        build
            .arg(&mut self.data)
            .arg(&other.data)
            .arg(&nstates)
            .arg(&self_stride)
            .arg(&rhs_stride)
            .arg(&other_nbatch_i32);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
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

        let nstates_u32 = nstates as u32;
        let nbatch_u32 = nbatch as u32;

        let f = self.context.function::<T>("vec_root_finding");
        let config = self.context.launch_config_2d_reduce(nstates_u32, nbatch_u32, &f, root_finding_blk_size::<T>);
        let blocks_per_batch = config.grid_dim.0 as usize;
        let total_blocks = blocks_per_batch * nbatch;

        let mut max_vals = unsafe {
            self.context.stream.alloc::<T>(total_blocks)
                .expect("Failed to allocate memory for max_vals")
        };
        let mut max_idxs = unsafe {
            self.context.stream.alloc::<c_int>(total_blocks)
                .expect("Failed to allocate memory for max_idxs")
        };
        let mut root_flag = self.context.stream.alloc_zeros::<c_int>(nbatch)
            .expect("Failed to allocate memory for root_flag");

        let mut build = self.context.stream.launch_builder(&f);

        let g0_stride = nstates as i32;
        let g0_nbatch_i32 = nbatch as i32;
        let g1_stride = g1_nstates as i32;
        let g1_nbatch_i32 = g1_nbatch as i32;

        build.arg(&self.data).arg(&g1.data)
             .arg(&nstates_u32).arg(&nbatch_u32)
             .arg(&g0_stride).arg(&g0_nbatch_i32)
             .arg(&g1_stride).arg(&g1_nbatch_i32)
             .arg(&mut root_flag)
             .arg(&mut max_vals)
             .arg(&mut max_idxs);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");

        let h_max_vals = self.context.stream.clone_dtoh(&max_vals)
            .expect("Failed to copy data from device to host");
        let h_max_idxs = self.context.stream.clone_dtoh(&max_idxs)
            .expect("Failed to copy data from device to host");
        let h_root_flag = self.context.stream.clone_dtoh(&root_flag)
            .expect("Failed to copy data from device to host");

        let mut first_result: Option<(bool, T, i32)> = None;
        for b in 0..nbatch {
            let found_root = h_root_flag[b] != 0;
            let start = b * blocks_per_batch;
            let mut max_val = T::zero();
            let mut max_idx = -1;
            for i in start..start + blocks_per_batch {
                if h_max_vals[i] > max_val {
                    max_val = h_max_vals[i];
                    max_idx = h_max_idxs[i];
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
        let self_nbatch = self.context.nbatch();
        let nindices_u32 = indices.len() as u32;
        if nindices_u32 == 0 {
            return;
        }
        let nstates = self.data.len() as IndexType / self_nbatch;
        let nbatch_u32 = self_nbatch as u32;
        let f = self.context.function::<T>("vec_assign_at_indices");
        let config = self.context.launch_config_2d(nindices_u32, nbatch_u32, &f);
        let mut build = self.context.stream.launch_builder(&f);
        let self_stride = nstates as i32;
        let self_nbatch_i32 = self_nbatch as i32;
        build
            .arg(&mut self.data)
            .arg(&indices.data)
            .arg(&value)
            .arg(&nindices_u32)
            .arg(&self_stride)
            .arg(&self_nbatch_i32);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }
    fn copy_from_indices(&mut self, other: &Self, indices: &Self::Index) {
        let self_nbatch = self.context.nbatch();
        let other_nbatch = other.context.nbatch();
        let nindices_u32 = indices.len() as u32;
        if nindices_u32 == 0 {
            return;
        }
        let nstates = self.data.len() as IndexType / self_nbatch;
        let nbatch_u32 = self_nbatch as u32;
        let f = self.context.function::<T>("vec_copy_from_indices");
        let config = self.context.launch_config_2d(nindices_u32, nbatch_u32, &f);
        let mut build = self.context.stream.launch_builder(&f);
        let self_stride = nstates as i32;
        let self_nbatch_i32 = self_nbatch as i32;
        let other_nstates = other.data.len() as IndexType / other_nbatch;
        let other_stride = other_nstates as i32;
        let other_nbatch_i32 = other_nbatch as i32;
        build
            .arg(&mut self.data)
            .arg(&other.data)
            .arg(&indices.data)
            .arg(&nindices_u32)
            .arg(&self_stride)
            .arg(&self_nbatch_i32)
            .arg(&other_stride)
            .arg(&other_nbatch_i32);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }
    fn gather(&mut self, other: &Self, indices: &Self::Index) {
        let self_nbatch = self.context.nbatch();
        let other_nbatch = other.context.nbatch();
        let nindices_u32 = indices.len() as u32;
        if nindices_u32 == 0 {
            return;
        }
        let nstates = self.data.len() as IndexType / self_nbatch;
        let nbatch_u32 = self_nbatch as u32;
        let f = self.context.function::<T>("vec_gather");
        let config = self.context.launch_config_2d(nindices_u32, nbatch_u32, &f);
        let mut build = self.context.stream.launch_builder(&f);
        let self_stride = nstates as i32;
        let self_nbatch_i32 = self_nbatch as i32;
        let other_nstates = other.data.len() as IndexType / other_nbatch;
        let other_stride = other_nstates as i32;
        let other_nbatch_i32 = other_nbatch as i32;
        build
            .arg(&mut self.data)
            .arg(&other.data)
            .arg(&indices.data)
            .arg(&nindices_u32)
            .arg(&self_stride)
            .arg(&self_nbatch_i32)
            .arg(&other_stride)
            .arg(&other_nbatch_i32);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }
    fn scatter(&self, indices: &Self::Index, other: &mut Self) {
        let self_nbatch = self.context.nbatch();
        let other_nbatch = other.context.nbatch();
        let nindices_u32 = indices.len() as u32;
        if nindices_u32 == 0 {
            return;
        }
        let nstates = self.data.len() as IndexType / self_nbatch;
        let nbatch_u32 = self_nbatch as u32;
        let f = self.context.function::<T>("vec_scatter");
        let config = self.context.launch_config_2d(nindices_u32, nbatch_u32, &f);
        let mut build = self.context.stream.launch_builder(&f);
        let self_stride = nstates as i32;
        let self_nbatch_i32 = self_nbatch as i32;
        let other_nstates = other.data.len() as IndexType / other_nbatch;
        let other_stride = other_nstates as i32;
        let other_nbatch_i32 = other_nbatch as i32;
        build
            .arg(&self.data)
            .arg(&indices.data)
            .arg(&mut other.data)
            .arg(&nindices_u32)
            .arg(&self_stride)
            .arg(&self_nbatch_i32)
            .arg(&other_stride)
            .arg(&other_nbatch_i32);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
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
                let mut dst_slice = ret.slice_mut(dst_start..dst_start + self.nstates);
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
        let nstates = self.nstates;
        let atol_nbatch = atol.context.nbatch();
        let y_nbatch = y.context.nbatch();

        let nstates_u32 = nstates as u32;
        let nbatch_u32 = nbatch as u32;

        let f = self.context.function::<T>("vec_squared_norm");
        let config = self.context.launch_config_2d_reduce(nstates_u32, nbatch_u32, &f, squared_norm_blk_size::<T>);
        let blocks_per_batch = config.grid_dim.0 as usize;
        let total_blocks = blocks_per_batch * nbatch;
        let mut partial_sums = unsafe {
            self.context.stream.alloc::<T>(total_blocks)
                .expect("Failed to allocate memory for partial sums")
        };
        let mut build = self.context.stream.launch_builder(&f);

        let self_data = self.data.slice(self.col_offset..);
        let y_stride = self.stride() as i32;
        let y_nbatch_i32 = nbatch as i32;
        let y0_stride = (y.data.len() as IndexType / y_nbatch) as i32;
        let y0_nbatch_i32 = y_nbatch as i32;
        let atol_stride = (atol.data.len() as IndexType / atol_nbatch) as i32;
        let atol_nbatch_i32 = atol_nbatch as i32;
        let rtol_val = rtol;

        build.arg(&self_data).arg(&y.data).arg(&atol.data)
             .arg(&rtol_val)
             .arg(&nstates_u32).arg(&nbatch_u32)
             .arg(&y_stride).arg(&y_nbatch_i32)
             .arg(&y0_stride).arg(&y0_nbatch_i32)
             .arg(&atol_stride).arg(&atol_nbatch_i32)
             .arg(&mut partial_sums);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
        let partial_sums = self.context.stream.clone_dtoh(&partial_sums)
            .expect("Failed to copy data from device to host");
        let nstates_t = T::from_f64(nstates as f64).unwrap();
        let mut max_norm = T::zero();
        for b in 0..nbatch {
            let start = b * blocks_per_batch;
            let sum = partial_sums[start..start + blocks_per_batch].iter().fold(T::zero(), |acc, x| acc + *x);
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
        let nstates_u32 = self.nstates as u32;
        let other_nstates = other.data.len() as IndexType / other_nbatch;
        if nstates_u32 == 0 {
            return;
        }
        let nbatch_u32 = nbatch as u32;
        let other_nbatch_i32 = other_nbatch as i32;
        let f = self.context.function::<T>("vec_copy");
        let config = self.context.launch_config_2d(nstates_u32, nbatch_u32, &f);
        let mut build = self.context.stream.launch_builder(&f);
        let self_stride = self.stride() as i32;
        let col_offset = self.col_offset;
        let mut self_data = self.data.slice_mut(col_offset..);
        let rhs_stride = other_nstates as i32;
        build
            .arg(&mut self_data)
            .arg(&other.data)
            .arg(&nstates_u32)
            .arg(&self_stride)
            .arg(&rhs_stride)
            .arg(&other_nbatch_i32);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }
    fn copy_from_view(&mut self, other: &Self::View) {
        let nbatch = self.context.nbatch();
        let other_nbatch = other.context.nbatch();
        self.context
            .assert_compatible_nbatch(other_nbatch, "copy_from_view");
        let nstates_u32 = self.nstates as u32;
        if nstates_u32 == 0 {
            return;
        }
        let nbatch_u32 = nbatch as u32;
        let other_nbatch_i32 = other_nbatch as i32;
        let f = self.context.function::<T>("vec_copy");
        let config = self.context.launch_config_2d(nstates_u32, nbatch_u32, &f);
        let mut build = self.context.stream.launch_builder(&f);
        let self_stride = self.stride() as i32;
        let col_offset = self.col_offset;
        let mut self_data = self.data.slice_mut(col_offset..);
        let rhs_data = other.data.slice(other.col_offset..);
        let rhs_stride = other.stride() as i32;
        build
            .arg(&mut self_data)
            .arg(&rhs_data)
            .arg(&nstates_u32)
            .arg(&self_stride)
            .arg(&rhs_stride)
            .arg(&other_nbatch_i32);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
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
        self.context.assert_compatible_nbatch(x_nbatch, "axpy");
        let nstates_u32 = self.nstates as u32;
        let nbatch_u32 = nbatch as u32;
        let f = self.context.function::<T>("vec_axpy");
        let config = self.context.launch_config_2d(nstates_u32, nbatch_u32, &f);
        let mut build = self.context.stream.launch_builder(&f);
        let self_stride = self.stride() as i32;
        let col_offset = self.col_offset;
        let x_nstates = x.data.len() as IndexType / x_nbatch;
        let x_stride = x_nstates as i32;
        let x_nbatch_i32 = x_nbatch as i32;
        let mut self_data = self.data.slice_mut(col_offset..);
        let alpha_val = alpha;
        let beta_val = beta;
        build
            .arg(&mut self_data)
            .arg(&x.data)
            .arg(&alpha_val)
            .arg(&beta_val)
            .arg(&nstates_u32)
            .arg(&self_stride)
            .arg(&x_stride)
            .arg(&x_nbatch_i32);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
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
