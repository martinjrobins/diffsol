use cudarc::cublas::sys as cublas;
use cudarc::{
    cublas::{sys::cublasOperation_t, CudaBlas},
    driver::{CudaSlice, CudaView, CudaViewMut, DevicePtr, DevicePtrMut, PushKernelArg},
};
use std::ffi::{c_int, c_longlong};
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use crate::{
    error::{DiffsolError, MatrixError},
    linear_solver::cuda::lu::CudaLU,
    matrix::default_solver::DefaultSolver,
    matrix_error, Context, CudaContext, CudaVec, CudaVecMut, CudaVecRef, IndexType, MatrixCommon,
    ScalarCuda, Scale, Vector, VectorIndex,
};

use super::{
    sparsity::{Dense, DenseRef},
    DenseMatrix, Matrix, MatrixView, MatrixViewMut,
};

/// Dense matrix stored in GPU memory via [`CudaSlice`].
///
/// # Data layout
///
/// Data is stored in **column-major** order as a flat contiguous array of
/// `nrows * ncols * nbatch` elements.  With `nbatch > 1`, all elements of
/// batch 0 appear first, then all elements of batch 1, etc.
///
/// ```text
/// Device memory: [b0(all), b1(all), ..., bN(all)]
/// ```
#[derive(Clone, Debug)]
pub struct CudaMat<T: ScalarCuda> {
    pub(crate) data: CudaSlice<T>,
    pub(crate) context: CudaContext,
    nrows: IndexType,
    ncols: IndexType,
}

/// Immutable reference to a [`CudaMat`], possibly with a strided layout.
///
/// When the view spans a subset of columns, `batch_stride` records the
/// parent matrix's total elements per batch (`nrows * ncols`) so that
/// CUDA kernels can correctly compute per-batch offsets.
#[derive(Debug)]
pub struct CudaMatRef<'a, T: ScalarCuda> {
    pub(crate) data: CudaView<'a, T>,
    pub(crate) context: CudaContext,
    nrows: IndexType,
    ncols: IndexType,
    batch_stride: IndexType,
}

/// Mutable reference to a [`CudaMat`], possibly with a strided layout.
///
/// See [`CudaMatRef`] for the layout description.
#[derive(Debug)]
pub struct CudaMatMut<'a, T: ScalarCuda> {
    pub(crate) data: CudaViewMut<'a, T>,
    pub(crate) context: CudaContext,
    nrows: IndexType,
    ncols: IndexType,
    batch_stride: IndexType,
}

impl CudaContext {
    #[allow(clippy::too_many_arguments)]
    fn gemv<T: ScalarCuda>(
        &self,
        nrows: IndexType,
        ncols: IndexType,
        alpha: T,
        beta: T,
        a: &CudaView<'_, T>,
        x: &CudaView<'_, T>,
        y: &mut CudaViewMut<'_, T>,
    ) {
        let (a, _) = a.device_ptr(&self.stream);
        let (x, _) = x.device_ptr(&self.stream);
        let (y, _) = y.device_ptr_mut(&self.stream);
        let blas = CudaBlas::new(self.stream.clone()).expect("Failed to create CudaBlas");
        let alpha = alpha.as_f64();
        let beta = beta.as_f64();
        unsafe {
            cublas::cublasDgemv_v2(
                *blas.handle(),
                cublasOperation_t::CUBLAS_OP_N,
                nrows as c_int,
                ncols as c_int,
                &alpha as *const f64,
                a as *const f64,
                nrows as c_int,
                x as *const f64,
                1,
                &beta as *const f64,
                y as *mut f64,
                1,
            )
        }
        .result()
        .expect("Failed to launch gemv");
    }
}

impl<T: ScalarCuda> CudaMat<T> {
    pub fn nrows(&self) -> IndexType {
        self.nrows
    }
    pub fn ncols(&self) -> IndexType {
        self.ncols
    }
    fn col_major_index(&self, i: IndexType, j: IndexType) -> IndexType {
        i + j * self.nrows()
    }
    fn get_index(&self, i: IndexType, j: IndexType) -> T {
        let index = self.col_major_index(i, j);
        self.context
            .stream
            .clone_dtoh(&self.data.slice(index..index + 1))
            .expect("Failed to copy data from device to host")[0]
    }
    fn set_index(&mut self, i: IndexType, j: IndexType, value: T) {
        let data = vec![value];
        let index = self.col_major_index(i, j);
        self.context
            .stream
            .memcpy_htod(&data, &mut self.data.slice_mut(index..index + 1))
            .expect("Failed to copy data from host to device");
    }
    fn diagonal(&self) -> CudaVec<T> {
        assert_eq!(
            self.nrows, self.ncols,
            "Matrix must be square to get diagonal"
        );
        let nbatch = self.context.nbatch();
        let n = self.nrows();
        let total = n * nbatch;
        let mut data: CudaSlice<T> = unsafe { self.context.stream.alloc(total) }
            .expect("Failed to allocate memory for diagonal");
        let f = self.context.function::<T>("mat_get_diagonal");
        let n_u32 = n as u32;
        let nbatch_u32 = nbatch as u32;
        let config = self.context.launch_config_2d(n_u32, nbatch_u32, &f);
        let mut build = self.context.stream.launch_builder(&f);
        let mat_stride = (n * n) as i32;
        let diag_stride = n as i32;
        let nbatch_i32 = nbatch as i32;
        build
            .arg(&self.data)
            .arg(&mut data)
            .arg(&n_u32)
            .arg(&mat_stride)
            .arg(&nbatch_i32)
            .arg(&diag_stride)
            .arg(&nbatch_i32);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
        CudaVec {
            data,
            context: self.context.clone(),
        }
    }
}

impl<T: ScalarCuda> DefaultSolver for CudaMat<T> {
    type LS = CudaLU<T>;
}

macro_rules! impl_matrix_common {
    ($mat:ty, $vec:ty, $con:ty, $in:ty) => {
        impl<T: ScalarCuda> MatrixCommon for $mat {
            type T = T;
            type V = $vec;
            type C = $con;
            type Inner = $in;

            fn nrows(&self) -> IndexType {
                self.nrows
            }
            fn ncols(&self) -> IndexType {
                self.ncols
            }
            fn inner(&self) -> &Self::Inner {
                &self.data
            }
        }
    };
}

macro_rules! impl_matrix_common_ref {
    ($mat:ty, $vec:ty, $con:ty, $in:ty) => {
        impl<'a, T: ScalarCuda> MatrixCommon for $mat {
            type T = T;
            type V = $vec;
            type C = $con;
            type Inner = $in;

            fn nrows(&self) -> IndexType {
                self.nrows
            }
            fn ncols(&self) -> IndexType {
                self.ncols
            }
            fn inner(&self) -> &Self::Inner {
                &self.data
            }
        }
    };
}

impl_matrix_common_ref!(
    CudaMatMut<'a, T>,
    CudaVec<T>,
    CudaContext,
    CudaViewMut<'a, T>
);
impl_matrix_common_ref!(CudaMatRef<'a, T>, CudaVec<T>, CudaContext, CudaView<'a, T>);
impl_matrix_common!(CudaMat<T>, CudaVec<T>, CudaContext, CudaSlice<T>);

macro_rules! impl_mul_scalar {
    ($mat_type:ty, $out:ty) => {
        impl<'a, T: ScalarCuda> Mul<Scale<T>> for $mat_type {
            type Output = $out;

            fn mul(mut self, rhs: Scale<T>) -> Self::Output {
                let f = self.context.function::<T>("vec_mul_assign_scalar");
                let nbatch = self.context.nbatch();
                let nstates = (self.data.len() / nbatch) as u32;
                let nbatch_u32 = nbatch as u32;
                let stride = nstates as i32;
                let scalar = rhs.value();
                let mut build = self.context.stream.launch_builder(&f);
                build
                    .arg(&mut self.data)
                    .arg(&scalar)
                    .arg(&nstates)
                    .arg(&nbatch_u32)
                    .arg(&stride);
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
                let mut ret = Self::Output::zeros(self.nrows(), self.ncols(), self.context.clone());
                let f = self.context.function::<T>("vec_mul_scalar");
                let nbatch = self.context.nbatch();
                let nstates = (self.data.len() / nbatch) as u32;
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
                    .arg(&nstates)
                    .arg(&ret_stride)
                    .arg(&src_stride)
                    .arg(&src_nbatch);
                let config = self.context.launch_config_2d(nstates, nbatch_u32, &f);
                unsafe { build.launch(config) }.expect("Failed to launch kernel");
                ret
            }
        }
    };
}

impl_mul_scalar_alloc!(CudaMatRef<'_, T>, CudaMat<T>, T);
impl_mul_scalar!(CudaMat<T>, CudaMat<T>);
impl_mul_scalar_alloc!(&CudaMat<T>, CudaMat<T>, T);

macro_rules! impl_mul_assign_scalar {
    ($col_type:ty, $scalar:ty) => {
        impl<'a, T: ScalarCuda> MulAssign<Scale<T>> for $col_type {
            fn mul_assign(&mut self, rhs: Scale<T>) {
                let f = self.context.function::<T>("vec_mul_assign_scalar");
                let nbatch = self.context.nbatch();
                let nstates = (self.data.len() / nbatch) as u32;
                let nbatch_u32 = nbatch as u32;
                let stride = nstates as i32;
                let mut build = self.context.stream.launch_builder(&f);
                let scalar = rhs.value();
                build
                    .arg(&mut self.data)
                    .arg(&scalar)
                    .arg(&nstates)
                    .arg(&nbatch_u32)
                    .arg(&stride);
                let config = self.context.launch_config_2d(nstates, nbatch_u32, &f);
                unsafe { build.launch(config) }.expect("Failed to launch kernel");
            }
        }
    };
}

impl_mul_assign_scalar!(CudaMatMut<'_, T>, T);

macro_rules! impl_sub_assign {
    ($lhs:ty, $rhs:ty) => {
        impl<T: ScalarCuda> SubAssign<$rhs> for $lhs {
            fn sub_assign(&mut self, rhs: $rhs) {
                let f = self.context.function::<T>("vec_sub_assign");
                let nbatch = self.context.nbatch();
                let nstates = (self.data.len() / nbatch) as u32;
                let nbatch_u32 = nbatch as u32;
                let self_stride = nstates as i32;
                let rhs_nbatch = rhs.context.nbatch() as i32;
                let rhs_nstates = (rhs.data.len() / rhs_nbatch as usize) as u32;
                let rhs_stride = rhs_nstates as i32;
                let mut build = self.context.stream.launch_builder(&f);
                build
                    .arg(&mut self.data)
                    .arg(&rhs.data)
                    .arg(&nstates)
                    .arg(&self_stride)
                    .arg(&rhs_stride)
                    .arg(&rhs_nbatch);
                let config = self.context.launch_config_2d(nstates, nbatch_u32, &f);
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
                let nbatch = self.context.nbatch();
                let nstates = (self.data.len() / nbatch) as u32;
                let nbatch_u32 = nbatch as u32;
                let self_stride = nstates as i32;
                let rhs_nbatch = rhs.context.nbatch() as i32;
                let rhs_nstates = (rhs.data.len() / rhs_nbatch as usize) as u32;
                let rhs_stride = rhs_nstates as i32;
                let mut build = self.context.stream.launch_builder(&f);
                build
                    .arg(&mut self.data)
                    .arg(&rhs.data)
                    .arg(&nstates)
                    .arg(&self_stride)
                    .arg(&rhs_stride)
                    .arg(&rhs_nbatch);
                let config = self.context.launch_config_2d(nstates, nbatch_u32, &f);
                unsafe { build.launch(config) }.expect("Failed to launch kernel");
            }
        }
    };
}

macro_rules! impl_sub_assign_mut {
    ($lhs:ty, $rhs:ty) => {
        impl<T: ScalarCuda> SubAssign<$rhs> for $lhs {
            fn sub_assign(&mut self, rhs: $rhs) {
                let f = self.context.function::<T>("vec_sub_assign");
                let nbatch = self.context.nbatch();
                let nstates = (self.data.len() / nbatch) as u32;
                let nbatch_u32 = nbatch as u32;
                let self_stride = nstates as i32;
                let rhs_nbatch = rhs.context.nbatch() as i32;
                let rhs_nstates = (rhs.data.len() / rhs_nbatch as usize) as u32;
                let rhs_stride = rhs_nstates as i32;
                let mut build = self.context.stream.launch_builder(&f);
                let rhs_data = rhs.data.as_view();
                build
                    .arg(&mut self.data)
                    .arg(&rhs_data)
                    .arg(&nstates)
                    .arg(&self_stride)
                    .arg(&rhs_stride)
                    .arg(&rhs_nbatch);
                let config = self.context.launch_config_2d(nstates, nbatch_u32, &f);
                unsafe { build.launch(config) }.expect("Failed to launch kernel");
            }
        }
    };
}

macro_rules! impl_add_assign_mut {
    ($lhs:ty, $rhs:ty) => {
        impl<T: ScalarCuda> AddAssign<$rhs> for $lhs {
            fn add_assign(&mut self, rhs: $rhs) {
                let f = self.context.function::<T>("vec_add_assign");
                let nbatch = self.context.nbatch();
                let nstates = (self.data.len() / nbatch) as u32;
                let nbatch_u32 = nbatch as u32;
                let self_stride = nstates as i32;
                let rhs_nbatch = rhs.context.nbatch() as i32;
                let rhs_nstates = (rhs.data.len() / rhs_nbatch as usize) as u32;
                let rhs_stride = rhs_nstates as i32;
                let mut build = self.context.stream.launch_builder(&f);
                let rhs_data = rhs.data.as_view();
                build
                    .arg(&mut self.data)
                    .arg(&rhs_data)
                    .arg(&nstates)
                    .arg(&self_stride)
                    .arg(&rhs_stride)
                    .arg(&rhs_nbatch);
                let config = self.context.launch_config_2d(nstates, nbatch_u32, &f);
                unsafe { build.launch(config) }.expect("Failed to launch kernel");
            }
        }
    };
}

impl_add_assign!(CudaMat<T>, &CudaMat<T>);
impl_add_assign!(CudaMat<T>, &CudaMatRef<'_, T>);
impl_add_assign!(CudaMatMut<'_, T>, &CudaMatRef<'_, T>);
impl_add_assign_mut!(CudaMatMut<'_, T>, &CudaMatMut<'_, T>);

impl_sub_assign!(CudaMat<T>, &CudaMat<T>);
impl_sub_assign!(CudaMat<T>, &CudaMatRef<'_, T>);
impl_sub_assign!(CudaMatMut<'_, T>, &CudaMatRef<'_, T>);
impl_sub_assign_mut!(CudaMatMut<'_, T>, &CudaMatMut<'_, T>);

macro_rules! impl_sub_both_ref {
    ($lhs:ty, $rhs:ty, $out:ty) => {
        impl<T: ScalarCuda> Sub<$rhs> for $lhs {
            type Output = $out;
            fn sub(self, rhs: $rhs) -> Self::Output {
                let nbatch = self.context.nbatch();
                let rhs_nbatch = rhs.context.nbatch();
                let max_nbatch = nbatch.max(rhs_nbatch);
                let nstates = (self.nrows() * self.ncols()) as u32;
                let nbatch_u32 = max_nbatch as u32;
                let self_stride = self.batch_stride as i32;
                let rhs_nbatch_i32 = rhs_nbatch as i32;
                let rhs_stride = (rhs.nrows() * rhs.ncols()) as i32;
                let mut ret = Self::Output::zeros(
                    self.nrows(),
                    self.ncols(),
                    self.context.clone_with_nbatch(max_nbatch).unwrap(),
                );
                let ret_nbatch = max_nbatch as i32;
                let ret_stride = (ret.nrows() * ret.ncols()) as i32;
                let f = self.context.function::<T>("vec_sub");
                let mut build = self.context.stream.launch_builder(&f);
                build
                    .arg(&self.data)
                    .arg(&rhs.data)
                    .arg(&mut ret.data)
                    .arg(&nstates)
                    .arg(&self_stride)
                    .arg(&rhs_stride)
                    .arg(&rhs_nbatch_i32)
                    .arg(&ret_stride)
                    .arg(&ret_nbatch);
                let config = self.context.launch_config_2d(nstates, nbatch_u32, &f);
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
                let nbatch = self.context.nbatch();
                let nstates = (self.data.len() / nbatch) as u32;
                let nbatch_u32 = nbatch as u32;
                let self_stride = nstates as i32;
                let rhs_nbatch = rhs.context.nbatch() as i32;
                let rhs_nstates = (rhs.data.len() / rhs_nbatch as usize) as u32;
                let rhs_stride = rhs_nstates as i32;
                let mut build = self.context.stream.launch_builder(&f);
                build
                    .arg(&mut self.data)
                    .arg(&rhs.data)
                    .arg(&nstates)
                    .arg(&self_stride)
                    .arg(&rhs_stride)
                    .arg(&rhs_nbatch);
                let config = self.context.launch_config_2d(nstates, nbatch_u32, &f);
                unsafe { build.launch(config) }.expect("Failed to launch kernel");
                self
            }
        }
    };
}

impl_sub_lhs!(CudaMat<T>, &CudaMat<T>, CudaMat<T>);
impl_sub_lhs!(CudaMat<T>, &CudaMatRef<'_, T>, CudaMat<T>);
impl_sub_both_ref!(CudaMatRef<'_, T>, &CudaMat<T>, CudaMat<T>);

macro_rules! impl_add_both_ref {
    ($lhs:ty, $rhs:ty, $out:ty) => {
        impl<T: ScalarCuda> Add<$rhs> for $lhs {
            type Output = $out;
            fn add(self, rhs: $rhs) -> Self::Output {
                let nbatch = self.context.nbatch();
                let rhs_nbatch = rhs.context.nbatch();
                let max_nbatch = nbatch.max(rhs_nbatch);
                let nstates = (self.nrows() * self.ncols()) as u32;
                let nbatch_u32 = max_nbatch as u32;
                let self_stride = self.batch_stride as i32;
                let rhs_nbatch_i32 = rhs_nbatch as i32;
                let rhs_stride = (rhs.nrows() * rhs.ncols()) as i32;
                let mut ret = Self::Output::zeros(
                    self.nrows(),
                    self.ncols(),
                    self.context.clone_with_nbatch(max_nbatch).unwrap(),
                );
                let ret_nbatch = max_nbatch as i32;
                let ret_stride = (ret.nrows() * ret.ncols()) as i32;
                let f = self.context.function::<T>("vec_add");
                let mut build = self.context.stream.launch_builder(&f);
                build
                    .arg(&self.data)
                    .arg(&rhs.data)
                    .arg(&mut ret.data)
                    .arg(&nstates)
                    .arg(&self_stride)
                    .arg(&rhs_stride)
                    .arg(&rhs_nbatch_i32)
                    .arg(&ret_stride)
                    .arg(&ret_nbatch);
                let config = self.context.launch_config_2d(nstates, nbatch_u32, &f);
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
                let nbatch = self.context.nbatch();
                let nstates = (self.data.len() / nbatch) as u32;
                let nbatch_u32 = nbatch as u32;
                let self_stride = nstates as i32;
                let rhs_nbatch = rhs.context.nbatch() as i32;
                let rhs_nstates = (rhs.data.len() / rhs_nbatch as usize) as u32;
                let rhs_stride = rhs_nstates as i32;
                let mut build = self.context.stream.launch_builder(&f);
                build
                    .arg(&mut self.data)
                    .arg(&rhs.data)
                    .arg(&nstates)
                    .arg(&self_stride)
                    .arg(&rhs_stride)
                    .arg(&rhs_nbatch);
                let config = self.context.launch_config_2d(nstates, nbatch_u32, &f);
                unsafe { build.launch(config) }.expect("Failed to launch kernel");
                self
            }
        }
    };
}

impl_add_lhs!(CudaMat<T>, &CudaMat<T>, CudaMat<T>);
impl_add_lhs!(CudaMat<T>, &CudaMatRef<'_, T>, CudaMat<T>);
impl_add_both_ref!(CudaMatRef<'_, T>, &CudaMat<T>, CudaMat<T>);

impl<'a, T: ScalarCuda> MatrixView<'a> for CudaMatRef<'a, T> {
    type Owned = CudaMat<T>;

    fn into_owned(self) -> Self::Owned {
        let nrows = self.nrows;
        let ncols = self.ncols;
        let nbatch = self.context.nbatch();
        let total = nrows * ncols * nbatch;
        let mut data = unsafe { self.context.stream.alloc(total) }
            .expect("Failed to allocate memory for CudaVec");
        let nstates = (nrows * ncols) as u32;
        let nbatch_u32 = nbatch as u32;
        let f = self.context.function::<T>("vec_copy");
        let config = self.context.launch_config_2d(nstates, nbatch_u32, &f);
        let mut build = self.context.stream.launch_builder(&f);
        let src_stride = self.batch_stride as i32;
        let dst_stride = (nrows * ncols) as i32;
        let nbatch_i32 = nbatch as i32;
        build
            .arg(&mut data)
            .arg(&self.data)
            .arg(&nstates)
            .arg(&dst_stride)
            .arg(&src_stride)
            .arg(&nbatch_i32);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
        CudaMat {
            data,
            context: self.context.clone(),
            nrows,
            ncols,
        }
    }

    fn gemv_o(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V) {
        let nbatch = self.context.nbatch();
        let x_nbatch = x.context.nbatch();
        self.context.assert_compatible_nbatch(x_nbatch, "gemv_o");
        let effective_nbatch = nbatch.max(x_nbatch);
        for b in 0..effective_nbatch {
            let self_b = if nbatch == 1 { 0 } else { b };
            let x_b = if x_nbatch == 1 { 0 } else { b };
            let x_nstates = x.data.len() as IndexType / x_nbatch;
            let a_start = self_b * self.batch_stride;
            let x_start = x_b * x_nstates;
            let y_start = b * self.nrows;
            let a_slice = self.data.slice(a_start..a_start + self.nrows * self.ncols);
            let x_slice = x.data.slice(x_start..x_start + self.ncols);
            let mut y_slice = y.data.slice_mut(y_start..y_start + self.nrows);
            self.context.gemv(
                self.nrows,
                self.ncols,
                alpha,
                beta,
                &a_slice,
                &x_slice,
                &mut y_slice,
            );
        }
    }
    fn gemv_v(
        &self,
        alpha: Self::T,
        x: &<Self::V as crate::vector::Vector>::View<'_>,
        beta: Self::T,
        y: &mut Self::V,
    ) {
        let nbatch = self.context.nbatch();
        let x_nbatch = x.context.nbatch();
        self.context.assert_compatible_nbatch(x_nbatch, "gemv_v");
        let effective_nbatch = nbatch.max(x_nbatch);
        for b in 0..effective_nbatch {
            let self_b = if nbatch == 1 { 0 } else { b };
            let x_b = if x_nbatch == 1 { 0 } else { b };
            let x_stride_val = x.data.len() as IndexType / x_nbatch;
            let a_start = self_b * self.batch_stride;
            let x_start = x_b * x_stride_val + x.col_offset;
            let y_start = b * self.nrows;
            let a_slice = self.data.slice(a_start..a_start + self.nrows * self.ncols);
            let x_slice = x.data.slice(x_start..x_start + self.ncols);
            let mut y_slice = y.data.slice_mut(y_start..y_start + self.nrows);
            self.context.gemv(
                self.nrows,
                self.ncols,
                alpha,
                beta,
                &a_slice,
                &x_slice,
                &mut y_slice,
            );
        }
    }
}

impl<'a, T: ScalarCuda> MatrixViewMut<'a> for CudaMatMut<'a, T> {
    type Owned = CudaMat<T>;
    type View = CudaMatRef<'a, T>;

    fn into_owned(self) -> Self::Owned {
        let total = self.nrows * self.ncols * self.context.nbatch();
        let mut data = unsafe { self.context.stream.alloc(total) }
            .expect("Failed to allocate memory for CudaVec");
        let nrows = self.nrows;
        let ncols = self.ncols;
        let nbatch = self.context.nbatch();
        let nstates = (nrows * ncols) as u32;
        let nbatch_u32 = nbatch as u32;
        let f = self.context.function::<T>("vec_copy");
        let config = self.context.launch_config_2d(nstates, nbatch_u32, &f);
        let mut build = self.context.stream.launch_builder(&f);
        let src_stride = self.batch_stride as i32;
        let dst_stride = (nrows * ncols) as i32;
        let nbatch_i32 = nbatch as i32;
        let src_data = self.data.slice(..);
        build
            .arg(&mut data)
            .arg(&src_data)
            .arg(&nstates)
            .arg(&dst_stride)
            .arg(&src_stride)
            .arg(&nbatch_i32);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
        CudaMat {
            data,
            context: self.context.clone(),
            nrows,
            ncols,
        }
    }

    fn gemm_oo(&mut self, alpha: Self::T, a: &Self::Owned, b: &Self::Owned, beta: Self::T) {
        let nbatch = self.context.nbatch();
        let a_nbatch = a.context.nbatch();
        let b_nbatch = b.context.nbatch();
        let self_nrows = self.nrows;
        let _self_ncols = self.ncols;
        let self_batch_stride = self.batch_stride;
        self.context.assert_compatible_nbatch(a_nbatch, "gemm_oo_a");
        self.context.assert_compatible_nbatch(b_nbatch, "gemm_oo_b");
        let effective_nbatch = nbatch.max(a_nbatch).max(b_nbatch) as c_int;
        let blas = CudaBlas::new(self.context.stream.clone()).expect("Failed to create CudaBlas");

        let nrows_a = a.nrows as c_int;
        let ncols_a = a.ncols as c_int;
        let nrows_b = b.nrows as c_int;
        let ncols_b = b.ncols as c_int;
        let nrows_c = self_nrows as c_int;

        let stride_a: c_longlong = if a_nbatch == 1 {
            0
        } else {
            (a.nrows * a.ncols) as c_longlong
        };
        let stride_b: c_longlong = if b_nbatch == 1 {
            0
        } else {
            (b.nrows * b.ncols) as c_longlong
        };
        let stride_c: c_longlong = if nbatch == 1 {
            0
        } else {
            self_batch_stride as c_longlong
        };

        let (a_ptr, _) = a.data.device_ptr(&self.context.stream);
        let (b_ptr, _) = b.data.device_ptr(&self.context.stream);
        let (c_ptr, _) = self.data.device_ptr_mut(&self.context.stream);

        let alpha_f64 = alpha.as_f64();
        let beta_f64 = beta.as_f64();
        unsafe {
            cublas::cublasDgemmStridedBatched(
                *blas.handle(),
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_N,
                nrows_a,
                ncols_b,
                ncols_a,
                &alpha_f64 as *const f64,
                a_ptr as *const f64,
                nrows_a,
                stride_a,
                b_ptr as *const f64,
                nrows_b,
                stride_b,
                &beta_f64 as *const f64,
                c_ptr as *mut f64,
                nrows_c,
                stride_c,
                effective_nbatch,
            )
        }
        .result()
        .expect("Failed to launch gemm");
    }
    fn gemm_vo(&mut self, alpha: Self::T, a: &Self::View, b: &Self::Owned, beta: Self::T) {
        let nbatch = self.context.nbatch();
        let a_nbatch = a.context.nbatch();
        let b_nbatch = b.context.nbatch();
        let self_nrows = self.nrows;
        let _self_ncols = self.ncols;
        let self_batch_stride = self.batch_stride;
        self.context.assert_compatible_nbatch(a_nbatch, "gemm_vo_a");
        self.context.assert_compatible_nbatch(b_nbatch, "gemm_vo_b");
        let effective_nbatch = nbatch.max(a_nbatch).max(b_nbatch) as c_int;
        let blas = CudaBlas::new(self.context.stream.clone()).expect("Failed to create CudaBlas");

        let nrows_a = a.nrows as c_int;
        let ncols_a = a.ncols as c_int;
        let nrows_b = b.nrows as c_int;
        let ncols_b = b.ncols as c_int;
        let nrows_c = self_nrows as c_int;

        let stride_a: c_longlong = if a_nbatch == 1 {
            0
        } else {
            a.batch_stride as c_longlong
        };
        let stride_b: c_longlong = if b_nbatch == 1 {
            0
        } else {
            (b.nrows * b.ncols) as c_longlong
        };
        let stride_c: c_longlong = if nbatch == 1 {
            0
        } else {
            self_batch_stride as c_longlong
        };

        let (a_ptr, _) = a.data.device_ptr(&self.context.stream);
        let (b_ptr, _) = b.data.device_ptr(&self.context.stream);
        let (c_ptr, _) = self.data.device_ptr_mut(&self.context.stream);

        let alpha_f64 = alpha.as_f64();
        let beta_f64 = beta.as_f64();
        unsafe {
            cublas::cublasDgemmStridedBatched(
                *blas.handle(),
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_N,
                nrows_a,
                ncols_b,
                ncols_a,
                &alpha_f64 as *const f64,
                a_ptr as *const f64,
                nrows_a,
                stride_a,
                b_ptr as *const f64,
                nrows_b,
                stride_b,
                &beta_f64 as *const f64,
                c_ptr as *mut f64,
                nrows_c,
                stride_c,
                effective_nbatch,
            )
        }
        .result()
        .expect("Failed to launch gemm");
    }
}

impl<T: ScalarCuda> DenseMatrix for CudaMat<T> {
    type View<'a> = CudaMatRef<'a, T>;
    type ViewMut<'a> = CudaMatMut<'a, T>;

    fn resize_cols(&mut self, new_ncols: IndexType) {
        let nbatch = self.context.nbatch();
        if new_ncols == self.ncols {
            return;
        }
        let old_ncols = self.ncols;
        let nrows = self.nrows;
        let cols_to_copy = old_ncols.min(new_ncols);
        let old_batch_elems = nrows * old_ncols;
        let new_batch_elems = nrows * new_ncols;
        let total_new = nrows * new_ncols * nbatch;
        let mut new_data = unsafe {
            self.context
                .stream
                .alloc(total_new)
                .expect("Failed to allocate memory for resized CudaMat")
        };
        let elements_per_batch = nrows * cols_to_copy;
        for b in 0..nbatch {
            let old_offset = b * old_batch_elems;
            let new_offset = b * new_batch_elems;
            if elements_per_batch > 0 {
                self.context
                    .stream
                    .memcpy_dtod(
                        &self.data.slice(old_offset..old_offset + elements_per_batch),
                        &mut new_data.slice_mut(new_offset..new_offset + elements_per_batch),
                    )
                    .expect("Failed to copy data during resize_cols");
            }
            if new_ncols > old_ncols {
                let zero_start = new_offset + elements_per_batch;
                let zero_len = nrows * (new_ncols - old_ncols);
                if zero_len > 0 {
                    self.context
                        .stream
                        .memset_zeros(&mut new_data.slice_mut(zero_start..zero_start + zero_len))
                        .expect("Failed to zero out new columns in resized CudaMat");
                }
            }
        }
        self.data = new_data;
        self.ncols = new_ncols;
    }

    fn from_vec(nrows: IndexType, ncols: IndexType, data: Vec<Self::T>, ctx: Self::C) -> Self {
        let nbatch = ctx.nbatch();
        assert_eq!(data.len(), nrows * ncols * nbatch);
        let mut device_data = unsafe {
            ctx.stream
                .alloc(data.len())
                .expect("Failed to allocate memory for CudaVec")
        };
        ctx.stream
            .memcpy_htod(&data, &mut device_data)
            .expect("Failed to copy data from host to device");
        Self {
            nrows,
            ncols,
            data: device_data,
            context: ctx,
        }
    }

    fn get_index(&self, i: IndexType, j: IndexType) -> Self::T {
        self.get_index(i, j)
    }

    fn gemm(&mut self, alpha: Self::T, a: &Self, b: &Self, beta: Self::T) {
        let nbatch = self.context.nbatch();
        let a_nbatch = a.context.nbatch();
        let b_nbatch = b.context.nbatch();
        let self_nrows = self.nrows;
        let self_ncols = self.ncols;
        self.context.assert_compatible_nbatch(a_nbatch, "gemm_a");
        self.context.assert_compatible_nbatch(b_nbatch, "gemm_b");
        let effective_nbatch = nbatch.max(a_nbatch).max(b_nbatch) as c_int;
        let blas = CudaBlas::new(self.context.stream.clone()).expect("Failed to create CudaBlas");

        let nrows_a = a.nrows as c_int;
        let ncols_a = a.ncols as c_int;
        let nrows_b = b.nrows as c_int;
        let ncols_b = b.ncols as c_int;
        let nrows_c = self_nrows as c_int;

        let stride_a: c_longlong = if a_nbatch == 1 {
            0
        } else {
            (a.nrows * a.ncols) as c_longlong
        };
        let stride_b: c_longlong = if b_nbatch == 1 {
            0
        } else {
            (b.nrows * b.ncols) as c_longlong
        };
        let stride_c: c_longlong = if nbatch == 1 {
            0
        } else {
            (self_nrows * self_ncols) as c_longlong
        };

        let (a_ptr, _) = a.data.device_ptr(&self.context.stream);
        let (b_ptr, _) = b.data.device_ptr(&self.context.stream);
        let (c_ptr, _) = self.data.device_ptr_mut(&self.context.stream);

        let alpha_f64 = alpha.as_f64();
        let beta_f64 = beta.as_f64();
        unsafe {
            cublas::cublasDgemmStridedBatched(
                *blas.handle(),
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_N,
                nrows_a,
                ncols_b,
                ncols_a,
                &alpha_f64 as *const f64,
                a_ptr as *const f64,
                nrows_a,
                stride_a,
                b_ptr as *const f64,
                nrows_b,
                stride_b,
                &beta_f64 as *const f64,
                c_ptr as *mut f64,
                nrows_c,
                stride_c,
                effective_nbatch,
            )
        }
        .result()
        .expect("Failed to launch gemm");
    }
    fn column_mut(&mut self, i: usize) -> <Self::V as Vector>::ViewMut<'_> {
        let nrows = self.nrows();
        let context = self.context.clone();
        CudaVecMut {
            data: self.data.as_view_mut(),
            context,
            nstates: nrows,
            col_offset: i * nrows,
        }
    }

    fn columns_mut(&mut self, start: usize, end: usize) -> Self::ViewMut<'_> {
        assert!(start < end, "Start index must be less than end index");
        assert!(end <= self.ncols(), "End index out of bounds");
        assert!(start < self.ncols(), "Start index out of bounds");
        let nbatch = self.context.nbatch();
        let nrows = self.nrows();
        let ncols = self.ncols();
        let num_cols = end - start;
        let start_i = start * nrows;
        let end_i = (nbatch - 1) * nrows * ncols + end * nrows;
        let data = self.data.slice_mut(start_i..end_i);
        CudaMatMut {
            data,
            context: self.context.clone(),
            nrows,
            ncols: num_cols,
            batch_stride: nrows * ncols,
        }
    }

    fn set_index(&mut self, i: IndexType, j: IndexType, value: Self::T) {
        self.set_index(i, j, value);
    }

    fn column(&self, i: usize) -> <Self::V as Vector>::View<'_> {
        let nrows = self.nrows();
        CudaVecRef {
            data: self.data.as_view(),
            context: self.context.clone(),
            nstates: nrows,
            col_offset: i * nrows,
        }
    }
    fn columns(&self, start: usize, end: usize) -> Self::View<'_> {
        assert!(start < end, "Start index must be less than end index");
        assert!(end <= self.ncols(), "End index out of bounds");
        assert!(start < self.ncols(), "Start index out of bounds");
        let nbatch = self.context.nbatch();
        let nrows = self.nrows();
        let _ncols = self.ncols();
        let num_cols = end - start;
        let start_i = start * nrows;
        let end_i = (nbatch - 1) * nrows * _ncols + end * nrows;
        let data = self.data.slice(start_i..end_i);
        CudaMatRef {
            data,
            context: self.context.clone(),
            nrows,
            ncols: num_cols,
            batch_stride: nrows * _ncols,
        }
    }

    fn column_axpy(&mut self, alpha: Self::T, j: IndexType, i: IndexType) {
        if i >= self.ncols() {
            panic!("Column index out of bounds");
        }
        if j >= self.ncols() {
            panic!("Column index out of bounds");
        }
        if i == j {
            panic!("Column index cannot be the same");
        }

        let nbatch = self.context.nbatch();
        let nrows = self.nrows();
        let ncols = self.ncols();
        let f = self.context.function::<T>("vec_axpy_offset");
        let nrows_u32 = nrows as u32;
        let nbatch_u32 = nbatch as u32;
        let config = self.context.launch_config_2d(nrows_u32, nbatch_u32, &f);
        let mut build = self.context.stream.launch_builder(&f);
        let alpha_val = alpha;
        let beta_val = T::one();
        let stride = (nrows * ncols) as i32;
        let nbatch_i32 = nbatch as i32;
        let x_offset = (j * nrows) as i32;
        let y_offset = (i * nrows) as i32;
        let data_ptr = &mut self.data as *mut CudaSlice<T>;
        unsafe {
            build
                .arg(&mut *data_ptr)
                .arg(&*data_ptr)
                .arg(&alpha_val)
                .arg(&beta_val)
                .arg(&y_offset)
                .arg(&x_offset)
                .arg(&nrows_u32)
                .arg(&stride)
                .arg(&stride)
                .arg(&nbatch_i32);
        }
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }
}

impl<T: ScalarCuda> Matrix for CudaMat<T> {
    type Sparsity = Dense<Self>;
    type SparsityRef<'a> = DenseRef<'a, Self>;

    fn sparsity(&self) -> Option<Self::SparsityRef<'_>> {
        None
    }

    fn context(&self) -> &Self::C {
        &self.context
    }

    fn gather(&mut self, other: &Self, indices: &<Self::V as Vector>::Index) {
        let nbatch = self.context.nbatch();
        let other_nbatch = other.context.nbatch();
        let self_nrows = self.nrows;
        let self_ncols = self.ncols;
        let other_nrows = other.nrows;
        let other_ncols = other.ncols;
        let f = self.context.function::<T>("vec_gather");
        let n_indices = indices.len() as u32;
        if n_indices == 0 {
            return;
        }
        let nbatch_u32 = nbatch as u32;
        let config = self.context.launch_config_2d(n_indices, nbatch_u32, &f);
        let mut build = self.context.stream.launch_builder(&f);
        let self_stride = (self_nrows * self_ncols) as i32;
        let self_nbatch_i32 = nbatch as i32;
        let other_stride = (other_nrows * other_ncols) as i32;
        let other_nbatch_i32 = other_nbatch as i32;
        build
            .arg(&mut self.data)
            .arg(&other.data)
            .arg(&indices.data)
            .arg(&n_indices)
            .arg(&self_stride)
            .arg(&self_nbatch_i32)
            .arg(&other_stride)
            .arg(&other_nbatch_i32);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }

    fn set_data_with_indices(
        &mut self,
        dst_indices: &<Self::V as Vector>::Index,
        src_indices: &<Self::V as Vector>::Index,
        data: &Self::V,
    ) {
        assert_eq!(
            dst_indices.len(),
            src_indices.len(),
            "Destination and source indices must have the same length"
        );
        let nbatch = self.context.nbatch();
        let data_nbatch = data.context.nbatch();
        let f = self.context.function::<T>("mat_set_data_with_indices");
        let n = dst_indices.len() as u32;
        if n == 0 {
            return;
        }
        let nbatch_u32 = nbatch as u32;
        let config = self.context.launch_config_2d(n, nbatch_u32, &f);
        let mut build = self.context.stream.launch_builder(&f);
        let self_stride = (self.nrows * self.ncols) as i32;
        let self_nbatch_i32 = nbatch as i32;
        let data_nstates = data.data.len() as IndexType / data_nbatch;
        let other_stride = data_nstates as i32;
        let other_nbatch_i32 = data_nbatch as i32;
        build
            .arg(&mut self.data)
            .arg(&data.data)
            .arg(&dst_indices.data)
            .arg(&src_indices.data)
            .arg(&n)
            .arg(&self_stride)
            .arg(&self_nbatch_i32)
            .arg(&other_stride)
            .arg(&other_nbatch_i32);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }

    fn add_column_to_vector(&self, j: IndexType, v: &mut Self::V) {
        let nbatch = self.context.nbatch();
        let v_nbatch = v.context.nbatch();
        self.context
            .assert_compatible_nbatch(v_nbatch, "add_column_to_vector");
        let nrows = self.nrows();
        let ncols = self.ncols();
        let v_nstates = v.data.len() as IndexType / v_nbatch;
        let f = self.context.function::<T>("vec_axpy_offset");
        let nrows_u32 = nrows as u32;
        let nbatch_u32 = nbatch as u32;
        let config = self.context.launch_config_2d(nrows_u32, nbatch_u32, &f);
        let mut build = self.context.stream.launch_builder(&f);
        let alpha_val = T::one();
        let beta_val = T::one();
        let v_stride = v_nstates as i32;
        let mat_stride = (nrows * ncols) as i32;
        let mat_nbatch_i32 = nbatch as i32;
        let y_offset: i32 = 0;
        let x_offset = (j * nrows) as i32;
        build
            .arg(&mut v.data)
            .arg(&self.data)
            .arg(&alpha_val)
            .arg(&beta_val)
            .arg(&y_offset)
            .arg(&x_offset)
            .arg(&nrows_u32)
            .arg(&v_stride)
            .arg(&mat_stride)
            .arg(&mat_nbatch_i32);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }

    fn triplet_iter(
        &self,
    ) -> (
        impl Iterator<Item = (IndexType, IndexType)> + '_,
        impl Iterator<Item = Self::T> + '_,
    ) {
        let nrows = self.nrows();
        let ncols = self.ncols();
        let nbatch = self.context.nbatch();
        let data = self
            .context
            .stream
            .clone_dtoh(&self.data)
            .expect("Failed to copy data from device to host");
        let indices = (0..ncols).flat_map(move |j| (0..nrows).map(move |i| (i, j)));
        let mut values = Vec::with_capacity(nrows * ncols * nbatch);
        for b in 0..nbatch {
            let offset = b * nrows * ncols;
            for j in 0..ncols {
                for i in 0..nrows {
                    values.push(data[offset + i + j * nrows]);
                }
            }
        }
        (indices, values.into_iter())
    }

    fn try_from_triplets(
        nrows: IndexType,
        ncols: IndexType,
        indices: Vec<(IndexType, IndexType)>,
        values: Vec<T>,
        ctx: Self::C,
    ) -> Result<Self, DiffsolError> {
        let nbatch = ctx.nbatch();
        let nnz = indices.len();
        assert_eq!(
            values.len(),
            nnz * nbatch,
            "Expected {} values ({} triplets * {} batches), got {}",
            nnz * nbatch,
            nnz,
            nbatch,
            values.len()
        );
        let mut m = vec![T::zero(); nrows * ncols * nbatch];
        for b in 0..nbatch {
            let batch_offset = b * nrows * ncols;
            for (k, &(i, j)) in indices.iter().enumerate() {
                if i >= nrows || j >= ncols {
                    return Err(matrix_error!(IndexOutOfBounds));
                }
                m[batch_offset + i + j * nrows] = values[b * nnz + k];
            }
        }
        Ok(Self::from_vec(nrows, ncols, m, ctx))
    }
    fn gemv(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V) {
        let nbatch = self.context.nbatch();
        let x_nbatch = x.context.nbatch();
        self.context.assert_compatible_nbatch(x_nbatch, "gemv");
        let effective_nbatch = nbatch.max(x_nbatch);
        for b in 0..effective_nbatch {
            let self_b = if nbatch == 1 { 0 } else { b };
            let x_b = if x_nbatch == 1 { 0 } else { b };
            let x_nstates = x.data.len() as IndexType / x_nbatch;
            let self_batch_size = self.nrows * self.ncols;
            let a_start = self_b * self_batch_size;
            let x_start = x_b * x_nstates;
            let y_start = b * self.nrows;
            let a_slice = self.data.slice(a_start..a_start + self_batch_size);
            let x_slice = x.data.slice(x_start..x_start + self.ncols);
            let mut y_slice = y.data.slice_mut(y_start..y_start + self.nrows);
            self.context.gemv(
                self.nrows,
                self.ncols,
                alpha,
                beta,
                &a_slice,
                &x_slice,
                &mut y_slice,
            );
        }
    }
    fn zeros(nrows: IndexType, ncols: IndexType, ctx: Self::C) -> Self {
        let nbatch = ctx.nbatch();
        let data = ctx
            .stream
            .alloc_zeros(nrows * ncols * nbatch)
            .expect("Failed to allocate memory for CudaMat");
        Self {
            data,
            context: ctx,
            nrows,
            ncols,
        }
    }
    fn copy_from(&mut self, other: &Self) {
        let self_nbatch = self.context.nbatch();
        let other_nbatch = other.context.nbatch();
        self.context
            .assert_compatible_nbatch(other_nbatch, "copy_from");
        let nrows = self.nrows;
        let self_ncols = self.ncols;
        let other_ncols = other.ncols;
        let f = self.context.function::<T>("vec_copy");
        let nstates = (nrows * self_ncols) as u32;
        let nbatch_u32 = self_nbatch as u32;
        let config = self.context.launch_config_2d(nstates, nbatch_u32, &f);
        let mut build = self.context.stream.launch_builder(&f);
        let self_stride = (nrows * self_ncols) as i32;
        let other_stride = (nrows * other_ncols) as i32;
        let other_nbatch_i32 = other_nbatch as i32;
        build
            .arg(&mut self.data)
            .arg(&other.data)
            .arg(&nstates)
            .arg(&self_stride)
            .arg(&other_stride)
            .arg(&other_nbatch_i32);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }

    fn from_diagonal(v: &Self::V) -> Self {
        let ctx = v.context.clone();
        let nbatch = ctx.nbatch();
        let nstates = v.data.len() as IndexType / nbatch;
        let mut data = ctx
            .stream
            .alloc_zeros(nstates * nstates * nbatch)
            .expect("Failed to allocate memory for CudaMat");
        let f = ctx.function::<T>("mat_from_diagonal");
        let n_u32 = nstates as u32;
        let nbatch_u32 = nbatch as u32;
        let config = ctx.launch_config_2d(n_u32, nbatch_u32, &f);
        let mut build = ctx.stream.launch_builder(&f);
        let mat_stride = (nstates * nstates) as i32;
        let mat_nbatch_i32 = nbatch as i32;
        let diag_stride = nstates as i32;
        let diag_nbatch_i32 = v.context.nbatch() as i32;
        build
            .arg(&mut data)
            .arg(&v.data)
            .arg(&n_u32)
            .arg(&mat_stride)
            .arg(&mat_nbatch_i32)
            .arg(&diag_stride)
            .arg(&diag_nbatch_i32);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
        Self {
            data,
            context: ctx,
            nrows: nstates,
            ncols: nstates,
        }
    }

    fn partition_indices_by_zero_diagonal(
        &self,
    ) -> (<Self::V as Vector>::Index, <Self::V as Vector>::Index) {
        let nbatch = self.context.nbatch();
        let diagonal = self.diagonal().clone_as_vec();
        let nstates = diagonal.len() / nbatch;
        let (zero_indices, nonzero_indices) = (0..nstates).fold(
            (Vec::new(), Vec::new()),
            |(mut zero_indices, mut nonzero_indices), i| {
                let val = diagonal[i];
                if val.is_zero() {
                    zero_indices.push(i);
                } else {
                    nonzero_indices.push(i);
                }
                (zero_indices, nonzero_indices)
            },
        );
        (
            <Self::V as Vector>::Index::from_vec(zero_indices, self.context.clone()),
            <Self::V as Vector>::Index::from_vec(nonzero_indices, self.context.clone()),
        )
    }
    fn set_column(&mut self, j: IndexType, v: &Self::V) {
        let nbatch = self.context.nbatch();
        let v_nbatch = v.context.nbatch();
        self.context
            .assert_compatible_nbatch(v_nbatch, "set_column");
        let nrows = self.nrows();
        let v_nstates = v.data.len() as IndexType / v_nbatch;
        assert_eq!(
            v_nstates, nrows,
            "Column length mismatch: {} != {}",
            v_nstates, nrows
        );
        let f = self.context.function::<T>("mat_set_column");
        let n_u32 = nrows as u32;
        let nbatch_u32 = nbatch as u32;
        let config = self.context.launch_config_2d(n_u32, nbatch_u32, &f);
        let mut build = self.context.stream.launch_builder(&f);
        let j_cint = j as c_int;
        let mat_stride = (nrows * self.ncols) as i32;
        let mat_nbatch_i32 = nbatch as i32;
        let col_stride = v_nstates as i32;
        let col_nbatch_i32 = v_nbatch as i32;
        build
            .arg(&mut self.data)
            .arg(&v.data)
            .arg(&j_cint)
            .arg(&n_u32)
            .arg(&mat_stride)
            .arg(&mat_nbatch_i32)
            .arg(&col_stride)
            .arg(&col_nbatch_i32);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }

    /// Perform the assignment self = x + beta * y where x and y are matrices and beta is a scalar
    fn scale_add_and_assign(&mut self, x: &Self, beta: Self::T, y: &Self) {
        let nbatch = self.context.nbatch();
        let x_nbatch = x.context.nbatch();
        let y_nbatch = y.context.nbatch();
        self.context
            .assert_compatible_nbatch(x_nbatch, "scale_add_and_assign_x");
        self.context
            .assert_compatible_nbatch(y_nbatch, "scale_add_and_assign_y");
        let f = self.context.function::<T>("mat_scale_add_assign");
        let nrows = self.nrows;
        let self_ncols = self.ncols;
        let x_ncols = x.ncols;
        let y_ncols = y.ncols;
        let nstates = (nrows * self_ncols) as u32;
        let nbatch_u32 = nbatch as u32;
        let config = self.context.launch_config_2d(nstates, nbatch_u32, &f);
        let mut build = self.context.stream.launch_builder(&f);
        let self_stride = (nrows * self_ncols) as i32;
        let x_stride = (nrows * x_ncols) as i32;
        let x_nbatch_i32 = x_nbatch as i32;
        let y_stride = (nrows * y_ncols) as i32;
        let y_nbatch_i32 = y_nbatch as i32;
        build
            .arg(&mut self.data)
            .arg(&x.data)
            .arg(&y.data)
            .arg(&beta)
            .arg(&nstates)
            .arg(&self_stride)
            .arg(&x_stride)
            .arg(&x_nbatch_i32)
            .arg(&y_stride)
            .arg(&y_nbatch_i32);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }

    fn new_from_sparsity(
        nrows: IndexType,
        ncols: IndexType,
        _sparsity: Option<Self::Sparsity>,
        ctx: Self::C,
    ) -> Self {
        Self::zeros(nrows, ncols, ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    super::super::generate_matrix_tests!(
        cuda,
        CudaMat<f64>,
        CudaContext::default(),
        CudaContext::default().with_nbatch(2)
    );

    super::super::generate_dense_matrix_tests!(
        cuda,
        CudaMat<f64>,
        CudaContext::default(),
        CudaContext::default().with_nbatch(2)
    );
}
