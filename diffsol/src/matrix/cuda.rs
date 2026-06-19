use cudarc::cublas::sys as cublas;
use cudarc::{
    cublas::{sys::cublasOperation_t, CudaBlas},
    driver::{CudaSlice, CudaView, CudaViewMut, DevicePtr, DevicePtrMut, PushKernelArg},
};
use std::ffi::c_int;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use crate::{
    error::{DiffsolError, MatrixError},
    linear_solver::cuda::lu::CudaLU,
    matrix::default_solver::DefaultSolver,
    matrix_error, Context, CudaContext, CudaType, CudaVec, CudaVecMut, CudaVecRef, IndexType,
    MatrixCommon, ScalarCuda, Scale, Vector, VectorIndex,
};

use super::{
    sparsity::{Dense, DenseRef},
    DenseMatrix, Matrix, MatrixView, MatrixViewMut,
};

/// a CUDA matrix in column-major order
#[derive(Clone, Debug)]
pub struct CudaMat<T: ScalarCuda> {
    pub(crate) data: CudaSlice<T>,
    pub(crate) context: CudaContext,
    nrows: IndexType,
    ncols: IndexType,
}

#[derive(Debug)]
pub struct CudaMatRef<'a, T: ScalarCuda> {
    pub(crate) data: CudaView<'a, T>,
    pub(crate) context: CudaContext,
    nrows: IndexType,
    ncols: IndexType,
    batch_stride: IndexType,
}

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
    fn gemv<T: ScalarCuda, A: DevicePtr<T>, X: DevicePtr<T>, Y: DevicePtrMut<T>>(
        &self,
        nrows: IndexType,
        ncols: IndexType,
        alpha: T,
        beta: T,
        a: &A,
        x: &X,
        y: &mut Y,
    ) {
        let (a, _syn_a) = a.device_ptr(&self.stream);
        let (x, _syn_x) = x.device_ptr(&self.stream);
        let (y, _syn_y) = y.device_ptr_mut(&self.stream);
        let blas = CudaBlas::new(self.stream.clone()).expect("Failed to create CudaBlas");
        match T::as_enum() {
            CudaType::F64 => {
                unsafe {
                    cublas::cublasDgemv_v2(
                        *blas.handle(),
                        cublasOperation_t::CUBLAS_OP_N,
                        nrows as c_int,
                        ncols as c_int,
                        &alpha.as_f64(),
                        a as *const f64,
                        nrows as c_int,
                        x as *const f64,
                        1,
                        &beta.as_f64(),
                        y as *mut f64,
                        1,
                    )
                }
                .result()
                .expect("Failed to launch gemv");
            }
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn gemm<T: ScalarCuda, A: DevicePtr<T>, B: DevicePtr<T>, C: DevicePtrMut<T>>(
        &self,
        nrows_a: IndexType,
        ncols_a: IndexType,
        nrows_b: IndexType,
        ncols_b: IndexType,
        nrows_c: IndexType,
        alpha: T,
        beta: T,
        a: &A,
        b: &B,
        c: &mut C,
    ) {
        assert_eq!(nrows_a, nrows_c);
        assert_eq!(ncols_a, nrows_b);
        let (a, _syn_a) = a.device_ptr(&self.stream);
        let (b, _syn_b) = b.device_ptr(&self.stream);
        let (c, _syn_c) = c.device_ptr_mut(&self.stream);
        let blas = CudaBlas::new(self.stream.clone()).expect("Failed to create CudaBlas");
        match T::as_enum() {
            CudaType::F64 => {
                unsafe {
                    cublas::cublasDgemm_v2(
                        *blas.handle(),
                        cublasOperation_t::CUBLAS_OP_N,
                        cublasOperation_t::CUBLAS_OP_N,
                        nrows_a as c_int,
                        ncols_b as c_int,
                        ncols_a as c_int,
                        &alpha.as_f64(),
                        a as *const f64,
                        nrows_a as c_int,
                        b as *const f64,
                        nrows_b as c_int,
                        &beta.as_f64(),
                        c as *mut f64,
                        nrows_c as c_int,
                    )
                }
                .result()
                .expect("Failed to launch gemm");
            }
        }
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
        let batch_stride = n * n;
        for b in 0..nbatch {
            let src_start = b * batch_stride;
            let dst_start = b * n;
            let mut build = self.context.stream.launch_builder(&f);
            let src_slice = self.data.slice(src_start..src_start + batch_stride);
            let mut dst_slice = data.slice_mut(dst_start..dst_start + n);
            build.arg(&src_slice).arg(&mut dst_slice).arg(&n_u32);
            let config = self.context.launch_config_1d(n_u32, &f);
            unsafe { build.launch(config) }.expect("Failed to launch kernel");
        }
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
                let mut ret = Self::Output::zeros(self.nrows(), self.ncols(), self.context.clone());
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

impl_mul_scalar_alloc!(CudaMatRef<'_, T>, CudaMat<T>, T);
impl_mul_scalar!(CudaMat<T>, CudaMat<T>);
impl_mul_scalar_alloc!(&CudaMat<T>, CudaMat<T>, T);

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

impl_mul_assign_scalar!(CudaMatMut<'_, T>, T);

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

macro_rules! impl_sub_assign_mut {
    ($lhs:ty, $rhs:ty) => {
        impl<T: ScalarCuda> SubAssign<$rhs> for $lhs {
            fn sub_assign(&mut self, rhs: $rhs) {
                let f = self.context.function::<T>("vec_sub_assign");
                let n = self.data.len() as u32;
                let mut build = self.context.stream.launch_builder(&f);
                let rhs_data = rhs.data.as_view();
                build.arg(&mut self.data).arg(&rhs_data).arg(&n);
                let config = self.context.launch_config_1d(n, &f);
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
                let n = self.data.len() as u32;
                let mut build = self.context.stream.launch_builder(&f);
                let rhs_data = rhs.data.as_view();
                build.arg(&mut self.data).arg(&rhs_data).arg(&n);
                let config = self.context.launch_config_1d(n, &f);
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
                assert_eq!(
                    self.data.len(),
                    rhs.data.len(),
                    "Vector length mismatch: {} != {}",
                    self.data.len(),
                    rhs.data.len()
                );
                let mut ret = Self::Output::zeros(self.nrows(), self.ncols(), self.context.clone());
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

impl_sub_lhs!(CudaMat<T>, &CudaMat<T>, CudaMat<T>);
impl_sub_lhs!(CudaMat<T>, &CudaMatRef<'_, T>, CudaMat<T>);
impl_sub_both_ref!(CudaMatRef<'_, T>, &CudaMat<T>, CudaMat<T>);

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
                let mut ret = Self::Output::zeros(self.nrows(), self.ncols(), self.context.clone());
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

impl_add_lhs!(CudaMat<T>, &CudaMat<T>, CudaMat<T>);
impl_add_lhs!(CudaMat<T>, &CudaMatRef<'_, T>, CudaMat<T>);
impl_add_both_ref!(CudaMatRef<'_, T>, &CudaMat<T>, CudaMat<T>);

impl<'a, T: ScalarCuda> MatrixView<'a> for CudaMatRef<'a, T> {
    type Owned = CudaMat<T>;

    fn into_owned(self) -> Self::Owned {
        let nrows = self.nrows;
        let ncols = self.ncols;
        let nbatch = self.context.nbatch();
        let batch_stride = self.batch_stride;
        let total = nrows * ncols * nbatch;
        let mut data = unsafe { self.context.stream.alloc(total) }
            .expect("Failed to allocate memory for CudaVec");
        for b in 0..nbatch {
            let src_start = b * batch_stride;
            let dst_start = b * nrows * ncols;
            self.context
                .stream
                .memcpy_dtod(
                    &self.data.slice(src_start..src_start + nrows * ncols),
                    &mut data.slice_mut(dst_start..dst_start + nrows * ncols),
                )
                .expect("Failed to copy data from device to host");
        }
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
        let nrows = self.nrows();
        let ncols = self.ncols();
        let batch_stride = self.batch_stride;
        for b in 0..nbatch {
            let x_b = if x_nbatch == 1 { 0 } else { b };
            let x_nstates = x.data.len() as IndexType / x_nbatch;
            let a_start = b * batch_stride;
            let x_start = x_b * x_nstates;
            let y_start = b * nrows;
            let a_slice = self.data.slice(a_start..a_start + nrows * ncols);
            let x_slice = x.data.slice(x_start..x_start + x_nstates);
            let mut y_slice = y.data.slice_mut(y_start..y_start + nrows);
            self.context
                .gemv(nrows, ncols, alpha, beta, &a_slice, &x_slice, &mut y_slice);
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
        let nrows = self.nrows();
        let ncols = self.ncols();
        let batch_stride = self.batch_stride;
        for b in 0..nbatch {
            let x_b = if x_nbatch == 1 { 0 } else { b };
            let x_stride = x.stride();
            let x_start = x_b * x_stride + x.col_offset;
            let x_nstates = x.nstates;
            let a_start = b * batch_stride;
            let y_start = b * nrows;
            let a_slice = self.data.slice(a_start..a_start + nrows * ncols);
            let x_slice = x.data.slice(x_start..x_start + x_nstates);
            let mut y_slice = y.data.slice_mut(y_start..y_start + nrows);
            self.context
                .gemv(nrows, ncols, alpha, beta, &a_slice, &x_slice, &mut y_slice);
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
        let batch_stride = self.batch_stride;
        let nrows = self.nrows;
        let ncols = self.ncols;
        let nbatch = self.context.nbatch();
        for b in 0..nbatch {
            let src_start = b * batch_stride;
            let dst_start = b * nrows * ncols;
            self.context
                .stream
                .memcpy_dtod(
                    &self.data.slice(src_start..src_start + nrows * ncols),
                    &mut data.slice_mut(dst_start..dst_start + nrows * ncols),
                )
                .expect("Failed to copy data from device to host");
        }
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
        let self_nrows = self.nrows();
        let self_ncols = self.ncols();
        let self_batch_stride = self.batch_stride;
        self.context.assert_compatible_nbatch(a_nbatch, "gemm_oo_a");
        self.context.assert_compatible_nbatch(b_nbatch, "gemm_oo_b");
        for b_idx in 0..nbatch {
            let a_b = if a_nbatch == 1 { 0 } else { b_idx };
            let bb = if b_nbatch == 1 { 0 } else { b_idx };
            let a_start = a_b * a.nrows * a.ncols;
            let b_start = bb * b.nrows * b.ncols;
            let self_start = b_idx * self_batch_stride;
            let a_slice = a.data.slice(a_start..a_start + a.nrows * a.ncols);
            let b_slice = b.data.slice(b_start..b_start + b.nrows * b.ncols);
            let mut self_slice = self
                .data
                .slice_mut(self_start..self_start + self_nrows * self_ncols);
            self.context.gemm(
                a.nrows(),
                a.ncols(),
                b.nrows(),
                b.ncols(),
                self_nrows,
                alpha,
                beta,
                &a_slice,
                &b_slice,
                &mut self_slice,
            );
        }
    }
    fn gemm_vo(&mut self, alpha: Self::T, a: &Self::View, b: &Self::Owned, beta: Self::T) {
        let nbatch = self.context.nbatch();
        let a_nbatch = a.context.nbatch();
        let b_nbatch = b.context.nbatch();
        let self_nrows = self.nrows();
        let self_ncols = self.ncols();
        let self_batch_stride = self.batch_stride;
        self.context.assert_compatible_nbatch(a_nbatch, "gemm_vo_a");
        self.context.assert_compatible_nbatch(b_nbatch, "gemm_vo_b");
        for b_idx in 0..nbatch {
            let a_b = if a_nbatch == 1 { 0 } else { b_idx };
            let bb = if b_nbatch == 1 { 0 } else { b_idx };
            let a_start = a_b * a.nrows * a.ncols;
            let b_start = bb * b.nrows * b.ncols;
            let self_start = b_idx * self_batch_stride;
            let a_slice = a.data.slice(a_start..a_start + a.nrows * a.ncols);
            let b_slice = b.data.slice(b_start..b_start + b.nrows * b.ncols);
            let mut self_slice = self
                .data
                .slice_mut(self_start..self_start + self_nrows * self_ncols);
            self.context.gemm(
                a.nrows(),
                a.ncols(),
                b.nrows(),
                b.ncols(),
                self_nrows,
                alpha,
                beta,
                &a_slice,
                &b_slice,
                &mut self_slice,
            );
        }
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
        let total_new = self.nrows * new_ncols * nbatch;
        let mut new_data = unsafe {
            self.context
                .stream
                .alloc(total_new)
                .expect("Failed to allocate memory for resized CudaMat")
        };
        let cols_to_copy = self.ncols.min(new_ncols);
        let elements_per_batch = self.nrows * cols_to_copy;
        for b in 0..nbatch {
            let old_offset = b * self.nrows * self.ncols;
            let new_offset = b * self.nrows * new_ncols;
            if elements_per_batch > 0 {
                self.context
                    .stream
                    .memcpy_dtod(
                        &self.data.slice(old_offset..old_offset + elements_per_batch),
                        &mut new_data.slice_mut(new_offset..new_offset + elements_per_batch),
                    )
                    .expect("Failed to copy data during resize_cols");
            }
            if new_ncols > self.ncols {
                let zero_start = new_offset + elements_per_batch;
                let zero_len = self.nrows * (new_ncols - self.ncols);
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
        let self_batch_size = self_nrows * self_ncols;
        self.context.assert_compatible_nbatch(a_nbatch, "gemm_a");
        self.context.assert_compatible_nbatch(b_nbatch, "gemm_b");
        let effective_nbatch = nbatch.max(a_nbatch).max(b_nbatch);
        for b_idx in 0..effective_nbatch {
            let a_b = if a_nbatch == 1 { 0 } else { b_idx };
            let bb = if b_nbatch == 1 { 0 } else { b_idx };
            let self_b = if nbatch == 1 { 0 } else { b_idx };
            let a_start = a_b * a.nrows * a.ncols;
            let b_start = bb * b.nrows * b.ncols;
            let self_start = self_b * self_batch_size;
            let a_slice = a.data.slice(a_start..a_start + a.nrows * a.ncols);
            let b_slice = b.data.slice(b_start..b_start + b.nrows * b.ncols);
            let mut self_slice = self
                .data
                .slice_mut(self_start..self_start + self_batch_size);
            self.context.gemm(
                a.nrows(),
                a.ncols(),
                b.nrows(),
                b.ncols(),
                self_nrows,
                alpha,
                beta,
                &a_slice,
                &b_slice,
                &mut self_slice,
            );
        }
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
        for b in 0..nbatch {
            let batch_offset = b * nrows * ncols;
            let start_x = batch_offset + j * nrows;
            let start_y = batch_offset + i * nrows;
            let x = {
                let x = self.data.slice(start_x..start_x + nrows);
                let (x, _syn_x) = x.device_ptr(&self.context.stream);
                x
            };
            let y = {
                let y = self.data.slice(start_y..start_y + nrows);
                let (y, _syn_y) = y.device_ptr(&self.context.stream);
                y
            };
            self.context.axpy_inner::<T>(alpha, x, y, nrows as c_int);
        }
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
        for b in 0..nbatch {
            let other_b = if other_nbatch == 1 { 0 } else { b };
            let self_nrows = self.nrows;
            let self_ncols = self.ncols;
            let other_nrows = other.nrows;
            let other_ncols = other.ncols;
            let self_batch_size = self_nrows * self_ncols;
            let other_batch_size = other_nrows * other_ncols;
            let self_start = b * self_batch_size;
            let other_start = other_b * other_batch_size;
            let f = self.context.function::<T>("vec_gather");
            let n = indices.len() as u32;
            if n == 0 {
                continue;
            }
            let config = self.context.launch_config_1d(n, &f);
            let mut build = self.context.stream.launch_builder(&f);
            let mut self_slice = self
                .data
                .slice_mut(self_start..self_start + self_batch_size);
            let other_slice = other
                .data
                .slice(other_start..other_start + other_batch_size);
            build
                .arg(&mut self_slice)
                .arg(&other_slice)
                .arg(&indices.data)
                .arg(&n);
            unsafe { build.launch(config) }.expect("Failed to launch kernel");
        }
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
        for b in 0..nbatch {
            let data_b = if data_nbatch == 1 { 0 } else { b };
            let self_batch_size = self.nrows * self.ncols;
            let data_nstates = data.data.len() as IndexType / data_nbatch;
            let self_start = b * self_batch_size;
            let data_start = data_b * data_nstates;
            let f = self.context.function::<T>("mat_set_data_with_indices");
            let n = dst_indices.len() as u32;
            if n == 0 {
                continue;
            }
            let config = self.context.launch_config_1d(n, &f);
            let mut build = self.context.stream.launch_builder(&f);
            let mut self_slice = self
                .data
                .slice_mut(self_start..self_start + self_batch_size);
            let data_slice = data.data.slice(data_start..data_start + data_nstates);
            build
                .arg(&mut self_slice)
                .arg(&data_slice)
                .arg(&dst_indices.data)
                .arg(&src_indices.data)
                .arg(&n);
            unsafe { build.launch(config) }.expect("Failed to launch kernel");
        }
    }

    fn add_column_to_vector(&self, j: IndexType, v: &mut Self::V) {
        let nbatch = self.context.nbatch();
        let v_nbatch = v.context.nbatch();
        self.context
            .assert_compatible_nbatch(v_nbatch, "add_column_to_vector");
        let nrows = self.nrows();
        let ncols = self.ncols();
        for b in 0..nbatch {
            let v_b = if v_nbatch == 1 { 0 } else { b };
            let v_nstates = v.data.len() as IndexType / v_nbatch;
            let col_start = b * nrows * ncols + j * nrows;
            let v_start = v_b * v_nstates;
            let col_slice = self.data.slice(col_start..col_start + nrows);
            let (x, _) = col_slice.device_ptr(&self.context.stream);
            let mut v_slice = v.data.slice_mut(v_start..v_start + v_nstates);
            let (y, _) = v_slice.device_ptr_mut(&self.context.stream);
            self.context.axpy_inner::<T>(T::one(), x, y, nrows as c_int);
        }
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
            let y_start = b * self.nrows();
            let a_slice = self.data.slice(a_start..a_start + self_batch_size);
            let x_slice = x.data.slice(x_start..x_start + self.ncols());
            let mut y_slice = y.data.slice_mut(y_start..y_start + self.nrows());
            self.context.gemv(
                self.nrows(),
                self.ncols(),
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
        if self_nbatch == other_nbatch {
            let f = self.context.function::<T>("vec_copy");
            let n = self.data.len() as u32;
            let mut build = self.context.stream.launch_builder(&f);
            build.arg(&mut self.data).arg(&other.data).arg(&n);
            let config = self.context.launch_config_1d(n, &f);
            unsafe { build.launch(config) }.expect("Failed to launch kernel");
        } else {
            for b in 0..self_nbatch {
                let src_b = if other_nbatch == 1 { 0 } else { b };
                let nrows = self.nrows;
                let self_ncols = self.ncols;
                let other_ncols = other.ncols;
                let self_start = b * nrows * self_ncols;
                let other_start = src_b * nrows * other_ncols;
                let n = (nrows * self_ncols) as u32;
                let f = self.context.function::<T>("vec_copy");
                let mut build = self.context.stream.launch_builder(&f);
                let mut self_slice = self
                    .data
                    .slice_mut(self_start..self_start + nrows * self_ncols);
                let other_slice = other
                    .data
                    .slice(other_start..other_start + nrows * other_ncols);
                build.arg(&mut self_slice).arg(&other_slice).arg(&n);
                let config = self.context.launch_config_1d(n, &f);
                unsafe { build.launch(config) }.expect("Failed to launch kernel");
            }
        }
    }

    fn from_diagonal(v: &Self::V) -> Self {
        let ctx = v.context.clone();
        let nbatch = ctx.nbatch();
        let nstates = v.data.len() as IndexType / nbatch;
        let mut data = ctx
            .stream
            .alloc_zeros(nstates * nstates * nbatch)
            .expect("Failed to allocate memory for CudaMat");
        for b in 0..nbatch {
            let f = ctx.function::<T>("mat_from_diagonal");
            let n = nstates as u32;
            let v_start = b * nstates;
            let mat_start = b * nstates * nstates;
            let mut mat_slice = data.slice_mut(mat_start..mat_start + nstates * nstates);
            let v_slice = v.data.slice(v_start..v_start + nstates);
            let mut build = ctx.stream.launch_builder(&f);
            build.arg(&mut mat_slice).arg(&v_slice).arg(&n);
            let config = ctx.launch_config_1d(n, &f);
            unsafe { build.launch(config) }.expect("Failed to launch kernel");
        }
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
        for b in 0..nbatch {
            let v_b = if v_nbatch == 1 { 0 } else { b };
            let v_nstates = v.data.len() as IndexType / v_nbatch;
            assert_eq!(
                v_nstates, nrows,
                "Column length mismatch: {} != {}",
                v_nstates, nrows
            );
            let f = self.context.function::<T>("mat_set_column");
            let n = nrows as u32;
            let j_cint = j as c_int;
            let mat_start = b * nrows * self.ncols;
            let v_start = v_b * v_nstates;
            let mut mat_slice = self
                .data
                .slice_mut(mat_start..mat_start + nrows * self.ncols);
            let v_slice = v.data.slice(v_start..v_start + v_nstates);
            let mut build = self.context.stream.launch_builder(&f);
            build.arg(&mut mat_slice).arg(&v_slice).arg(&j_cint).arg(&n);
            let config = self.context.launch_config_1d(n, &f);
            unsafe { build.launch(config) }.expect("Failed to launch kernel");
        }
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
        for b in 0..nbatch {
            let x_b = if x_nbatch == 1 { 0 } else { b };
            let y_b = if y_nbatch == 1 { 0 } else { b };
            let nrows = self.nrows;
            let self_ncols = self.ncols;
            let x_ncols = x.ncols;
            let y_ncols = y.ncols;
            let self_start = b * nrows * self_ncols;
            let x_start = x_b * nrows * x_ncols;
            let y_start = y_b * nrows * y_ncols;
            let n = (nrows * self_ncols) as u32;
            let mut self_slice = self
                .data
                .slice_mut(self_start..self_start + nrows * self_ncols);
            let x_slice = x.data.slice(x_start..x_start + nrows * x_ncols);
            let y_slice = y.data.slice(y_start..y_start + nrows * y_ncols);
            let mut build = self.context.stream.launch_builder(&f);
            build
                .arg(&mut self_slice)
                .arg(&x_slice)
                .arg(&y_slice)
                .arg(&beta)
                .arg(&n);
            let config = self.context.launch_config_1d(n, &f);
            unsafe { build.launch(config) }.expect("Failed to launch kernel");
        }
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
