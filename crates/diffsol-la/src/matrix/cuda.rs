use cudarc::cublas::sys as cublas;
use cudarc::{
    cublas::{sys::cublasOperation_t, CudaBlas},
    driver::{CudaSlice, CudaView, CudaViewMut, DevicePtr, DevicePtrMut, PushKernelArg},
};
use std::ffi::c_int;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use crate::context::broadcast_batch;
use crate::{
    error::LaError, linear_solver::cuda::lu::CudaLU, matrix::default_solver::DefaultSolver,
    matrix_error, Context, CudaContext, CudaVec, CudaVecMut, CudaVecRef, IndexType, MatrixCommon,
    ScalarCuda, Scale, Vector, VectorIndex,
};

use super::{
    sparsity::{Dense, DenseRef},
    DenseMatrix, Matrix,
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

impl_matrix_common!(CudaMat<T>, CudaVec<T>, CudaContext, CudaSlice<T>);

macro_rules! impl_mul_scalar {
    ($mat_type:ty, $out:ty) => {
        impl<'a, T: ScalarCuda> Mul<Scale<T>> for $mat_type {
            type Output = $out;

            fn mul(mut self, rhs: Scale<T>) -> Self::Output {
                let f = self.context.function::<T>("vec_mul_assign_scalar");
                let nbatch = self.context.nbatch();
                let nstates = (self.nrows() * self.ncols()) as u32;
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
                let nstates = (self.nrows() * self.ncols()) as u32;
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

impl_mul_scalar!(CudaMat<T>, CudaMat<T>);
impl_mul_scalar_alloc!(&CudaMat<T>, CudaMat<T>, T);

macro_rules! impl_sub_assign {
    ($lhs:ty, $rhs:ty) => {
        impl<T: ScalarCuda> SubAssign<$rhs> for $lhs {
            fn sub_assign(&mut self, rhs: $rhs) {
                let f = self.context.function::<T>("vec_sub_assign");
                let nbatch = self.context.nbatch();
                let nstates = (self.nrows() * self.ncols()) as u32;
                let nbatch_u32 = nbatch as u32;
                let self_stride = nstates as i32;
                let rhs_nbatch = rhs.context.nbatch() as i32;
                let rhs_nstates = (rhs.nrows() * rhs.ncols()) as u32;
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
                let nstates = (self.nrows() * self.ncols()) as u32;
                let nbatch_u32 = nbatch as u32;
                let self_stride = nstates as i32;
                let rhs_nbatch = rhs.context.nbatch() as i32;
                let rhs_nstates = (rhs.nrows() * rhs.ncols()) as u32;
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

impl_add_assign!(CudaMat<T>, &CudaMat<T>);

impl_sub_assign!(CudaMat<T>, &CudaMat<T>);

macro_rules! impl_sub_lhs {
    ($lhs:ty, $rhs:ty, $out:ty) => {
        impl<T: ScalarCuda> Sub<$rhs> for $lhs {
            type Output = $out;
            fn sub(mut self, rhs: $rhs) -> Self::Output {
                // `self` is owned, so it is the destination and `rhs` broadcasts into it
                self.context
                    .assert_broadcastable_into(rhs.context.nbatch(), "sub");
                let nbatch = self.context.nbatch();
                let nstates = (self.nrows() * self.ncols()) as u32;
                let nbatch_u32 = nbatch as u32;
                let self_stride = nstates as i32;
                let rhs_nbatch = rhs.context.nbatch() as i32;
                let rhs_nstates = (rhs.nrows() * rhs.ncols()) as u32;
                let rhs_stride = rhs_nstates as i32;
                let f = self.context.function::<T>("vec_sub_assign");
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

macro_rules! impl_add_lhs {
    ($lhs:ty, $rhs:ty, $out:ty) => {
        impl<T: ScalarCuda> Add<$rhs> for $lhs {
            type Output = $out;
            fn add(mut self, rhs: $rhs) -> Self::Output {
                // `self` is owned, so it is the destination and `rhs` broadcasts into it
                self.context
                    .assert_broadcastable_into(rhs.context.nbatch(), "add");
                let nbatch = self.context.nbatch();
                let nstates = (self.nrows() * self.ncols()) as u32;
                let nbatch_u32 = nbatch as u32;
                let self_stride = nstates as i32;
                let rhs_nbatch = rhs.context.nbatch() as i32;
                let rhs_nstates = (rhs.nrows() * rhs.ncols()) as u32;
                let rhs_stride = rhs_nstates as i32;
                let f = self.context.function::<T>("vec_add_assign");
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

/// Kernel-argument counterpart of `struct GemvColsWeights_f64`, passed by value so the small
/// host-computed coefficient list reaches the device without a copy per call. cuBLAS cannot
/// serve a host-side `x` at all — its vector arguments must be device pointers.
///
/// The array length must match `MAX_SMALL_COLS` in `cuda_kernels/cuda_kernels_common.h`; nothing checks
/// that across the language boundary, so the two are kept in step by convention (as
/// `MulColsByRhs` and `mul_cols_by.cu` already are).
#[repr(C)]
#[derive(Clone, Copy)]
struct GemvColsWeights<T: ScalarCuda> {
    w: [T; crate::matrix::MAX_SMALL_COLS],
}

// SAFETY: `repr(C)` holding only `T: ScalarCuda` values, which are themselves `DeviceRepr`, so
// the layout matches the kernel's struct parameter.
unsafe impl<T: ScalarCuda> cudarc::driver::DeviceRepr for GemvColsWeights<T> {}

/// Kernel-argument counterpart of `struct MulColsByRhs_f64`, passed by value so the small
/// host-computed coefficient block reaches the device without a copy per call.
#[repr(C)]
#[derive(Clone, Copy)]
struct MulColsByRhs<T: ScalarCuda> {
    m: [T; crate::matrix::MAX_SMALL_COLS * crate::matrix::MAX_SMALL_COLS],
}

// SAFETY: `MulColsByRhs` is `repr(C)` and holds only `T: ScalarCuda` values, which are
// themselves `DeviceRepr`, so its layout matches the kernel's struct parameter.
unsafe impl<T: ScalarCuda> cudarc::driver::DeviceRepr for MulColsByRhs<T> {}

impl<T: ScalarCuda> DenseMatrix for CudaMat<T> {
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

    fn mul_cols_by(&mut self, ncols: IndexType, rhs: &[T]) {
        assert!(
            ncols <= self.ncols(),
            "mul_cols_by: column range out of bounds"
        );
        assert_eq!(
            rhs.len(),
            ncols * ncols,
            "mul_cols_by: rhs must hold ncols * ncols values"
        );
        if ncols == 0 {
            return;
        }
        assert!(
            ncols <= crate::matrix::MAX_SMALL_COLS,
            "mul_cols_by: ncols exceeds MAX_SMALL_COLS"
        );
        let nrows = self.nrows();
        let nbatch = self.context.nbatch();
        let f = self.context.function::<T>("mul_cols_by");
        let config = self
            .context
            .launch_config_2d(nrows as u32, nbatch as u32, &f);
        let mut kernel_rhs = MulColsByRhs {
            m: [T::zero(); crate::matrix::MAX_SMALL_COLS * crate::matrix::MAX_SMALL_COLS],
        };
        kernel_rhs.m[..rhs.len()].copy_from_slice(rhs);
        let ncols_i32 = ncols as i32;
        let nrows_i32 = nrows as i32;
        let mat_stride = (nrows * self.ncols) as i32;
        let mut build = self.context.stream.launch_builder(&f);
        build
            .arg(&mut self.data)
            .arg(&kernel_rhs)
            .arg(&ncols_i32)
            .arg(&nrows_i32)
            .arg(&mat_stride);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }

    fn update_backward_diff(&mut self, order: IndexType, d: &Self::V) {
        assert!(order + 2 < self.ncols(), "order out of bounds");
        self.context
            .assert_broadcastable_into(d.context.nbatch(), "update_backward_diff");

        let nbatch = self.context.nbatch();
        let nrows = self.nrows();
        let ncols = self.ncols();
        let d_nbatch = d.context.nbatch();
        let d_stride = (d.data.len() / d_nbatch) as i32;
        let f = self.context.function::<T>("backward_diff_update");
        let nrows_u32 = nrows as u32;
        let nbatch_u32 = nbatch as u32;
        let config = self.context.launch_config_2d(nrows_u32, nbatch_u32, &f);
        let mut build = self.context.stream.launch_builder(&f);
        let order_i32 = order as i32;
        let nrows_i32 = nrows as i32;
        let diff_stride = (nrows * ncols) as i32;
        let d_nbatch_i32 = d_nbatch as i32;
        build
            .arg(&mut self.data)
            .arg(&d.data)
            .arg(&order_i32)
            .arg(&nrows_i32)
            .arg(&diff_stride)
            .arg(&d_stride)
            .arg(&d_nbatch_i32);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }

    fn gemv_cols(
        &self,
        start: IndexType,
        end: IndexType,
        alpha: T,
        x: &[T],
        beta: T,
        y: &mut Self::V,
    ) {
        assert!(start <= end, "gemv_cols: column range start > end");
        assert!(end <= self.ncols, "gemv_cols: column range out of bounds");
        let nc = end - start;
        assert!(
            nc <= crate::matrix::MAX_SMALL_COLS,
            "gemv_cols: column range exceeds MAX_SMALL_COLS"
        );
        assert!(
            x.len() >= nc,
            "gemv_cols: x must hold at least end - start values"
        );
        y.context
            .assert_broadcastable_into(self.context.nbatch(), "gemv_cols");
        // an empty column range contributes nothing, leaving y = beta * y
        if nc == 0 {
            if beta.is_zero() {
                y.fill(T::zero());
            } else if !beta.is_one() {
                y.mul_assign(Scale(beta));
            }
            return;
        }
        let nrows = self.nrows();
        let y_nbatch = y.context.nbatch();
        let f = self.context.function::<T>("gemv_cols");
        let config = self
            .context
            .launch_config_2d(nrows as u32, y_nbatch as u32, &f);
        let mut weights = GemvColsWeights {
            w: [T::zero(); crate::matrix::MAX_SMALL_COLS],
        };
        weights.w[..nc].copy_from_slice(&x[..nc]);
        let nc_i32 = nc as i32;
        let start_i32 = start as i32;
        let nrows_i32 = nrows as i32;
        let y_stride = (y.data.len() / y_nbatch) as i32;
        let mat_stride = (nrows * self.ncols) as i32;
        let mat_nbatch_i32 = self.context.nbatch() as i32;
        let mut build = self.context.stream.launch_builder(&f);
        build
            .arg(&mut y.data)
            .arg(&self.data)
            .arg(&weights)
            .arg(&nc_i32)
            .arg(&alpha)
            .arg(&beta)
            .arg(&start_i32)
            .arg(&nrows_i32)
            .arg(&y_stride)
            .arg(&mat_stride)
            .arg(&mat_nbatch_i32);
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
    fn inner_mut(&mut self) -> &mut Self::Inner {
        &mut self.data
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
        let data_nstates = data.len();
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
        v.context
            .assert_broadcastable_into(nbatch, "add_column_to_vector");
        let nrows = self.nrows();
        let ncols = self.ncols();
        let v_nstates = v.len();
        let f = self.context.function::<T>("vec_axpy_offset");
        let nrows_u32 = nrows as u32;
        // `v` is the destination, so it carries the launch's batch count and the matrix
        // broadcasts over it (as on the CPU backends)
        let config = self
            .context
            .launch_config_2d(nrows_u32, v_nbatch as u32, &f);
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
    ) -> Result<Self, LaError> {
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
        let y_nbatch = y.context.nbatch();
        y.context.assert_broadcastable_into(nbatch, "gemv");
        y.context.assert_broadcastable_into(x_nbatch, "gemv");
        // `y` is the destination, so it carries the batch count of the result
        for b in 0..y_nbatch {
            let self_b = broadcast_batch(b, nbatch, y_nbatch);
            let x_b = broadcast_batch(b, x_nbatch, y_nbatch);
            let x_nstates = self.ncols;
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
            .assert_broadcastable_into(other_nbatch, "copy_from");
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
        let nstates = v.len();
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
        let diagonal = self.diagonal().clone_as_vec();
        let nstates = self.nrows();
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
            .assert_broadcastable_into(v_nbatch, "set_column");
        let nrows = self.nrows();
        let v_nstates = v.len();
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
            .assert_broadcastable_into(x_nbatch, "scale_add_and_assign_x");
        self.context
            .assert_broadcastable_into(y_nbatch, "scale_add_and_assign_y");
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

    super::super::generate_matrix_tests_nonbatched!(cuda, CudaMat<f64>);

    super::super::generate_matrix_tests_batched!(
        cuda,
        CudaMat<f64>,
        CudaContext::default(),
        CudaContext::default().with_nbatch(2)
    );

    super::super::generate_dense_matrix_tests_nonbatched!(cuda, CudaMat<f64>);

    super::super::generate_dense_matrix_tests_batched!(
        cuda,
        CudaMat<f64>,
        CudaContext::default(),
        CudaContext::default().with_nbatch(2)
    );
}
