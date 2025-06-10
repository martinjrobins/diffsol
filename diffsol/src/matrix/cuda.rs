use cudarc::cublas::sys as cublas;
use cudarc::{
    cublas::{sys::cublasOperation_t, CudaBlas},
    driver::{CudaSlice, CudaView, CudaViewMut, DevicePtr, DevicePtrMut, PushKernelArg},
};
use std::ffi::c_int;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use crate::{
    error::{DiffsolError, MatrixError},
    matrix_error, CudaContext, CudaType, CudaVec, CudaVecMut, CudaVecRef, IndexType, MatrixCommon,
    ScalarCuda, Scale, Vector, VectorIndex,
};

use super::{
    sparsity::{Dense, DenseRef},
    DenseMatrix, Matrix, MatrixView, MatrixViewMut,
};

/// triplet iterator for a dense matrix held in column-major order
struct DenseMatTripletIter<T: ScalarCuda> {
    data: Vec<T>,
    nrows: IndexType,
    ncols: IndexType,
    current_row: IndexType,
    current_col: IndexType,
}

impl<T: ScalarCuda> DenseMatTripletIter<T> {
    fn new(data: Vec<T>, nrows: IndexType, ncols: IndexType) -> Self {
        Self {
            data,
            nrows,
            ncols,
            current_row: 0,
            current_col: 0,
        }
    }
}

impl<T: ScalarCuda> Iterator for DenseMatTripletIter<T> {
    type Item = (IndexType, IndexType, T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_col >= self.ncols {
            return None;
        }
        let index = self.current_row + self.current_col * self.nrows;
        let value = self.data[index];
        let triplet = (self.current_row, self.current_col, value);
        self.current_row += 1;
        if self.current_row >= self.nrows {
            self.current_row = 0;
            self.current_col += 1;
        }
        Some(triplet)
    }
}

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
}

#[derive(Debug)]
pub struct CudaMatMut<'a, T: ScalarCuda> {
    pub(crate) data: CudaViewMut<'a, T>,
    pub(crate) context: CudaContext,
    nrows: IndexType,
    ncols: IndexType,
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
            .memcpy_dtov(&self.data.slice(index..index + 1))
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
        let mut data: CudaSlice<T> = unsafe { self.context.stream.alloc(self.nrows) }
            .expect("Failed to allocate memory for diagonal");
        let f = self.context.function::<T>("mat_get_diagonal");
        let n = self.nrows() as u32;
        let mut build = self.context.stream.launch_builder(&f);
        build.arg(&self.data).arg(&mut data).arg(&n);
        let config = self.context.launch_config_1d(n, &f);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
        CudaVec {
            data,
            context: self.context.clone(),
        }
    }
}

//impl<T: Scalar> DefaultSolver for CudaMat<T> {
//    type LS = CudaLU<T>;
//}

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
        let mut data = unsafe { self.context.stream.alloc(self.nrows * self.ncols) }
            .expect("Failed to allocate memory for CudaVec");
        self.context
            .stream
            .memcpy_dtod(&self.data, &mut data)
            .expect("Failed to copy data from device to host");
        CudaMat {
            data,
            context: self.context.clone(),
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }

    fn gemv_o(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V) {
        self.context.gemv(
            self.nrows(),
            self.ncols(),
            alpha,
            beta,
            &self.data,
            &x.data,
            &mut y.data,
        );
    }
    fn gemv_v(
        &self,
        alpha: Self::T,
        x: &<Self::V as crate::vector::Vector>::View<'_>,
        beta: Self::T,
        y: &mut Self::V,
    ) {
        self.context.gemv(
            self.nrows(),
            self.ncols(),
            alpha,
            beta,
            &self.data,
            &x.data,
            &mut y.data,
        );
    }
}

impl<'a, T: ScalarCuda> MatrixViewMut<'a> for CudaMatMut<'a, T> {
    type Owned = CudaMat<T>;
    type View = CudaMatRef<'a, T>;

    fn into_owned(self) -> Self::Owned {
        let mut data = unsafe { self.context.stream.alloc(self.nrows * self.ncols) }
            .expect("Failed to allocate memory for CudaVec");
        self.context
            .stream
            .memcpy_dtod(&self.data, &mut data)
            .expect("Failed to copy data from device to host");
        CudaMat {
            data,
            context: self.context.clone(),
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }

    fn gemm_oo(&mut self, alpha: Self::T, a: &Self::Owned, b: &Self::Owned, beta: Self::T) {
        self.context.gemm(
            a.nrows(),
            a.ncols(),
            b.nrows(),
            b.ncols(),
            self.nrows(),
            alpha,
            beta,
            &a.data,
            &b.data,
            &mut self.data,
        );
    }
    fn gemm_vo(&mut self, alpha: Self::T, a: &Self::View, b: &Self::Owned, beta: Self::T) {
        self.context.gemm(
            a.nrows(),
            a.ncols(),
            b.nrows(),
            b.ncols(),
            self.nrows(),
            alpha,
            beta,
            &a.data,
            &b.data,
            &mut self.data,
        );
    }
}

impl<T: ScalarCuda> DenseMatrix for CudaMat<T> {
    type View<'a> = CudaMatRef<'a, T>;
    type ViewMut<'a> = CudaMatMut<'a, T>;

    fn from_vec(nrows: IndexType, ncols: IndexType, data: Vec<Self::T>, ctx: Self::C) -> Self {
        assert_eq!(data.len(), nrows * ncols);
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
        self.context.gemm(
            a.nrows(),
            a.ncols(),
            b.nrows(),
            b.ncols(),
            self.nrows(),
            alpha,
            beta,
            &a.data,
            &b.data,
            &mut self.data,
        );
    }
    fn column_mut(&mut self, i: usize) -> <Self::V as Vector>::ViewMut<'_> {
        let start_i = self.col_major_index(0, i);
        let end_i = self.col_major_index(self.nrows(), i);
        let data = self.data.slice_mut(start_i..end_i);
        CudaVecMut {
            data,
            context: self.context.clone(),
        }
    }

    fn columns_mut(&mut self, start: usize, end: usize) -> Self::ViewMut<'_> {
        assert!(start < end, "Start index must be less than end index");
        assert!(end <= self.ncols(), "End index out of bounds");
        assert!(start < self.ncols(), "Start index out of bounds");
        let start_i = self.col_major_index(0, start);
        let end_i = self.col_major_index(0, end);
        let nrows = self.nrows();
        let ncols = end - start;
        let data = self.data.slice_mut(start_i..end_i);
        CudaMatMut {
            data,
            context: self.context.clone(),
            nrows,
            ncols,
        }
    }

    fn set_index(&mut self, i: IndexType, j: IndexType, value: Self::T) {
        self.set_index(i, j, value);
    }

    fn column(&self, i: usize) -> <Self::V as Vector>::View<'_> {
        let start_i = self.col_major_index(0, i);
        let end_i = self.col_major_index(self.nrows(), i);
        let data = self.data.slice(start_i..end_i);
        CudaVecRef {
            data,
            context: self.context.clone(),
        }
    }
    fn columns(&self, start: usize, end: usize) -> Self::View<'_> {
        assert!(start < end, "Start index must be less than end index");
        assert!(end <= self.ncols(), "End index out of bounds");
        assert!(start < self.ncols(), "Start index out of bounds");
        let start_i = self.col_major_index(0, start);
        let end_i = self.col_major_index(0, end);
        let data = self.data.slice(start_i..end_i);
        CudaMatRef {
            data,
            context: self.context.clone(),
            nrows: self.nrows(),
            ncols: end - start,
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

        let start_x = self.col_major_index(0, j);
        let end_x = self.col_major_index(self.nrows(), j);
        let start_y = self.col_major_index(0, i);
        let end_y = self.col_major_index(self.nrows(), i);
        // need to use unsafe api since we're accessing different columns in the same matrix
        let x = {
            let x = self.data.slice(start_x..end_x);
            let (x, _syn_x) = x.device_ptr(&self.context.stream);
            x
        };
        let y = {
            let y = self.data.slice(start_y..end_y);
            let (y, _syn_y) = y.device_ptr(&self.context.stream);
            y
        };
        self.context
            .axpy_inner::<T>(alpha, x, y, self.nrows() as c_int);
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
        let f = self.context.function::<T>("mat_set_data_with_indices");
        let n = dst_indices.len() as u32;
        let mut build = self.context.stream.launch_builder(&f);
        build
            .arg(&mut self.data)
            .arg(&data.data)
            .arg(&dst_indices.data)
            .arg(&src_indices.data)
            .arg(&n);
        let config = self.context.launch_config_1d(n, &f);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }

    fn add_column_to_vector(&self, j: IndexType, v: &mut Self::V) {
        v.add_assign(&self.column(j));
    }

    fn triplet_iter(&self) -> impl Iterator<Item = (IndexType, IndexType, Self::T)> {
        let data = self
            .context
            .stream
            .memcpy_dtov(&self.data)
            .expect("Failed to copy data from device to host");
        DenseMatTripletIter::new(data, self.nrows(), self.ncols())
    }

    fn try_from_triplets(
        nrows: IndexType,
        ncols: IndexType,
        triplets: Vec<(IndexType, IndexType, T)>,
        ctx: Self::C,
    ) -> Result<Self, DiffsolError> {
        let mut m = vec![T::zero(); nrows * ncols];
        for (i, j, v) in triplets {
            if i >= nrows || j >= ncols {
                return Err(matrix_error!(IndexOutOfBounds));
            }
            let idx = i + j * nrows;
            m[idx] = v;
        }
        Ok(Self::from_vec(nrows, ncols, m, ctx))
    }
    fn gemv(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V) {
        self.context.gemv(
            self.nrows(),
            self.ncols(),
            alpha,
            beta,
            &self.data,
            &x.data,
            &mut y.data,
        );
    }
    fn zeros(nrows: IndexType, ncols: IndexType, ctx: Self::C) -> Self {
        let data = ctx
            .stream
            .alloc_zeros(nrows * ncols)
            .expect("Failed to allocate memory for CudaMat");
        Self {
            data,
            context: ctx,
            nrows,
            ncols,
        }
    }
    fn copy_from(&mut self, other: &Self) {
        let f = self.context.function::<T>("vec_copy");
        let n = self.data.len() as u32;
        let mut build = self.context.stream.launch_builder(&f);
        build.arg(&mut self.data).arg(&other.data).arg(&n);
        let config = self.context.launch_config_1d(n, &f);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }

    fn from_diagonal(v: &Self::V) -> Self {
        let ctx = v.context.clone();
        let mut data = ctx
            .stream
            .alloc_zeros(v.len() * v.len())
            .expect("Failed to allocate memory for CudaMat");
        let f = ctx.function::<T>("mat_from_diagonal");
        let n = v.len() as u32;
        let mut build = ctx.stream.launch_builder(&f);
        build.arg(&mut data).arg(&v.data).arg(&n);
        let config = ctx.launch_config_1d(n, &f);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
        Self {
            data,
            context: ctx,
            nrows: v.len(),
            ncols: v.len(),
        }
    }

    fn partition_indices_by_zero_diagonal(
        &self,
    ) -> (<Self::V as Vector>::Index, <Self::V as Vector>::Index) {
        let diagonal = self.diagonal().clone_as_vec();
        let (zero_indices, nonzero_indices) = diagonal.iter().enumerate().fold(
            (Vec::new(), Vec::new()),
            |(mut zero_indices, mut nonzero_indices), (i, &v)| {
                if v.is_zero() {
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
        assert_eq!(
            v.len(),
            self.nrows(),
            "Column length mismatch: {} != {}",
            v.len(),
            self.nrows()
        );
        let f = self.context.function::<T>("mat_set_column");
        let n = self.nrows() as u32;
        let j = j as c_int;
        let mut build = self.context.stream.launch_builder(&f);
        build.arg(&mut self.data).arg(&v.data).arg(&j).arg(&n);
        let config = self.context.launch_config_1d(n, &f);
        unsafe { build.launch(config) }.expect("Failed to launch kernel");
    }

    /// Perform the assignment self = x + beta * y where x and y are matrices and beta is a scalar
    fn scale_add_and_assign(&mut self, x: &Self, beta: Self::T, y: &Self) {
        let f = self.context.function::<T>("mat_scale_add_assign");
        let n = (self.nrows() * self.ncols()) as u32;
        let mut build = self.context.stream.launch_builder(&f);
        build
            .arg(&mut self.data)
            .arg(&x.data)
            .arg(&y.data)
            .arg(&beta)
            .arg(&n);
        let config = self.context.launch_config_1d(n, &f);
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
    use crate::{vector::Vector, CudaIndex};

    #[test]
    fn test_cudamat_creation() {
        let ctx = CudaContext::default();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mat = CudaMat::from_vec(2, 2, data.clone(), ctx.clone());
        assert_eq!(mat.nrows(), 2);
        assert_eq!(mat.ncols(), 2);
        assert_eq!(mat.get_index(0, 0), 1.0);
        assert_eq!(mat.get_index(1, 1), 4.0);
    }

    #[test]
    fn test_cudamat_set_index() {
        let ctx = CudaContext::default();
        let mut mat = CudaMat::zeros(2, 2, ctx.clone());
        mat.set_index(0, 0, 5.0);
        assert_eq!(mat.get_index(0, 0), 5.0);
    }

    #[test]
    fn test_cudamat_diagonal() {
        let ctx = CudaContext::default();
        let data = vec![1.0, 0.0, 0.0, 2.0];
        let mat = CudaMat::from_vec(2, 2, data, ctx.clone());
        let diag = mat.diagonal();
        assert_eq!(diag.clone_as_vec(), vec![1.0, 2.0]);
    }

    #[test]
    fn test_cudamatref_gemv() {
        let ctx = CudaContext::default();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mat = CudaMat::from_vec(2, 2, data, ctx.clone());
        let x = CudaVec::from_vec(vec![1.0, 1.0], ctx.clone());
        let mut y = CudaVec::from_vec(vec![0.0, 0.0], ctx.clone());
        mat.columns(0, 2).gemv_o(1.0, &x, 0.0, &mut y);
        assert_eq!(y.clone_as_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_cudamatmut_gemm() {
        let ctx = CudaContext::default();
        let a = CudaMat::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0], ctx.clone());
        let b = CudaMat::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0], ctx.clone());
        let mut c = CudaMat::zeros(2, 2, ctx.clone());
        c.columns_mut(0, 2).gemm_oo(1.0, &a, &b, 0.0);
        // Check the result
        // [ 1, 3 ] *  [ 5, 7 ] = [ 23, 31 ]
        // [ 2, 4 ]    [ 6, 8 ]   [ 34, 46 ]
        assert_eq!(c.get_index(0, 0), 23.0);
        assert_eq!(c.get_index(1, 1), 46.0);
        assert_eq!(c.get_index(0, 1), 31.0);
        assert_eq!(c.get_index(1, 0), 34.0);
    }

    #[test]
    fn test_cudamat_try_from_triplets() {
        let ctx = CudaContext::default();
        let triplets = vec![(0, 0, 1.0), (1, 1, 2.0)];
        let mat = CudaMat::try_from_triplets(2, 2, triplets, ctx.clone()).unwrap();
        assert_eq!(mat.get_index(0, 0), 1.0);
        assert_eq!(mat.get_index(1, 1), 2.0);
    }

    #[test]
    fn test_cudamat_scale_add_and_assign() {
        let ctx = CudaContext::default();
        let mut mat = CudaMat::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0], ctx.clone());
        let x = CudaMat::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0], ctx.clone());
        let y = CudaMat::from_vec(2, 2, vec![1.0, 1.0, 1.0, 1.0], ctx.clone());
        mat.scale_add_and_assign(&x, 2.0, &y);
        // Check the result
        // [ 5, 7 ] + 2 * [ 1, 1 ] = [ 7, 9 ]
        // [ 6, 8 ]       [ 1, 1 ] = [ 8, 10 ]
        assert_eq!(mat.get_index(0, 0), 7.0);
        assert_eq!(mat.get_index(1, 1), 10.0);
        assert_eq!(mat.get_index(0, 1), 9.0);
        assert_eq!(mat.get_index(1, 0), 8.0);
    }

    #[test]
    fn test_column_axpy() {
        super::super::tests::test_column_axpy::<CudaMat<f64>>();
    }

    #[test]
    fn test_partition_indices_by_zero_diagonal() {
        super::super::tests::test_partition_indices_by_zero_diagonal::<CudaMat<f64>>();
    }

    #[test]
    fn test_cudamat_mul() {
        let ctx = CudaContext::default();
        let mat = CudaMat::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0], ctx.clone());
        let scaled_mat = mat * Scale(2.0);
        assert_eq!(scaled_mat.get_index(0, 0), 2.0);
        assert_eq!(scaled_mat.get_index(1, 1), 8.0);
    }

    #[test]
    fn test_cudamat_mul_assign() {
        let ctx = CudaContext::default();
        let mut mat = CudaMat::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0], ctx.clone());
        let mut mat_mut = mat.columns_mut(0, 2);
        mat_mut *= Scale(2.0);
        assert_eq!(mat.get_index(0, 0), 2.0);
        assert_eq!(mat.get_index(1, 1), 8.0);
    }

    #[test]
    fn test_cudamat_add() {
        let ctx = CudaContext::default();
        let mat1 = CudaMat::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0], ctx.clone());
        let mat2 = CudaMat::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0], ctx.clone());
        let result = mat1 + &mat2;
        assert_eq!(result.get_index(0, 0), 6.0);
        assert_eq!(result.get_index(1, 1), 12.0);
    }

    #[test]
    fn test_cudamat_add_assign() {
        let ctx = CudaContext::default();
        let mut mat1 = CudaMat::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0], ctx.clone());
        let mat2 = CudaMat::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0], ctx.clone());
        mat1 += &mat2;
        assert_eq!(mat1.get_index(0, 0), 6.0);
        assert_eq!(mat1.get_index(1, 1), 12.0);
    }

    #[test]
    fn test_cudamat_sub() {
        let ctx = CudaContext::default();
        let mat1 = CudaMat::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0], ctx.clone());
        let mat2 = CudaMat::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0], ctx.clone());
        let result = mat1 - &mat2;
        assert_eq!(result.get_index(0, 0), 4.0);
        assert_eq!(result.get_index(1, 1), 4.0);
    }

    #[test]
    fn test_cudamat_sub_assign() {
        let ctx = CudaContext::default();
        let mut mat1 = CudaMat::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0], ctx.clone());
        let mat2 = CudaMat::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0], ctx.clone());
        mat1 -= &mat2;
        assert_eq!(mat1.get_index(0, 0), 4.0);
        assert_eq!(mat1.get_index(1, 1), 4.0);
    }

    #[test]
    fn test_cudamat_copy_from() {
        let ctx = CudaContext::default();
        let mat1 = CudaMat::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0], ctx.clone());
        let mut mat2 = CudaMat::zeros(2, 2, ctx.clone());
        mat2.copy_from(&mat1);
        assert_eq!(mat2.get_index(0, 0), 1.0);
        assert_eq!(mat2.get_index(1, 1), 4.0);
    }

    #[test]
    fn test_cudamat_gather() {
        let ctx = CudaContext::default();
        // M = [ 1, 4, 7 ]
        //     [ 2, 5, 8 ]
        //     [ 3, 6, 9 ]
        let mat1 = CudaMat::from_vec(
            3,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            ctx.clone(),
        );
        let mut mat2 = CudaMat::zeros(2, 2, ctx.clone());
        let indices = CudaIndex::from_vec(vec![0, 1, 3, 4], ctx.clone());
        mat2.gather(&mat1, &indices);
        assert_eq!(mat2.get_index(0, 0), 1.0);
        assert_eq!(mat2.get_index(0, 1), 4.0);
        assert_eq!(mat2.get_index(1, 1), 5.0);
        assert_eq!(mat2.get_index(1, 0), 2.0);
    }

    #[test]
    fn test_cudamat_set_data_with_indices() {
        let ctx = CudaContext::default();
        let mut mat = CudaMat::zeros(2, 2, ctx.clone());
        let dst_indices = CudaIndex::from_vec(vec![0, 3], ctx.clone());
        let src_indices = CudaIndex::from_vec(vec![0, 1], ctx.clone());
        let data = CudaVec::from_vec(vec![5.0, 6.0], ctx.clone());
        mat.set_data_with_indices(&dst_indices, &src_indices, &data);
        assert_eq!(mat.get_index(0, 0), 5.0);
        assert_eq!(mat.get_index(1, 1), 6.0);
    }

    #[test]
    fn test_cudamat_add_column_to_vector() {
        let ctx = CudaContext::default();
        let mat = CudaMat::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0], ctx.clone());
        let mut vec = CudaVec::from_vec(vec![0.0, 0.0], ctx.clone());
        mat.add_column_to_vector(1, &mut vec);
        assert_eq!(vec.clone_as_vec(), vec![3.0, 4.0]);
    }

    #[test]
    fn test_cudamat_set_column() {
        let ctx = CudaContext::default();
        let mut mat = CudaMat::zeros(2, 2, ctx.clone());
        let vec = CudaVec::from_vec(vec![5.0, 6.0], ctx.clone());
        mat.set_column(1, &vec);
        assert_eq!(mat.get_index(0, 1), 5.0);
        assert_eq!(mat.get_index(1, 1), 6.0);
    }

    #[test]
    fn test_cudamat_into_owned() {
        let ctx = CudaContext::default();
        let mut mat = CudaMat::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0], ctx.clone());
        let mat_ref = mat.columns(0, 2);
        assert_eq!(mat_ref.nrows(), 2);
        assert_eq!(mat_ref.ncols(), 2);
        let mat_copy = mat_ref.into_owned();
        assert_eq!(mat_copy.nrows(), 2);
        assert_eq!(mat_copy.ncols(), 2);
        assert_eq!(mat_copy.get_index(0, 0), 1.0);
        assert_eq!(mat_copy.get_index(1, 1), 4.0);
        let mat_ref = mat.columns_mut(0, 2);
        assert_eq!(mat_ref.nrows(), 2);
        assert_eq!(mat_ref.ncols(), 2);
        let mat_copy = mat_ref.into_owned();
        assert_eq!(mat_copy.nrows(), 2);
        assert_eq!(mat_copy.ncols(), 2);
        assert_eq!(mat_copy.get_index(0, 0), 1.0);
        assert_eq!(mat_copy.get_index(1, 1), 4.0);

        // Check that the original matrix is unchanged
        assert_eq!(mat.nrows(), 2);
        assert_eq!(mat.ncols(), 2);
        assert_eq!(mat.get_index(0, 0), 1.0);
        assert_eq!(mat.get_index(1, 1), 4.0);
    }

    #[test]
    fn test_cudamat_columns() {
        let ctx = CudaContext::default();
        let mut mat = CudaMat::from_vec(
            2,
            4,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            ctx.clone(),
        );
        let cols = mat.columns(1, 3);
        assert_eq!(cols.nrows(), 2);
        assert_eq!(cols.ncols(), 2);
        let cols = cols.into_owned();
        assert_eq!(cols.get_index(0, 0), 3.0);
        assert_eq!(cols.get_index(1, 1), 6.0);
        let cols = mat.columns_mut(1, 3);
        assert_eq!(cols.nrows(), 2);
        assert_eq!(cols.ncols(), 2);
        let cols = cols.into_owned();
        assert_eq!(cols.get_index(0, 0), 3.0);
        assert_eq!(cols.get_index(1, 1), 6.0);
    }
}
