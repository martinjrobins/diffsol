use cudarc::{cublas::CudaBlas, driver::{CudaSlice, DevicePtr, DevicePtrMut}};
use cudarc::cublas::sys::lib as cublas;
use std::ops::{Add, AddAssign, Index, Mul, MulAssign, Sub, SubAssign};
use std::ffi::c_int;

use crate::{CudaContext, CudaType, CudaVec, IndexType, MatrixCommon, ScalarCuda, Scale};

use super::MatrixView;

/// a CUDA matrix in column-major order
#[derive(Clone, Debug, PartialEq)]
pub struct CudaMat<T: ScalarCuda> {
    pub(crate) data: CudaSlice<T>,
    pub(crate) context: CudaContext,
    nrows: IndexType,
    ncols: IndexType,
}


#[derive(Clone, Debug, PartialEq)]
pub struct CudaMatRef<'a, T: ScalarCuda>(&'a CudaMat<T>);

#[derive(Debug, PartialEq)]
pub struct CudaMatMut<'a, T: ScalarCuda>(&'a mut CudaMat<T>);

impl<T: ScalarCuda> CudaMat<T> {
    fn new(nrows: IndexType, ncols: IndexType, data: CudaSlice<T>, context: CudaContext) -> Self {
        assert_eq!(data.len(), nrows * ncols);
        Self { data, context, nrows, ncols }
    }
    fn nrows(&self) -> IndexType {
        self.data.nrows()
    }
    fn ncols(&self) -> IndexType {
        self.data.ncols()
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
                self.data.nrows()
            }
            fn ncols(&self) -> IndexType {
                self.data.ncols()
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
                self.data.nrows()
            }
            fn ncols(&self) -> IndexType {
                self.data.ncols()
            }
            fn inner(&self) -> &Self::Inner {
                &self.data
            }
        }
    };
}

impl_matrix_common_ref!(CudaMatMut<'a, T>, CudaVec<T>, CudaContext, CudaMatMut<'a, T>);
impl_matrix_common_ref!(CudaMatRef<'a, T>, CudaVec<T>, CudaContext, CudaMatRef<'a, T>);
impl_matrix_common!(CudaMat<T>, CudaVec<T>, CudaContext, CudaMat<T>);

macro_rules! impl_mul_scalar {
    ($mat_type:ty, $out:ty) => {
        impl<'a, T: ScalarCuda> Mul<Scale<T>> for $mat_type {
            type Output = $out;

            fn mul(self, rhs: Scale<T>) -> Self::Output {
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

impl_mul_assign_scalar!(CudaMatMut<'_, T>);

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

impl_add_assign!(CudaMat<T>, &CudaMat<T>);
impl_add_assign!(CudaMat<T>, &CudaMatRef<'_, T>);
impl_add_assign!(CudaMatMut<'_, T>, &CudaMatRef<'_, T>);
impl_add_assign!(CudaMatMut<'_, T>, &CudaMatMut<'_, T>);

impl_sub_assign!(CudaMat<T>, &CudaMat<T>);
impl_sub_assign!(CudaMat<T>, &CudaMatRef<'_, T>);
impl_sub_assign!(CudaMatMut<'_, T>, &CudaMatRef<'_, T>);
impl_sub_assign!(CudaMatMut<'_, T>, &CudaMatMut<'_, T>);

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

impl_add_lhs!(CudaMat<T>, &CudaMat<T>, CudaMat<T>);
impl_add_lhs!(CudaMat<T>, &CudaMatRef<'_, T>, CudaMat<T>);
impl_add_both_ref!(CudaMatRef<'_, T>, &CudaMat<T>, CudaMat<T>);


impl<'a, T: ScalarCuda> MatrixView<'a> for CudaMatRef<'a, T> {
    type Owned = CudaMat<T>;

    Source§
unsafe fn gemv<A: DevicePtr<f32>, X: DevicePtr<f32>, Y: DevicePtrMut<f32>>(
    &self,
    cfg: GemvConfig<f32>,
    a: &A,
    x: &X,
    y: &mut Y,
) -> Result<(), CublasError>

    fn gemv_o(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V) {
        gemv

        y.mul_assign(Scale(beta));
        matmul(
            y.data.as_mut(),
            Accum::Add,
            self.data.as_ref(),
            x.data.as_ref(),
            alpha,
            get_global_parallelism(),
        );
    }
    fn gemv_v(
        &self,
        alpha: Self::T,
        x: &<Self::V as crate::vector::Vector>::View<'_>,
        beta: Self::T,
        y: &mut Self::V,
    ) {
        y.mul_assign(Scale(beta));
        matmul(
            y.data.as_mut(),
            Accum::Add,
            self.data.as_ref(),
            x.data.as_ref(),
            alpha,
            get_global_parallelism(),
        );
    }
}

impl<'a, T: Scalar> MatrixViewMut<'a> for CudaMatMut<'a, T> {
    type Owned = CudaMat<T>;
    type View = CudaMatRef<'a, T>;

    fn gemm_oo(&mut self, alpha: Self::T, a: &Self::Owned, b: &Self::Owned, beta: Self::T) {
        self.mul_assign(Scale(beta));
        matmul(
            self.data.as_mut(),
            Accum::Add,
            a.data.as_ref(),
            b.data.as_ref(),
            alpha,
            get_global_parallelism(),
        )
    }
    fn gemm_vo(&mut self, alpha: Self::T, a: &Self::View, b: &Self::Owned, beta: Self::T) {
        self.mul_assign(Scale(beta));
        matmul(
            self.data.as_mut(),
            Accum::Add,
            a.data.as_ref(),
            b.data.as_ref(),
            alpha,
            get_global_parallelism(),
        )
    }
}

impl<T: Scalar> DenseMatrix for CudaMat<T> {
    type View<'a> = CudaMatRef<'a, T>;
    type ViewMut<'a> = CudaMatMut<'a, T>;

    fn from_vec(nrows: IndexType, ncols: IndexType, data: Vec<Self::T>, ctx: Self::C) -> Self {
        let data = Mat::from_fn(nrows, ncols, |i, j| data[i + j * nrows]);
        Self { data, context: ctx }
    }

    fn get_index(&self, i: IndexType, j: IndexType) -> Self::T {
        self.data[(i, j)]
    }

    fn gemm(&mut self, alpha: Self::T, a: &Self, b: &Self, beta: Self::T) {
        self.data.mul_assign(faer::Scale(beta));
        matmul(
            self.data.as_mut(),
            Accum::Add,
            a.data.as_ref(),
            b.data.as_ref(),
            alpha,
            get_global_parallelism(),
        )
    }
    fn column_mut(&mut self, i: usize) -> <Self::V as Vector>::ViewMut<'_> {
        let data = self.data.get_mut(0..self.nrows(), i);
        CudaVecMut {
            data,
            context: self.context.clone(),
        }
    }

    fn columns_mut(&mut self, start: usize, ncols: usize) -> Self::ViewMut<'_> {
        let data = self.data.get_mut(0..self.data.nrows(), start..ncols);
        CudaMatMut {
            data,
            context: self.context.clone(),
        }
    }

    fn set_index(&mut self, i: IndexType, j: IndexType, value: Self::T) {
        self.data[(i, j)] = value;
    }

    fn column(&self, i: usize) -> <Self::V as Vector>::View<'_> {
        let data = self.data.get(0..self.data.nrows(), i);
        CudaVecRef {
            data,
            context: self.context.clone(),
        }
    }
    fn columns(&self, start: usize, nrows: usize) -> Self::View<'_> {
        let data = self.data.get(0..self.nrows(), start..nrows);
        CudaMatRef {
            data,
            context: self.context.clone(),
        }
    }

    fn column_axpy(&mut self, alpha: Self::T, j: IndexType, beta: Self::T, i: IndexType) {
        if i > self.ncols() {
            panic!("Column index out of bounds");
        }
        if j > self.ncols() {
            panic!("Column index out of bounds");
        }
        if i == j {
            panic!("Column index cannot be the same");
        }
        for k in 0..self.nrows() {
            let value = unsafe {
                beta * *self.data.get_unchecked(k, i) + alpha * *self.data.get_unchecked(k, j)
            };
            unsafe { *self.data.get_mut_unchecked(k, i) = value };
        }
    }
}

impl<T: Scalar> Matrix for CudaMat<T> {
    type Sparsity = Dense<Self>;
    type SparsityRef<'a> = DenseRef<'a, Self>;

    fn sparsity(&self) -> Option<Self::SparsityRef<'_>> {
        None
    }

    fn context(&self) -> &Self::C {
        &self.context
    }

    fn gather(&mut self, other: &Self, indices: &<Self::V as Vector>::Index) {
        assert_eq!(indices.len(), self.nrows() * self.ncols());
        let mut idx = indices.data.iter().peekable();
        for j in 0..self.ncols() {
            let other_col = other.data.col(*idx.peek().unwrap() / other.nrows());
            for self_ij in self.data.col_mut(j).iter_mut() {
                let other_i = idx.next().unwrap() % other.nrows();
                *self_ij = other_col[other_i];
            }
        }
    }

    fn set_data_with_indices(
        &mut self,
        dst_indices: &<Self::V as Vector>::Index,
        src_indices: &<Self::V as Vector>::Index,
        data: &Self::V,
    ) {
        for (dst_i, src_i) in dst_indices.data.iter().zip(src_indices.data.iter()) {
            let i = dst_i % self.nrows();
            let j = dst_i / self.nrows();
            self.data[(i, j)] = data[*src_i];
        }
    }

    fn add_column_to_vector(&self, j: IndexType, v: &mut Self::V) {
        v.add_assign(&self.column(j));
    }

    fn triplet_iter(&self) -> impl Iterator<Item = (IndexType, IndexType, &Self::T)> {
        (0..self.ncols())
            .flat_map(move |j| (0..self.nrows()).map(move |i| (i, j, &self.data[(i, j)])))
    }

    fn try_from_triplets(
        nrows: IndexType,
        ncols: IndexType,
        triplets: Vec<(IndexType, IndexType, T)>,
        ctx: Self::C,
    ) -> Result<Self, DiffsolError> {
        let mut m = Mat::zeros(nrows, ncols);
        for (i, j, v) in triplets {
            m[(i, j)] = v;
        }
        Ok(Self {
            data: m,
            context: ctx,
        })
    }
    fn gemv(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V) {
        y.mul_assign(Scale(beta));
        matmul(
            y.data.as_mut(),
            Accum::Add,
            self.data.as_ref(),
            x.data.as_ref(),
            alpha,
            get_global_parallelism(),
        );
    }
    fn zeros(nrows: IndexType, ncols: IndexType, ctx: Self::C) -> Self {
        let data = Mat::zeros(nrows, ncols);
        Self { data, context: ctx }
    }
    fn copy_from(&mut self, other: &Self) {
        self.data.copy_from(&other.data);
    }
    fn from_diagonal(v: &Self::V) -> Self {
        let dim = v.len();
        let data = Mat::from_fn(dim, dim, |i, j| if i == j { v[i] } else { T::zero() });
        Self {
            data,
            context: v.context().clone(),
        }
    }
    fn partition_indices_by_zero_diagonal(
        &self,
    ) -> (<Self::V as Vector>::Index, <Self::V as Vector>::Index) {
        let diagonal = self.data.diagonal().column_vector();
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
        self.column_mut(j).copy_from(v);
    }

    fn scale_add_and_assign(&mut self, x: &Self, beta: Self::T, y: &Self) {
        zip!(self.data.as_mut(), x.data.as_ref(), y.data.as_ref())
            .for_each(|unzip!(s, x, y)| *s = *x + beta * *y);
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

    #[test]
    fn test_column_axpy() {
        // M = [1 2]
        //     [3 4]
        let mut a = CudaMat::zeros(2, 2, Default::default());
        a.set_index(0, 0, 1.0);
        a.set_index(0, 1, 2.0);
        a.set_index(1, 0, 3.0);
        a.set_index(1, 1, 4.0);

        // op is M(:, 1) = 2 * M(:, 0) + M(:, 1)
        a.column_axpy(2.0, 0, 1.0, 1);
        // M = [1 4]
        //     [3 10]
        assert_eq!(a.get_index(0, 0), 1.0);
        assert_eq!(a.get_index(0, 1), 4.0);
        assert_eq!(a.get_index(1, 0), 3.0);
        assert_eq!(a.get_index(1, 1), 10.0);
    }

    #[test]
    fn test_partition_indices_by_zero_diagonal() {
        super::super::tests::test_partition_indices_by_zero_diagonal::<CudaMat<f64>>();
    }
}
