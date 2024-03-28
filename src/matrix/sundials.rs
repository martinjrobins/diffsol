use std::{
    cmp::min,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
};

use sundials_sys::{
    realtype, SUNDenseMatrix, SUNDenseMatrix_Cols, SUNDenseMatrix_Columns, SUNDenseMatrix_Rows,
    SUNMatClone, SUNMatCopy, SUNMatDestroy, SUNMatScaleAdd, SUNMatZero, SUNMatrix,
};

use crate::{
    ode_solver::sundials::sundials_check,
    vector::sundials::{get_suncontext, SundialsVector},
    IndexType, Vector,
};

use super::{Matrix, MatrixCommon};
use anyhow::anyhow;

#[derive(Debug)]
pub struct SundialsMatrix {
    sm: SUNMatrix,
    owned: bool,
}

impl SundialsMatrix {
    pub fn new_dense(m: IndexType, n: IndexType) -> Self {
        let ctx = get_suncontext();
        let sm = unsafe { SUNDenseMatrix(m.try_into().unwrap(), n.try_into().unwrap(), *ctx) };
        SundialsMatrix { sm, owned: true }
    }
    pub fn new_not_owned(sm: SUNMatrix) -> Self {
        SundialsMatrix { sm, owned: false }
    }
    pub fn new_clone(v: &Self) -> Self {
        let sm = unsafe { SUNMatClone(v.sm) };
        SundialsMatrix { sm, owned: true }
    }
    pub fn sundials_matrix(&self) -> SUNMatrix {
        self.sm
    }

    fn map_inplace<F>(&mut self, f: F)
    where
        F: Fn(realtype) -> realtype,
    {
        let n = self.ncols();
        let m = self.nrows();
        for i in 0..m {
            for j in 0..n {
                self[(i, j)] = f(self[(i, j)]);
            }
        }
    }
}

impl Drop for SundialsMatrix {
    fn drop(&mut self) {
        if self.owned {
            unsafe { SUNMatDestroy(self.sm) };
        }
    }
}

impl MatrixCommon for SundialsMatrix {
    type V = SundialsVector;
    type T = realtype;

    fn nrows(&self) -> crate::IndexType {
        unsafe { SUNDenseMatrix_Rows(self.sm).try_into().unwrap() }
    }

    fn ncols(&self) -> crate::IndexType {
        unsafe { SUNDenseMatrix_Columns(self.sm).try_into().unwrap() }
    }
}

// index
impl Index<(IndexType, IndexType)> for SundialsMatrix {
    type Output = realtype;

    fn index(&self, index: (IndexType, IndexType)) -> &Self::Output {
        let (i, j) = index;
        let n = self.ncols();
        let m = self.nrows();
        if i >= m || j >= n {
            panic!("Index out of bounds");
        }
        unsafe { &*(*SUNDenseMatrix_Cols(self.sm).add(j)).add(i) }
    }
}

// index_mut
impl IndexMut<(IndexType, IndexType)> for SundialsMatrix {
    fn index_mut(&mut self, index: (IndexType, IndexType)) -> &mut Self::Output {
        let (i, j) = index;
        let n = self.ncols();
        let m = self.nrows();
        if i >= m || j >= n {
            panic!("Index out of bounds");
        }
        unsafe { &mut *(*SUNDenseMatrix_Cols(self.sm).add(j)).add(i) }
    }
}

// clone
impl Clone for SundialsMatrix {
    fn clone(&self) -> Self {
        let ret = SundialsMatrix::new_clone(self);
        sundials_check(unsafe { SUNMatCopy(self.sm, ret.sm) }).unwrap();
        ret
    }
}

// add assign and subtract assign
impl AddAssign<&SundialsMatrix> for SundialsMatrix {
    fn add_assign(&mut self, rhs: &SundialsMatrix) {
        sundials_check(unsafe { SUNMatScaleAdd(1.0, self.sm, rhs.sm) }).unwrap();
    }
}

impl SubAssign<&SundialsMatrix> for SundialsMatrix {
    fn sub_assign(&mut self, rhs: &SundialsMatrix) {
        sundials_check(unsafe { SUNMatScaleAdd(-1.0, self.sm, rhs.sm) }).unwrap();
        self.mul_assign(-1.0);
    }
}

// add and subtract
// create a macro for both add and subtract
macro_rules! impl_bin_op {
    ($trait:ident, $fn:ident, $op:tt) => {
        impl $trait<&SundialsMatrix> for SundialsMatrix {
            type Output = SundialsMatrix;

            fn $fn(mut self, rhs: &SundialsMatrix) -> Self::Output {
                self $op rhs;
                self
            }
        }

        impl $trait<SundialsMatrix> for SundialsMatrix {
            type Output = SundialsMatrix;

            fn $fn(mut self, rhs: SundialsMatrix) -> Self::Output {
                self $op &rhs;
                self
            }
        }

        impl $trait<SundialsMatrix> for &SundialsMatrix {
            type Output = SundialsMatrix;

            fn $fn(self, mut rhs: SundialsMatrix) -> Self::Output {
                rhs $op self;
                rhs
            }
        }

        impl $trait<&SundialsMatrix> for &SundialsMatrix {
            type Output = SundialsMatrix;

            fn $fn(self, rhs: &SundialsMatrix) -> Self::Output {
                let mut m = SundialsMatrix::new_clone(self);
                m $op rhs;
                m
            }
        }
    };
}

impl_bin_op!(Add, add, +=);
impl_bin_op!(Sub, sub, -=);

// mul and div by scalar
macro_rules! impl_scalar_op {
    ($trait:ident, $fn:ident, $op:tt) => {
        impl $trait<realtype> for SundialsMatrix {
            type Output = SundialsMatrix;

            fn $fn(mut self, rhs: realtype) -> Self::Output {
                self.map_inplace(|x| x $op rhs);
                self
            }
        }

        impl $trait<realtype> for &SundialsMatrix {
            type Output = SundialsMatrix;

            fn $fn(self, rhs: realtype) -> Self::Output {
                let mut m = SundialsMatrix::new_clone(self);
                m.map_inplace(|x| x $op rhs);
                m
            }
        }
    };
}

impl_scalar_op!(Mul, mul, *);
impl_scalar_op!(Div, div, /);

macro_rules! impl_scalar_assign_op {
    ($trait:ident, $fn:ident, $op:tt) => {
        impl $trait<realtype> for SundialsMatrix {
            fn $fn(&mut self, rhs: realtype) {
                self.map_inplace(|x| x $op rhs);
            }
        }
    }
}

impl_scalar_assign_op!(MulAssign, mul_assign, *);
impl_scalar_assign_op!(DivAssign, div_assign, /);

impl Matrix for SundialsMatrix {
    fn diagonal(&self) -> Self::V {
        let n = min(self.nrows(), self.ncols());
        let mut v = SundialsVector::new_serial(n);
        for i in 0..n {
            v[i] = self[(i, i)];
        }
        v
    }

    fn copy_from(&mut self, other: &Self) {
        let ret = unsafe { SUNMatCopy(other.sm, self.sm) };
        if ret != 0 {
            panic!("Error copying matrix");
        }
    }

    fn zeros(nrows: IndexType, ncols: IndexType) -> Self {
        let m = SundialsMatrix::new_dense(nrows, ncols);
        unsafe { SUNMatZero(m.sm) };
        m
    }

    fn from_diagonal(v: &Self::V) -> Self {
        let n = v.len();
        let mut m = SundialsMatrix::new_dense(n, n);
        for i in 0..n {
            m[(i, i)] = v[i];
        }
        m
    }

    fn try_from_triplets(
        nrows: crate::IndexType,
        ncols: crate::IndexType,
        triplets: Vec<(crate::IndexType, crate::IndexType, Self::T)>,
    ) -> anyhow::Result<Self> {
        let mut m = Self::zeros(nrows, ncols);
        for (i, j, val) in triplets {
            if i >= nrows || j >= ncols {
                return Err(anyhow!("Index out of bounds"));
            }
            m[(i, j)] = val;
        }
        Ok(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indexing() {
        let mut m = SundialsMatrix::new_dense(2, 2);
        m[(0, 0)] = 1.0;
        m[(0, 1)] = 2.0;
        m[(1, 0)] = 3.0;
        m[(1, 1)] = 4.0;
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(0, 1)], 2.0);
        assert_eq!(m[(1, 0)], 3.0);
        assert_eq!(m[(1, 1)], 4.0);
    }
}
