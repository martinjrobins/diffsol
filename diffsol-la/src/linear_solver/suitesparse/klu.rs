use std::cell::RefCell;

#[cfg(target_pointer_width = "32")]
use suitesparse_sys::{
    klu_analyze, klu_common, klu_defaults, klu_factor, klu_free_numeric, klu_free_symbolic,
    klu_numeric, klu_solve, klu_symbolic,
};

#[cfg(target_pointer_width = "32")]
type KluIndextype = i32;

#[cfg(target_pointer_width = "64")]
use suitesparse_sys::{
    klu_l_analyze as klu_analyze, klu_l_common as klu_common, klu_l_defaults as klu_defaults,
    klu_l_factor as klu_factor, klu_l_free_numeric as klu_free_numeric,
    klu_l_free_symbolic as klu_free_symbolic, klu_l_numeric as klu_numeric,
    klu_l_solve as klu_solve, klu_l_symbolic as klu_symbolic,
};

#[cfg(target_pointer_width = "64")]
type KluIndextype = i64;

use crate::{
    error::LaError, linear_solver::LinearSolver, linear_solver_error, vector::Vector,
    FaerSparseMat, FaerVec, LinearOp, Matrix,
};

trait MatrixKLU: Matrix<T = f64> {
    fn column_pointers(&self) -> *const KluIndextype;
    fn row_indices(&self) -> *const KluIndextype;
    fn values_ptr(&mut self) -> *mut f64;
}

impl MatrixKLU for FaerSparseMat<f64> {
    fn column_pointers(&self) -> *const KluIndextype {
        self.data.symbolic().col_ptr().as_ptr() as *const KluIndextype
    }

    fn row_indices(&self) -> *const KluIndextype {
        self.data.symbolic().row_idx().as_ptr() as *const KluIndextype
    }

    fn values_ptr(&mut self) -> *mut f64 {
        self.data.val_mut().as_mut_ptr()
    }
}

trait VectorKLU: Vector {
    fn values_mut_ptr(&mut self) -> *mut f64;
}

impl VectorKLU for FaerVec<f64> {
    fn values_mut_ptr(&mut self) -> *mut f64 {
        self.data.as_mut().as_ptr_mut()
    }
}

struct KluSymbolic {
    inner: *mut klu_symbolic,
    common: *mut klu_common,
}

impl KluSymbolic {
    fn try_from_matrix(mat: &impl MatrixKLU, common: *mut klu_common) -> Result<Self, LaError> {
        let n = mat.nrows() as i64;
        let inner = unsafe {
            klu_analyze(
                n,
                mat.column_pointers() as *mut KluIndextype,
                mat.row_indices() as *mut KluIndextype,
                common,
            )
        };
        if inner.is_null() {
            return Err(linear_solver_error!(KluFailedToAnalyze));
        };
        Ok(Self { inner, common })
    }
}

impl Drop for KluSymbolic {
    fn drop(&mut self) {
        unsafe {
            klu_free_symbolic(&mut self.inner, self.common);
        }
    }
}

struct KluNumeric {
    inner: *mut klu_numeric,
    common: *mut klu_common,
}

impl KluNumeric {
    fn try_from_raw(
        symbolic: &mut KluSymbolic,
        col_ptrs: *mut KluIndextype,
        row_indices: *mut KluIndextype,
        values: *mut f64,
    ) -> Result<Self, LaError> {
        let inner = unsafe {
            klu_factor(
                col_ptrs,
                row_indices,
                values,
                symbolic.inner,
                symbolic.common,
            )
        };
        if inner.is_null() {
            return Err(linear_solver_error!(KluFailedToFactorize));
        };
        Ok(Self {
            inner,
            common: symbolic.common,
        })
    }
}

impl Drop for KluNumeric {
    fn drop(&mut self) {
        unsafe {
            klu_free_numeric(&mut self.inner, self.common);
        }
    }
}

#[derive(Clone)]
struct KluCommon {
    inner: klu_common,
}

impl Default for KluCommon {
    fn default() -> Self {
        let mut inner = klu_common::default();
        unsafe { klu_defaults(&mut inner) };
        Self { inner }
    }
}

impl KluCommon {
    fn as_mut(&mut self) -> *mut klu_common {
        &mut self.inner
    }
}

pub struct KLU<M>
where
    M: Matrix,
{
    klu_common: RefCell<KluCommon>,
    klu_symbolic: Option<KluSymbolic>,
    klu_numeric: Option<KluNumeric>,
    matrix: Option<M>,
}

impl<M> Default for KLU<M>
where
    M: Matrix,
{
    fn default() -> Self {
        let klu_common = KluCommon::default();
        let klu_common = RefCell::new(klu_common);
        Self {
            klu_common,
            klu_numeric: None,
            klu_symbolic: None,
            matrix: None,
        }
    }
}

impl<M> LinearSolver<M> for KLU<M>
where
    M: MatrixKLU,
    M::V: VectorKLU,
{
    fn set_linearisation<C: LinearOp<T = M::T, V = M::V, M = M, C = M::C>>(&mut self, op: &C) {
        let matrix = self.matrix.as_mut().expect("Matrix not set");
        op.matrix_inplace(matrix);
        let col_ptrs = matrix.column_pointers() as *mut KluIndextype;
        let row_indices = matrix.row_indices() as *mut KluIndextype;
        let values = matrix.values_ptr();
        self.klu_numeric = Some(
            KluNumeric::try_from_raw(
                self.klu_symbolic.as_mut().expect("Symbolic not set"),
                col_ptrs,
                row_indices,
                values,
            )
            .expect("Failed to factorise matrix"),
        );
    }

    fn solve_in_place(&self, x: &mut M::V) -> Result<(), LaError> {
        if self.klu_numeric.is_none() {
            return Err(linear_solver_error!(LuNotInitialized));
        }
        let klu_numeric = self.klu_numeric.as_ref().unwrap();
        let klu_symbolic = self.klu_symbolic.as_ref().unwrap();
        let n = self.matrix.as_ref().unwrap().nrows() as KluIndextype;
        let mut klu_common = self.klu_common.borrow_mut();
        let x_ptr = x.values_mut_ptr();
        unsafe {
            klu_solve(
                klu_symbolic.inner,
                klu_numeric.inner,
                n,
                1,
                x_ptr,
                klu_common.as_mut(),
            )
        };
        Ok(())
    }

    fn set_sparsity<C: LinearOp<T = M::T, V = M::V, M = M, C = M::C>>(&mut self, op: &C) {
        let ncols = op.ncols();
        let nrows = op.nrows();
        let matrix = C::M::new_from_sparsity(nrows, ncols, op.sparsity(), op.context().clone());
        let mut klu_common = self.klu_common.borrow_mut();
        self.klu_symbolic = KluSymbolic::try_from_matrix(&matrix, klu_common.as_mut()).ok();
        self.matrix = Some(matrix);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FaerSparseMat, LinearSolver, Vector};

    #[test]
    fn test_klu_identity() {
        let mut s = KLU::<FaerSparseMat<f64>>::default();
        let op = crate::linear_solver::tests::diagonal_op::<FaerSparseMat<f64>>(2.0);
        s.set_sparsity(&op);
        s.set_linearisation(&op);
        let b = FaerVec::from_vec(vec![2.0, 4.0], Default::default());
        let x = s.solve(&b).unwrap();
        x.assert_eq_st(
            &FaerVec::from_vec(vec![1.0, 2.0], Default::default()),
            1e-10,
        );
    }
}
