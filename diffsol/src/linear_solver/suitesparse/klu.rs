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
    error::{DiffsolError, LinearSolverError},
    linear_solver::LinearSolver,
    linear_solver_error,
    matrix::MatrixCommon,
    vector::Vector,
    FaerSparseMat, FaerVec, Matrix, NonLinearOpJacobian,
};

trait MatrixKLU: Matrix<T = f64> {
    fn column_pointers_mut_ptr(&mut self) -> *mut KluIndextype;
    fn row_indices_mut_ptr(&mut self) -> *mut KluIndextype;
    fn values_mut_ptr(&mut self) -> *mut f64;
}

impl MatrixKLU for FaerSparseMat<f64> {
    fn column_pointers_mut_ptr(&mut self) -> *mut KluIndextype {
        let ptrs = self.data.symbolic().col_ptr();
        ptrs.as_ptr() as *mut KluIndextype
    }

    fn row_indices_mut_ptr(&mut self) -> *mut KluIndextype {
        let indices = self.data.symbolic().row_idx();
        indices.as_ptr() as *mut KluIndextype
    }

    fn values_mut_ptr(&mut self) -> *mut f64 {
        let values = self.data.val();
        values.as_ptr() as *mut f64
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
    fn try_from_matrix(
        mat: &mut impl MatrixKLU,
        common: *mut klu_common,
    ) -> Result<Self, DiffsolError> {
        let n = mat.nrows() as i64;
        let inner = unsafe {
            klu_analyze(
                n,
                mat.column_pointers_mut_ptr(),
                mat.row_indices_mut_ptr(),
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
    fn try_from_symbolic(
        symbolic: &mut KluSymbolic,
        mat: &mut impl MatrixKLU,
    ) -> Result<Self, DiffsolError> {
        // TODO: there is also klu_refactor which is faster and reuses inner
        let inner = unsafe {
            klu_factor(
                mat.column_pointers_mut_ptr(),
                mat.row_indices_mut_ptr(),
                mat.values_mut_ptr(),
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
    fn set_linearisation<C: NonLinearOpJacobian<T = M::T, V = M::V, M = M>>(
        &mut self,
        op: &C,
        x: &M::V,
        t: M::T,
    ) {
        let matrix = self.matrix.as_mut().expect("Matrix not set");
        op.jacobian_inplace(x, t, matrix);
        self.klu_numeric = KluNumeric::try_from_symbolic(
            self.klu_symbolic.as_mut().expect("Symbolic not set"),
            matrix,
        )
        .ok();
    }

    fn solve_in_place(&self, x: &mut M::V) -> Result<(), DiffsolError> {
        if self.klu_numeric.is_none() {
            return Err(linear_solver_error!(LuNotInitialized));
        }
        let klu_numeric = self.klu_numeric.as_ref().unwrap();
        let klu_symbolic = self.klu_symbolic.as_ref().unwrap();
        let n = self.matrix.as_ref().unwrap().nrows() as KluIndextype;
        let mut klu_common = self.klu_common.borrow_mut();
        unsafe {
            klu_solve(
                klu_symbolic.inner,
                klu_numeric.inner,
                n,
                1,
                x.values_mut_ptr(),
                klu_common.as_mut(),
            )
        };
        Ok(())
    }

    fn set_problem<C: NonLinearOpJacobian<T = M::T, V = M::V, M = M, C = M::C>>(&mut self, op: &C) {
        let ncols = op.nstates();
        let nrows = op.nout();
        let mut matrix =
            C::M::new_from_sparsity(nrows, ncols, op.jacobian_sparsity(), op.context().clone());
        let mut klu_common = self.klu_common.borrow_mut();
        self.klu_symbolic = KluSymbolic::try_from_matrix(&mut matrix, klu_common.as_mut()).ok();
        self.matrix = Some(matrix);
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        linear_solver::tests::{linear_problem, test_linear_solver},
        op::ParameterisedOp,
        FaerSparseMat, Op,
    };

    use super::*;

    #[test]
    fn test_klu() {
        let (op, rtol, atol, solns) = linear_problem::<FaerSparseMat<f64>>();
        let p = FaerVec::zeros(0, op.context().clone());
        let op = ParameterisedOp::new(&op, &p);
        let s = KLU::default();
        test_linear_solver(s, op, rtol, &atol, solns);
    }
}
