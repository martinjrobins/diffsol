use std::{cell::RefCell, rc::Rc};

use faer::Col;

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
    linear_solver::LinearSolver, matrix::MatrixCommon, op::linearise::LinearisedOp, vector::Vector,
    LinearOp, Matrix, MatrixSparsityRef, NonLinearOp, Op, SolverProblem, SparseColMat,
};

trait MatrixKLU: Matrix<T = f64> {
    fn column_pointers_mut_ptr(&mut self) -> *mut KluIndextype;
    fn row_indices_mut_ptr(&mut self) -> *mut KluIndextype;
    fn values_mut_ptr(&mut self) -> *mut f64;
}

impl MatrixKLU for SparseColMat<f64> {
    fn column_pointers_mut_ptr(&mut self) -> *mut KluIndextype {
        let ptrs = self.faer().symbolic().col_ptrs();
        ptrs.as_ptr() as *mut KluIndextype
    }

    fn row_indices_mut_ptr(&mut self) -> *mut KluIndextype {
        let indices = self.faer().symbolic().row_indices();
        indices.as_ptr() as *mut KluIndextype
    }

    fn values_mut_ptr(&mut self) -> *mut f64 {
        let values = self.faer().values();
        values.as_ptr() as *mut f64
    }
}

trait VectorKLU: Vector {
    fn values_mut_ptr(&mut self) -> *mut f64;
}

impl VectorKLU for Col<f64> {
    fn values_mut_ptr(&mut self) -> *mut f64 {
        self.as_mut().as_ptr_mut()
    }
}

struct KluSymbolic {
    inner: *mut klu_symbolic,
    common: *mut klu_common,
}

impl KluSymbolic {
    fn try_from_matrix(mat: &mut impl MatrixKLU, common: *mut klu_common) -> Result<Self> {
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
            return Err(anyhow::anyhow!("KLU failed to analyze"));
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
    fn try_from_symbolic(symbolic: &mut KluSymbolic, mat: &mut impl MatrixKLU) -> Result<Self> {
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
            return Err(anyhow::anyhow!("KLU failed to factorize"));
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

pub struct KLU<M, C>
where
    M: Matrix,
    C: NonLinearOp<M = M, V = M::V, T = M::T>,
{
    klu_common: RefCell<KluCommon>,
    klu_symbolic: Option<KluSymbolic>,
    klu_numeric: Option<KluNumeric>,
    problem: Option<SolverProblem<LinearisedOp<C>>>,
    matrix: Option<M>,
}

impl<M, C> Default for KLU<M, C>
where
    M: Matrix,
    C: NonLinearOp<M = M, V = M::V, T = M::T>,
{
    fn default() -> Self {
        let klu_common = KluCommon::default();
        let klu_common = RefCell::new(klu_common);
        Self {
            klu_common,
            klu_numeric: None,
            klu_symbolic: None,
            problem: None,
            matrix: None,
        }
    }
}

impl<M, C> LinearSolver<C> for KLU<M, C>
where
    M: MatrixKLU,
    M::V: VectorKLU,
    C: NonLinearOp<M = M, V = M::V, T = M::T>,
{
    fn set_linearisation(&mut self, x: &C::V, t: C::T) {
        Rc::<LinearisedOp<C>>::get_mut(&mut self.problem.as_mut().expect("Problem not set").f)
            .unwrap()
            .set_x(x);
        let matrix = self.matrix.as_mut().expect("Matrix not set");
        self.problem.as_ref().unwrap().f.matrix_inplace(t, matrix);
        self.klu_numeric = KluNumeric::try_from_symbolic(
            self.klu_symbolic.as_mut().expect("Symbolic not set"),
            matrix,
        )
        .ok();
    }

    fn solve_in_place(&self, x: &mut C::V) -> Result<()> {
        if self.klu_numeric.is_none() {
            return Err(anyhow::anyhow!("LU not initialized"));
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

    fn set_problem(&mut self, problem: &SolverProblem<C>) {
        let linearised_problem = problem.linearise();
        let ncols = linearised_problem.f.nstates();
        let nrows = linearised_problem.f.nout();
        let mut matrix = C::M::new_from_sparsity(
            nrows,
            ncols,
            linearised_problem.f.sparsity().map(|s| s.to_owned()),
        );
        self.problem = Some(linearised_problem);
        let mut klu_common = self.klu_common.borrow_mut();
        self.klu_symbolic = KluSymbolic::try_from_matrix(&mut matrix, klu_common.as_mut()).ok();
        self.matrix = Some(matrix);
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        linear_solver::tests::{linear_problem, test_linear_solver},
        SparseColMat,
    };

    use super::*;

    #[test]
    fn test_klu() {
        let (p, solns) = linear_problem::<SparseColMat<f64>>();
        let s = KLU::default();
        test_linear_solver(s, p, solns);
    }
}
