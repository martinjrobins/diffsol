use std::{cell::RefCell, rc::Rc};

use anyhow::Result;
use suitesparse_sys::{klu_l_analyze, klu_l_common, klu_l_defaults, klu_l_solve, klu_l_factor};

use crate::{linear_solver::LinearSolver, matrix::MatrixCommon, op::linearise::LinearisedOp, scalar::IndexType, vector::Vector, LinearOp, Matrix, MatrixSparsityRef, NonLinearOp, Op, SolverProblem};

trait MatrixKLU: Matrix<T = f64>{
    fn column_pointers_mut(&mut self) -> &mut [IndexType];
    fn row_indices_mut(&mut self) -> &mut [IndexType];
    fn values_mut(&mut self) -> &mut [f64];
}

trait VectorKLU: Vector {
    fn values_mut(&mut self) -> *mut f64;
}

struct KluSymbolic {
    inner: *mut suitesparse_sys::klu_l_symbolic,
    common: *mut suitesparse_sys::klu_l_common,
}

impl KluSymbolic {
    fn try_from_matrix(mat: &mut impl MatrixKLU, common: *mut klu_l_common) -> Result<Self> {
        let n = mat.nrows() as i64;
        let inner = unsafe { klu_l_analyze(n, mat.column_pointers_mut().as_mut_ptr() as *mut i64, mat.row_indices_mut().as_mut_ptr() as *mut i64, common) };
        if inner.is_null() {
            return Err(anyhow::anyhow!("KLU failed to analyze"));
        };
        Ok(Self {
            inner, common
        })
    }
}

impl Drop for KluSymbolic {
    fn drop(&mut self) {
        unsafe {
            klu_free_l_symbolic(&mut self.inner, self.common);
        }
    }
}

struct KluNumeric {
    inner: *mut klu_l_numeric,
    common: *mut klu_l_common,
}

impl KluNumeric {
    fn try_from_symbolic(symbolic: &mut KluSymbolic, mat: &mut impl MatrixKLU) -> Result<Self> {
        let inner = unsafe { klu_l_factor( mat.column_pointers_mut().as_mut_ptr() as *mut i64, mat.row_indices_mut().as_mut_ptr() as *mut i64, mat.values_mut().as_mut_ptr(), symbolic.inner, symbolic.common) };
        if inner.is_null() {
            return Err(anyhow::anyhow!("KLU failed to factorize"));
        };
        Ok(Self {
            inner, common: symbolic.common
        })
    }
}


impl Drop for KluNumeric {
    fn drop(&mut self) {
        unsafe {
            suitesparse_sys::klu_free_numeric(&mut self.inner, self.common);
        }
    }
}

struct KluCommon {
    inner: suitesparse_sys::klu_l_common,
}

impl Default for KluCommon {
    fn default() -> Self {
        let mut inner = suitesparse_sys::klu_l_common::default();
        unsafe { klu_l_defaults(&mut inner) };
        Self { inner }
    }
}

impl KluCommon {
    fn as_mut(&mut self) -> *mut suitesparse_sys::klu_l_common {
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
        self.klu_numeric = KluNumeric::try_from_symbolic(self.klu_symbolic.as_mut().expect("Symbolic not set"), matrix).ok();
    }

    fn solve_in_place(&self, x: &mut C::V) -> Result<()> {
        if self.klu_numeric.is_none() {
            return Err(anyhow::anyhow!("LU not initialized"));
        }
        let klu_numeric = self.klu_numeric.as_ref().unwrap();
        let klu_symbolic = self.klu_symbolic.as_ref().unwrap();
        let n = self.matrix.as_ref().unwrap().nrows() as i32;
        let mut klu_common= self.klu_common.borrow_mut();
        unsafe { klu_solve(klu_symbolic.inner, klu_numeric.inner, n, 1, x.values_mut(), klu_common.as_mut()) };
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
        let mut klu_common= self.klu_common.borrow_mut();
        self.klu_symbolic = KluSymbolic::try_from_matrix(&mut matrix, klu_common.as_mut()).ok();
        self.matrix = Some(matrix);
    }
}
