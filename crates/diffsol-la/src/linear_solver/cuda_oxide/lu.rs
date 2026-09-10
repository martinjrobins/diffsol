//! Dense LU on cuSOLVER, for the `cuda-oxide` backend.
//!
//! cuda-oxide does not wrap cuSOLVER, so this goes through cudarc's raw `sys`
//! bindings with cuda-oxide's buffers handed over as `CUdeviceptr`s. Both
//! libraries retain the *primary* context for a device ordinal, so there is one
//! `CUcontext` and a pointer means the same thing on either side.
//!
//! One factorisation per batch lane, sharing one workspace, and one lane's
//! factorisation can serve several right-hand-side lanes (grouped broadcast).

use std::cell::RefCell;
use std::ffi::c_int;
use std::mem::MaybeUninit;

use cuda_core::DeviceBuffer;
use cudarc::cusolver::sys::{
    cublasOperation_t, cusolverDnCreate, cusolverDnDestroy, cusolverDnDgetrf,
    cusolverDnDgetrf_bufferSize, cusolverDnDgetrs, cusolverDnHandle_t, cusolverDnSetStream,
    cusolverStatus_t,
};

use crate::context::broadcast_batch;
use crate::{
    error::LaError, linear_solver_error, Context, LinearOp, LinearSolver, Matrix, OxideContext,
    OxideMat, OxideVec, Vector,
};

/// Panics unless a cuSOLVER call succeeded.
///
/// The `cuda` backend discards these statuses; a wrong `lwork` or a bad handle
/// would otherwise show up as silently wrong numbers much later.
fn check(status: cusolverStatus_t, what: &str) {
    assert_eq!(
        status,
        cusolverStatus_t::CUSOLVER_STATUS_SUCCESS,
        "cuSOLVER call {} failed: {:?}",
        what,
        status
    );
}

pub struct OxideLU {
    work: Option<DeviceBuffer<f64>>,
    pivots: Option<DeviceBuffer<i32>>,
    /// cuSOLVER's per-lane `info` output. Written by the device, so it needs
    /// interior mutability to be filled from `&self` in `solve_in_place`.
    nfo: Option<RefCell<DeviceBuffer<i32>>>,
    matrix: Option<OxideMat>,
    handle: cusolverDnHandle_t,
    linearisation_set: bool,
}

impl Default for OxideLU {
    fn default() -> Self {
        let handle = {
            let mut handle = MaybeUninit::uninit();
            // SAFETY: `handle` is a valid out-pointer; the status is checked.
            unsafe {
                check(cusolverDnCreate(handle.as_mut_ptr()), "cusolverDnCreate");
                handle.assume_init()
            }
        };
        Self {
            matrix: None,
            work: None,
            pivots: None,
            nfo: None,
            handle,
            linearisation_set: false,
        }
    }
}

impl Drop for OxideLU {
    fn drop(&mut self) {
        // SAFETY: created in `Default::default` and not used again.
        unsafe {
            cusolverDnDestroy(self.handle);
        }
    }
}

impl OxideLU {
    /// Points cuSOLVER at the context's stream, so its work is ordered against
    /// the kernel launches rather than racing them.
    fn bind_stream(&self, ctx: &OxideContext) {
        // SAFETY: the handle is live and the stream belongs to this context.
        unsafe {
            check(
                cusolverDnSetStream(self.handle, ctx.stream.cu_stream() as _),
                "cusolverDnSetStream",
            );
        }
    }
}

impl LinearSolver<OxideMat> for OxideLU {
    fn set_sparsity<C: LinearOp<T = f64, V = OxideVec, M = OxideMat, C = OxideContext>>(
        &mut self,
        op: &C,
    ) {
        let nrows = op.nrows();
        let ncols = op.ncols();
        let ctx = op.context().clone();
        let nbatch = ctx.nbatch();
        let matrix = OxideMat::new_from_sparsity(nrows, ncols, op.sparsity(), ctx.clone());
        self.matrix = Some(matrix);
        self.bind_stream(&ctx);

        let m = c_int::try_from(nrows).unwrap();
        let n = c_int::try_from(ncols).unwrap();
        let lda = m;
        let mut lwork = 0;
        let a = self.matrix.as_ref().unwrap().data.cu_deviceptr();
        // SAFETY: `a` addresses `nrows * ncols * nbatch` doubles in this
        // context, so it is a valid `lda`-by-`n` matrix; `lwork` is a valid
        // out-pointer.
        unsafe {
            check(
                cusolverDnDgetrf_bufferSize(self.handle, m, n, a as *mut f64, lda, &mut lwork),
                "cusolverDnDgetrf_bufferSize",
            );
        }

        self.work = Some(
            DeviceBuffer::zeroed(&ctx.stream, lwork as usize).expect("Failed to allocate work"),
        );
        self.pivots = Some(
            DeviceBuffer::zeroed(&ctx.stream, nrows * nbatch).expect("Failed to allocate pivots"),
        );
        self.nfo = Some(RefCell::new(
            DeviceBuffer::zeroed(&ctx.stream, nbatch).expect("Failed to allocate info"),
        ));
        self.linearisation_set = false;
    }

    fn set_linearisation<C: LinearOp<T = f64, V = OxideVec, M = OxideMat, C = OxideContext>>(
        &mut self,
        op: &C,
    ) {
        op.matrix_inplace(self.matrix.as_mut().expect("Matrix not set"));
        let matrix = self.matrix.as_ref().unwrap();
        let nbatch = op.context().nbatch();
        let nrows = matrix.nrows();
        let ncols = matrix.ncols();
        self.bind_stream(op.context());

        let m = c_int::try_from(nrows).unwrap();
        let n = c_int::try_from(ncols).unwrap();
        let lda = m;
        let a = matrix.data.cu_deviceptr() as *mut f64;
        let ws = self
            .work
            .as_ref()
            .expect("Work space not set")
            .cu_deviceptr() as *mut f64;
        let piv = self.pivots.as_ref().expect("Pivots not set").cu_deviceptr() as *mut i32;
        let info = self
            .nfo
            .as_ref()
            .expect("Info not set")
            .borrow()
            .cu_deviceptr() as *mut i32;
        for b in 0..nbatch {
            // SAFETY: lane `b` of each buffer is in bounds -- the matrix holds
            // `nbatch` blocks of `nrows * ncols`, the pivots `nbatch` blocks of
            // `nrows`, and the info array `nbatch` entries. The workspace is
            // shared because the factorisations run in sequence on one stream.
            unsafe {
                check(
                    cusolverDnDgetrf(
                        self.handle,
                        m,
                        n,
                        a.add(b * nrows * ncols),
                        lda,
                        ws,
                        piv.add(b * nrows),
                        info.add(b),
                    ),
                    "cusolverDnDgetrf",
                );
            }
        }
        self.linearisation_set = true;
    }

    fn solve_in_place(&self, x: &mut OxideVec) -> Result<(), LaError> {
        let matrix = if let Some(ref matrix) = self.matrix {
            if matrix.nrows() != matrix.ncols() {
                Err(linear_solver_error!(LinearSolverMatrixNotSquare))?;
            }
            matrix
        } else {
            Err(linear_solver_error!(LinearSolverNotSetup))?
        };
        if !self.linearisation_set {
            Err(linear_solver_error!(LinearSolverNotSetup))?;
        }
        let ctx = x.context.clone();
        let nbatch = ctx.nbatch();
        let lu_nbatch = matrix.context().nbatch();
        ctx.assert_broadcastable_into(lu_nbatch, "lu_solve");
        let nrows = matrix.nrows();
        let ncols = matrix.ncols();
        let x_nstates = x.len();
        if x_nstates != nrows {
            Err(linear_solver_error!(LinearSolverMatrixVectorNotCompatible))?;
        }
        self.bind_stream(&ctx);

        let n = c_int::try_from(nrows).unwrap();
        let lda = n;
        let nrhs = 1;
        let a = matrix.data.cu_deviceptr() as *const f64;
        let piv = self.pivots.as_ref().unwrap().cu_deviceptr() as *const i32;
        let nfo = self.nfo.as_ref().expect("Info not set").borrow_mut();
        let info = nfo.cu_deviceptr() as *mut i32;
        let xp = x.data.cu_deviceptr() as *mut f64;
        for b in 0..nbatch {
            // one factorization can serve several right-hand side batches
            let lu_b = broadcast_batch(b, lu_nbatch, nbatch);
            // SAFETY: as in `set_linearisation`; `lu_b < lu_nbatch` by
            // `broadcast_batch` and `b < nbatch`, which is `x`'s lane count.
            unsafe {
                check(
                    cusolverDnDgetrs(
                        self.handle,
                        cublasOperation_t::CUBLAS_OP_N,
                        n,
                        nrhs,
                        a.add(lu_b * nrows * ncols),
                        lda,
                        piv.add(lu_b * nrows),
                        xp.add(b * x_nstates),
                        n,
                        info.add(lu_b),
                    ),
                    "cusolverDnDgetrs",
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        linear_solver::tests::{diagonal_op, test_grouped_lu_solve, test_narrow_state_lu_solve},
        Vector,
    };

    #[test]
    fn test_lu() {
        let mut s = OxideLU::default();
        let op = diagonal_op::<OxideMat>(2.0);
        s.set_sparsity(&op);
        s.set_linearisation(&op);
        let b = OxideVec::from_vec(vec![2.0, 4.0], Default::default());
        let x = s.solve(&b).unwrap();
        x.assert_eq_st(
            &OxideVec::from_vec(vec![1.0, 2.0], Default::default()),
            1e-10,
        );
    }

    #[test]
    #[should_panic(expected = "incompatible nbatch")]
    fn test_narrow_state_lu() {
        test_narrow_state_lu_solve::<OxideMat, OxideLU>(OxideContext::default().with_nbatch(2));
    }

    #[test]
    fn test_grouped_lu() {
        test_grouped_lu_solve::<OxideMat, OxideLU>(OxideContext::default().with_nbatch(2));
    }
}
