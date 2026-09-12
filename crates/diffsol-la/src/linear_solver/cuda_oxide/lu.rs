//! Dense LU on cuSOLVER and cuBLAS, for the `cuda-oxide` backend.
//!
//! Neither library is wrapped by cuda-oxide, so this goes through cudarc's raw
//! `sys` bindings with cuda-oxide's buffers handed over as `CUdeviceptr`s. Both
//! libraries retain the *primary* context for a device ordinal, so there is one
//! `CUcontext` and a pointer means the same thing on either side.
//!
//! Small matrices with many batch lanes go through cuBLAS's batched LU
//! (`getrfBatched`/`getrsBatched`), which factorises or solves every lane in one
//! launch. Large matrices keep the per-lane cuSOLVER loop, whose blocked
//! factorisation wins once there is real work per lane. Either way one lane's
//! factorisation can serve several right-hand-side lanes (grouped broadcast).

use std::cell::{Cell, RefCell};
use std::ffi::c_int;
use std::mem::MaybeUninit;

use cuda_core::DeviceBuffer;
use cudarc::cublas::sys as cublas;
use cudarc::cusolver::sys::{
    cublasOperation_t, cusolverDnCreate, cusolverDnDestroy, cusolverDnDgetrf,
    cusolverDnDgetrf_bufferSize, cusolverDnDgetrs, cusolverDnHandle_t, cusolverDnSetStream,
    cusolverStatus_t,
};

use crate::context::{broadcast_batch, cuda_oxide::copy_at};
use crate::{
    error::LaError, linear_solver_error, Context, LinearOp, LinearSolver, Matrix, OxideContext,
    OxideMat, OxideVec, Vector,
};

/// For nrows above this, cuSolver is always faster.
const BATCHED_MAX_N: usize = 1024;

/// For nrows below this, cuBLAS is always faster
const BATCHED_SMALL_N: usize = 16;

/// when solving, right-hand-side lanes a batched cuBLAS solve needs to beat the cuSOLVER per-lane loop.
const BATCHED_MIN_SOLVE_LANES: usize = 8;

/// Whether a factorisation of `nbatch` lanes of an `nrows`-square matrix goes
/// through cuBLAS's batched LU rather than the per-lane cuSOLVER loop.
///
/// The two libraries write the same LAPACK factorisation and pivot array, so
/// the factorisation and the solve decide separately.
fn batched_factor(nrows: usize, nbatch: usize) -> bool {
    nrows <= BATCHED_MAX_N && (nbatch > 1 || nrows <= BATCHED_SMALL_N)
}

/// Whether a solve of `nbatch` right-hand-side lanes of size `nrows`,
/// goes through cuBLAS's batched triangular solve or cuSOLVER's per-lane loop.
fn batched_solve(nrows: usize, nbatch: usize) -> bool {
    nrows <= BATCHED_MAX_N && (nbatch >= BATCHED_MIN_SOLVE_LANES || nrows <= BATCHED_SMALL_N)
}

/// Panics unless a cuSOLVER call succeeded.
fn check(status: cusolverStatus_t, what: &str) {
    assert_eq!(
        status,
        cusolverStatus_t::CUSOLVER_STATUS_SUCCESS,
        "cuSOLVER call {} failed: {:?}",
        what,
        status
    );
}

/// Device pointers to the `nbatch` lanes of a buffer whose lanes are
/// `lane_elems` `f64`s apart, with lane `b` reading source lane
/// `broadcast_batch(b, src_nbatch, nbatch)`.
///
/// Pass `src_nbatch == nbatch` for the plain, non-broadcasting list.
fn lane_ptrs(base: u64, lane_elems: usize, src_nbatch: usize, nbatch: usize) -> Vec<u64> {
    let stride = (lane_elems * size_of::<f64>()) as u64;
    (0..nbatch)
        .map(|b| base + broadcast_batch(b, src_nbatch, nbatch) as u64 * stride)
        .collect()
}

/// One lane count's right-hand-side pointer array. `base` is the vector the
/// array currently points into, or 0 before it is first filled.
struct XPtrs {
    base: u64,
    nbatch: usize,
    buf: DeviceBuffer<u64>,
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
    /// One pointer per matrix lane, for the batched calls. Built in `set_sparsity`.
    a_ptrs: Option<DeviceBuffer<u64>>,
    /// [`Self::a_ptrs`] under grouped rhs broadcast.
    a_ptrs_bcast: RefCell<Option<(usize, DeviceBuffer<u64>)>>,
    /// One pointer per right-hand-side lane, one slot per lane count.
    /// (lane counts can alternate with sensitivity solves)
    x_ptrs: RefCell<Vec<XPtrs>>,
    /// The pivots of [`Self::pivots`] replicated per right-hand-side lane, for
    /// grouped broadcast, keyed on that lane count.
    piv_bcast: RefCell<Option<(usize, DeviceBuffer<i32>)>>,
    /// Set by `set_linearisation`, cleared when [`Self::piv_bcast`] is refilled.
    piv_dirty: Cell<bool>,
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
            a_ptrs: None,
            a_ptrs_bcast: RefCell::new(None),
            x_ptrs: RefCell::new(Vec::new()),
            piv_bcast: RefCell::new(None),
            piv_dirty: Cell::new(true),
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

    /// Device array of `nbatch` pointers into the factorised matrix, one per
    /// right-hand-side lane.
    fn solve_a_ptrs(
        &self,
        ctx: &OxideContext,
        lane_elems: usize,
        lu_nbatch: usize,
        nbatch: usize,
    ) -> u64 {
        if lu_nbatch == nbatch {
            return self
                .a_ptrs
                .as_ref()
                .expect("Pointers not set")
                .cu_deviceptr();
        }
        let base = self.matrix.as_ref().unwrap().data.cu_deviceptr();
        let mut cache = self.a_ptrs_bcast.borrow_mut();
        if cache.as_ref().is_none_or(|(n, _)| *n != nbatch) {
            let ptrs = lane_ptrs(base, lane_elems, lu_nbatch, nbatch);
            let buf = DeviceBuffer::from_host(&ctx.stream, &ptrs)
                .expect("Failed to allocate matrix pointers");
            *cache = Some((nbatch, buf));
        }
        cache.as_ref().unwrap().1.cu_deviceptr()
    }

    /// Device array of `nbatch` pointers into `x`, one per lane.
    fn solve_x_ptrs(&self, ctx: &OxideContext, x: &OxideVec, nstates: usize, nbatch: usize) -> u64 {
        let base = x.data.cu_deviceptr();
        let mut slots = self.x_ptrs.borrow_mut();
        let slot = match slots.iter().position(|s| s.nbatch == nbatch) {
            Some(i) => &mut slots[i],
            None => {
                let buf = DeviceBuffer::zeroed(&ctx.stream, nbatch)
                    .expect("Failed to allocate rhs pointers");
                slots.push(XPtrs {
                    base: 0,
                    nbatch,
                    buf,
                });
                slots.last_mut().unwrap()
            }
        };
        if slot.base != base {
            let ptrs = lane_ptrs(base, nstates, nbatch, nbatch);
            slot.buf
                .copy_from_host(&ctx.stream, &ptrs)
                .expect("Failed to fill rhs pointers");
            slot.base = base;
        }
        slot.buf.cu_deviceptr()
    }

    /// Pivots laid out per right-hand-side lane, as `getrsBatched` indexes them.
    ///
    /// `devIpiv` is one contiguous array indexed by lane *position*, not a
    /// pointer array, so a factorisation shared by several lanes needs its
    /// pivots replicated.
    ///
    // TODO: nbatch device-to-device copies per factorisation, single
    // broadcast kernel or a strided 2D copy could be faster
    fn solve_pivots(
        &self,
        ctx: &OxideContext,
        nrows: usize,
        lu_nbatch: usize,
        nbatch: usize,
    ) -> u64 {
        let pivots = self.pivots.as_ref().expect("Pivots not set");
        if lu_nbatch == nbatch {
            return pivots.cu_deviceptr();
        }
        let mut cache = self.piv_bcast.borrow_mut();
        let stale = self.piv_dirty.get() || cache.as_ref().is_none_or(|(n, _)| *n != nbatch);
        if stale {
            let buf = match cache.take() {
                Some((n, buf)) if n == nbatch => buf,
                _ => DeviceBuffer::zeroed(&ctx.stream, nrows * nbatch)
                    .expect("Failed to allocate pivots"),
            };
            for b in 0..nbatch {
                let lu_b = broadcast_batch(b, lu_nbatch, nbatch);
                copy_at(&ctx.stream, &buf, b * nrows, pivots, lu_b * nrows, nrows)
                    .expect("Failed to copy pivots");
            }
            *cache = Some((nbatch, buf));
            self.piv_dirty.set(false);
        }
        cache.as_ref().unwrap().1.cu_deviceptr()
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
        let ptrs = lane_ptrs(a, nrows * ncols, nbatch, nbatch);
        self.a_ptrs = Some(
            DeviceBuffer::from_host(&ctx.stream, &ptrs)
                .expect("Failed to allocate matrix pointers"),
        );
        *self.a_ptrs_bcast.borrow_mut() = None;
        self.x_ptrs.borrow_mut().clear();
        *self.piv_bcast.borrow_mut() = None;
        self.piv_dirty.set(true);
        self.linearisation_set = false;
    }

    fn set_linearisation<C: LinearOp<T = f64, V = OxideVec, M = OxideMat, C = OxideContext>>(
        &mut self,
        op: &C,
    ) {
        op.matrix_inplace(self.matrix.as_mut().expect("Matrix not set"));
        let matrix = self.matrix.as_ref().unwrap();
        let ctx = op.context();
        let nbatch = ctx.nbatch();
        let nrows = matrix.nrows();
        let ncols = matrix.ncols();
        self.bind_stream(ctx);

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

        if nrows == ncols && batched_factor(nrows, nbatch) {
            let a_ptrs = self
                .a_ptrs
                .as_ref()
                .expect("Pointers not set")
                .cu_deviceptr();
            ctx.with_blas(|handle| {
                // SAFETY: `a_ptrs` holds the `nbatch` lane pointers of the
                // matrix, the pivots `nbatch` blocks of `nrows` and the info
                // array `nbatch` entries, all in this context.
                unsafe {
                    cublas::cublasDgetrfBatched(
                        handle,
                        n,
                        a_ptrs as *const *mut f64,
                        lda,
                        piv,
                        info,
                        c_int::try_from(nbatch).unwrap(),
                    )
                    .result()
                    .expect("Failed to launch cublasDgetrfBatched");
                }
            });
        } else {
            for b in 0..nbatch {
                // SAFETY: lane `b` of each buffer is in bounds -- the matrix
                // holds `nbatch` blocks of `nrows * ncols`, the pivots `nbatch`
                // blocks of `nrows`, and the info array `nbatch` entries. The
                // workspace is shared because the factorisations run in
                // sequence on one stream.
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
        }
        self.piv_dirty.set(true);
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

        let n = c_int::try_from(nrows).unwrap();
        let lda = n;
        let nrhs = 1;

        if batched_solve(nrows, nbatch) {
            let a_ptrs = self.solve_a_ptrs(&ctx, nrows * ncols, lu_nbatch, nbatch);
            let piv = self.solve_pivots(&ctx, nrows, lu_nbatch, nbatch);
            let x_ptrs = self.solve_x_ptrs(&ctx, x, x_nstates, nbatch);
            let mut info = 0;
            ctx.with_blas(|handle| {
                // SAFETY: both pointer arrays hold `nbatch` lane pointers in
                // this context, and the pivots `nbatch` blocks of `nrows`.
                unsafe {
                    cublas::cublasDgetrsBatched(
                        handle,
                        cublas::cublasOperation_t::CUBLAS_OP_N,
                        n,
                        nrhs,
                        a_ptrs as *const *const f64,
                        lda,
                        piv as *const c_int,
                        x_ptrs as *const *mut f64,
                        n,
                        &mut info,
                        c_int::try_from(nbatch).unwrap(),
                    )
                    .result()
                    .expect("Failed to launch cublasDgetrsBatched");
                }
            });
            assert_eq!(info, 0, "cublasDgetrsBatched rejected parameter {}", -info);
            return Ok(());
        }

        self.bind_stream(&ctx);
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
        linear_solver::tests::{
            batched_diagonal_op, diagonal_op, diagonal_op_n, test_grouped_lu_solve,
            test_narrow_state_lu_solve,
        },
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

    /// The batched path with as many right-hand-side lanes as factorisations.
    #[test]
    fn test_batched_lu() {
        let ctx = OxideContext::default().with_nbatch(4);
        let op = batched_diagonal_op::<OxideMat>(&[2.0, 4.0, 5.0, 8.0], ctx.clone());
        let mut s = OxideLU::default();
        s.set_sparsity(&op);
        s.set_linearisation(&op);
        let b = OxideVec::from_vec((1..=8).map(|i| i as f64).collect(), ctx.clone());
        let x = s.solve(&b).unwrap();
        let expected = OxideVec::from_vec(vec![0.5, 1.0, 0.75, 1.0, 1.0, 1.2, 0.875, 1.0], ctx);
        x.assert_eq_st(&expected, 1e-10);
    }

    /// cuSOLVER factorises (one lane, `n` past the small-n cut) and cuBLAS
    /// solves the broadcast right-hand sides -- the two write the same LAPACK
    /// factorisation and pivots.
    #[test]
    fn test_cusolver_factor_batched_solve() {
        let ctx = OxideContext::default();
        let op = diagonal_op_n::<OxideMat>(BATCHED_SMALL_N + 4, 2.0, ctx);
        let mut s = OxideLU::default();
        s.set_sparsity(&op);
        s.set_linearisation(&op);
        let wide = OxideContext::default().with_nbatch(BATCHED_MIN_SOLVE_LANES);
        let b = OxideVec::from_element(BATCHED_SMALL_N + 4, 4.0, wide.clone());
        let x = s.solve(&b).unwrap();
        x.assert_eq_st(
            &OxideVec::from_element(BATCHED_SMALL_N + 4, 2.0, wide),
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
