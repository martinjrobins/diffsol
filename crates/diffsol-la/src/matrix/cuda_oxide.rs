//! Dense matrices for the `cuda-oxide` backend.
//!
//! Same layout and batching rules as the `cuda` backend's
//! [`CudaMat`](crate::matrix::cuda::CudaMat): column-major, `nrows * ncols`
//! elements per batch, whole batches contiguous.
//!
//! ```text
//! Device memory: [b0(all), b1(all), ..., bN(all)]
//! ```
//!
//! Because a batch's elements are contiguous, most matrix operations are the
//! vector kernels over `nrows * ncols` elements, and a single column is a
//! window starting at `j * nrows` — so `set_column`, `column`, `column_mut` and
//! `add_column_to_vector` need no matrix-specific kernel at all.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_int;
use std::fmt::{self, Debug};
use std::mem::MaybeUninit;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use cudarc::cublas::sys as cublas;

use cuda_core::DeviceBuffer;

use crate::context::cuda_oxide::{copy_at, read_at, write_at};
use crate::vector::cuda_oxide::{
    launch_assign, launch_axpy, launch_mul_assign_scalar, launch_mul_scalar, AssignOp, Operand,
    OperandMut,
};
use crate::{
    context::broadcast_batch, error::LaError, linear_solver::cuda_oxide::lu::OxideLU,
    matrix::default_solver::DefaultSolver, matrix_error, Context, IndexType, MatrixCommon,
    OxideContext, OxideVec, OxideVecMut, OxideVecRef, Scale, Vector, VectorIndex,
};

use super::{
    sparsity::{Dense, DenseRef},
    DenseMatrix, Matrix, MAX_SMALL_COLS,
};

/// Dense matrix in GPU memory. See the module docs for the layout.
pub struct OxideMat {
    pub(crate) data: DeviceBuffer<f64>,
    pub(crate) context: OxideContext,
    nrows: IndexType,
    ncols: IndexType,
}

impl Debug for OxideMat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OxideMat")
            .field("nrows", &self.nrows)
            .field("ncols", &self.ncols)
            .field("nbatch", &self.context.nbatch())
            .finish()
    }
}

impl Clone for OxideMat {
    fn clone(&self) -> Self {
        let mut data = DeviceBuffer::zeroed(&self.context.stream, self.data.len())
            .expect("Failed to allocate device memory");
        data.copy_from_device_async(&self.data, &self.context.stream)
            .expect("Failed to copy device to device");
        Self {
            data,
            context: self.context.clone(),
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }
}

impl OxideMat {
    pub fn nrows(&self) -> IndexType {
        self.nrows
    }
    pub fn ncols(&self) -> IndexType {
        self.ncols
    }
    /// Elements in one batch.
    fn batch_len(&self) -> IndexType {
        self.nrows * self.ncols
    }
    /// The whole matrix as a launch source: `nrows * ncols` elements per
    /// batch.
    pub(crate) fn operand(&self) -> Operand<'_> {
        let n = self.batch_len();
        Operand::new(&self.data, 0, n, n, self.context.nbatch())
    }
    /// The whole matrix as a launch destination.
    pub(crate) fn operand_mut(&mut self) -> OperandMut<'_> {
        let (n, nbatch) = (self.batch_len(), self.context.nbatch());
        OperandMut::new(&mut self.data, 0, n, n, nbatch)
    }
    /// Columns `start..start + ncols` of every batch, as a launch source of
    /// `ncols * nrows` elements starting at that column.
    fn cols_operand(&self, start: IndexType, ncols: IndexType) -> Operand<'_> {
        Operand::new(
            &self.data,
            start * self.nrows,
            self.batch_len(),
            ncols * self.nrows,
            self.context.nbatch(),
        )
    }
    /// [`Self::cols_operand`] as a launch destination.
    fn cols_operand_mut(&mut self, start: IndexType, ncols: IndexType) -> OperandMut<'_> {
        let (offset, stride) = (start * self.nrows, self.batch_len());
        let (nstates, nbatch) = (ncols * self.nrows, self.context.nbatch());
        OperandMut::new(&mut self.data, offset, stride, nstates, nbatch)
    }
    fn col_major_index(&self, i: IndexType, j: IndexType) -> IndexType {
        i + j * self.nrows
    }
    fn get_index_impl(&self, i: IndexType, j: IndexType) -> f64 {
        let mut out = [0.0f64];
        read_at(
            &self.context.stream,
            &self.data,
            self.col_major_index(i, j),
            &mut out,
        )
        .expect("Failed to copy data from device to host");
        out[0]
    }
    fn set_index_impl(&mut self, i: IndexType, j: IndexType, value: f64) {
        let index = self.col_major_index(i, j);
        write_at(&self.context.stream, &self.data, index, &[value])
            .expect("Failed to copy data from host to device");
    }
    fn diagonal(&self) -> OxideVec {
        assert_eq!(
            self.nrows, self.ncols,
            "Matrix must be square to get diagonal"
        );
        let ctx = self.context.clone();
        let nbatch = ctx.nbatch();
        let n = self.nrows;
        let mut diag = OxideVec::zeros(n, ctx.clone());
        if n == 0 {
            return diag;
        }
        let mat = self.operand();
        let mut dest = diag.operand_mut();
        let total = (n * nbatch) as u32;
        let cfg = OxideContext::config_1d(total);
        let m = &ctx.module;
        let p = m
            .prepare_mat_get_diagonal(cfg)
            .expect("prepare mat_get_diagonal");
        m.mat_get_diagonal(
            &ctx.stream,
            &p,
            &mut dest.window,
            &mat.window,
            total,
            n as u32,
            mat.stride,
            mat.nbatch,
            nbatch as u32,
        )
        .expect("launch mat_get_diagonal");
        diag
    }
}

// ============================================================
// cuBLAS
//
// `gemv` goes through cudarc's raw `sys` bindings (cuda-oxide does
// not wrap these).
// ============================================================

/// A cuBLAS handle owned by one thread, for one device ordinal.
///
/// A handle is bound to the context current when it is created, and cuBLAS does
/// not support concurrent calls on one handle, so handles are kept per thread
/// rather than in the shared device registry.
struct BlasHandle(cublas::cublasHandle_t);

impl Drop for BlasHandle {
    fn drop(&mut self) {
        // SAFETY: created by `cublasCreate_v2` below and not used again.
        unsafe { cublas::cublasDestroy_v2(self.0) };
    }
}

thread_local! {
    static BLAS: RefCell<HashMap<usize, BlasHandle>> = RefCell::new(HashMap::new());
}

impl OxideContext {
    /// Runs `f` with this thread's cuBLAS handle for this device, bound to the
    /// context's stream so cuBLAS work is ordered against the kernel launches.
    pub(crate) fn with_blas<R>(&self, f: impl FnOnce(cublas::cublasHandle_t) -> R) -> R {
        let ordinal = self.stream.context().ordinal();
        let cu_stream = self.stream.cu_stream() as cublas::cudaStream_t;
        BLAS.with(|handles| {
            let mut handles = handles.borrow_mut();
            let handle = handles.entry(ordinal).or_insert_with(|| {
                let mut handle = MaybeUninit::uninit();
                // SAFETY: `handle` is a valid out-pointer, and this context is
                // bound to the thread by `OxideContext::new`.
                let handle = unsafe {
                    cublas::cublasCreate_v2(handle.as_mut_ptr())
                        .result()
                        .expect("Failed to create cuBLAS handle");
                    handle.assume_init()
                };
                BlasHandle(handle)
            });
            // SAFETY: the handle is live and the stream belongs to this context.
            unsafe {
                cublas::cublasSetStream_v2(handle.0, cu_stream)
                    .result()
                    .expect("Failed to set cuBLAS stream");
            }
            f(handle.0)
        })
    }

    /// `y = alpha * a * x + beta * y` over `count` lanes, column-major with unit increments,
    /// consecutive lanes `stride_*` elements apart. A stride of zero feeds every lane from the
    /// same operand.
    ///
    /// The pointers are raw so that a per-batch slice needs no view type; each must address at
    /// least `nrows * ncols`, `ncols` and `nrows` elements per lane.
    #[allow(clippy::too_many_arguments)]
    fn gemv_batched(
        &self,
        nrows: IndexType,
        ncols: IndexType,
        alpha: f64,
        beta: f64,
        a: u64,
        stride_a: i64,
        x: u64,
        stride_x: i64,
        y: u64,
        stride_y: i64,
        count: IndexType,
    ) {
        self.with_blas(|handle| {
            // SAFETY: the pointers are device pointers in this context, sized as documented
            // above, and each stride walks `count` lanes inside its own buffer; `nrows`/`ncols`
            // fit in `c_int` for any matrix that fits in device memory.
            unsafe {
                cublas::cublasDgemvStridedBatched(
                    handle,
                    cublas::cublasOperation_t::CUBLAS_OP_N,
                    nrows as c_int,
                    ncols as c_int,
                    &alpha as *const f64,
                    a as *const f64,
                    nrows as c_int,
                    stride_a,
                    x as *const f64,
                    1,
                    stride_x,
                    &beta as *const f64,
                    y as *mut f64,
                    1,
                    stride_y,
                    count as c_int,
                )
                .result()
                .expect("Failed to launch batched gemv");
            }
        });
    }

    /// [`Self::gemv_batched`] where each lane's `x` and `y` hold `k` columns: one matrix product
    /// per lane, column-major throughout.
    #[allow(clippy::too_many_arguments)]
    fn gemm_batched(
        &self,
        nrows: IndexType,
        k: IndexType,
        ncols: IndexType,
        alpha: f64,
        beta: f64,
        a: u64,
        stride_a: i64,
        x: u64,
        stride_x: i64,
        y: u64,
        stride_y: i64,
        count: IndexType,
    ) {
        self.with_blas(|handle| {
            // SAFETY: as `Self::gemv_batched`, with `x` and `y` holding `k` columns per lane.
            unsafe {
                cublas::cublasDgemmStridedBatched(
                    handle,
                    cublas::cublasOperation_t::CUBLAS_OP_N,
                    cublas::cublasOperation_t::CUBLAS_OP_N,
                    nrows as c_int,
                    k as c_int,
                    ncols as c_int,
                    &alpha as *const f64,
                    a as *const f64,
                    nrows as c_int,
                    stride_a,
                    x as *const f64,
                    ncols as c_int,
                    stride_x,
                    &beta as *const f64,
                    y as *mut f64,
                    nrows as c_int,
                    stride_y,
                    count as c_int,
                )
                .result()
                .expect("Failed to launch batched gemm");
            }
        });
    }
}

/// Group width at which one batched gemm beats `k` batched gemvs whatever the matrix size.
const GEMM_MIN_GROUP: usize = 16;

/// Matrix size below which one batched gemm wins at any group width.
const GEMM_MAX_NROWS: usize = 128;

/// Whether a grouped broadcast of `k` result lanes per matrix lane goes through one batched gemm
/// rather than `k` batched gemvs.
///
/// A gemm reads the matrix once per group instead of once per result lane, but cuBLAS pays for a
/// full tile width whatever `k` is, so a narrow group over a large matrix is faster as `k` gemvs
/// (A40, k = 4: 2.4x at 256 rows, 2.2x at 500; the gemm wins everywhere by k = 16).
fn grouped_as_gemm(nrows: IndexType, k: IndexType) -> bool {
    k >= GEMM_MIN_GROUP || nrows <= GEMM_MAX_NROWS
}

/// The cuBLAS calls a [`Matrix::gemv`] maps onto, chosen by [`GemvPlan::choose`].
#[derive(Debug, PartialEq, Eq)]
enum GemvPlan {
    /// `k` batched gemvs, sub-batch `j` covering result lanes `j, j + k, j + 2k, ...`: `count`
    /// lanes with each operand walking `stride_*` elements per lane. `k == 1` is the whole batch
    /// in one launch.
    Strided {
        k: IndexType,
        count: IndexType,
        stride_a: i64,
        stride_x: i64,
    },
    /// One batched gemm per matrix lane, `k` result lanes wide.
    Gemm { k: IndexType },
}

impl GemvPlan {
    /// How `nbatch` lanes of an `nrows x ncols` matrix and `x_nbatch` lanes of `x` multiply into
    /// `y_nbatch` result lanes, which `assert_broadcastable_into` has already checked are a whole
    /// multiple of both.
    fn choose(
        nrows: IndexType,
        ncols: IndexType,
        nbatch: IndexType,
        x_nbatch: IndexType,
        y_nbatch: IndexType,
    ) -> Self {
        // a grouped operand repeats each of its lanes over `k_src` contiguous result lanes; read
        // once per result lane, or broadcast over all of them, is `k_src == 1`
        let group_width = |src_nbatch: IndexType| {
            if src_nbatch == 1 || src_nbatch == y_nbatch {
                1
            } else {
                y_nbatch / src_nbatch
            }
        };
        let (k_a, k_x) = (group_width(nbatch), group_width(x_nbatch));
        // a grouped matrix against one `x` lane per result: a group's result lanes are
        // contiguous, and so are their `x` lanes, so the group is one `nrows x k_a` matrix product
        if k_a > 1 && x_nbatch == y_nbatch && grouped_as_gemm(nrows, k_a) {
            return Self::Gemm { k: k_a };
        }
        // otherwise, the result lanes sharing a source lane are contiguous, so splitting the
        // launch into `k` interleaved sub-batches gives every operand a constant stride again
        let k = k_a / gcd(k_a, k_x) * k_x;
        let stride = |src_nbatch: IndexType, lane: IndexType| match src_nbatch {
            // one source lane over every result lane
            1 => 0,
            // `k / k_src` source lanes per sub-batch lane, exactly because `k_src` divides `k`
            _ => (k * src_nbatch / y_nbatch * lane) as i64,
        };
        Self::Strided {
            k,
            count: y_nbatch / k,
            stride_a: stride(nbatch, nrows * ncols),
            stride_x: stride(x_nbatch, ncols),
        }
    }
}

fn gcd(a: IndexType, b: IndexType) -> IndexType {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

impl DefaultSolver for OxideMat {
    type LS = OxideLU;
}

impl MatrixCommon for OxideMat {
    type T = f64;
    type V = OxideVec;
    type C = OxideContext;
    type Inner = DeviceBuffer<f64>;

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

// ============================================================
// Operators. A matrix batch is contiguous, so these are the vector kernels
// over `nrows * ncols` elements.
// ============================================================

impl Mul<Scale<f64>> for OxideMat {
    type Output = OxideMat;
    fn mul(mut self, rhs: Scale<f64>) -> Self::Output {
        let ctx = self.context.clone();
        let n = self.batch_len();
        let dest = self.operand_mut();
        launch_mul_assign_scalar(&ctx, dest, n, rhs.value());
        self
    }
}

impl Mul<Scale<f64>> for &OxideMat {
    type Output = OxideMat;
    fn mul(self, rhs: Scale<f64>) -> Self::Output {
        let ctx = self.context.clone();
        let mut ret = OxideMat::zeros(self.nrows, self.ncols, ctx.clone());
        let n = self.batch_len();
        let src = self.operand();
        let dest = ret.operand_mut();
        launch_mul_scalar(&ctx, dest, n, &src, rhs.value());
        ret
    }
}

macro_rules! impl_mat_assign {
    ($Op:ident, $method:ident, $op:expr, $label:expr) => {
        impl $Op<&OxideMat> for OxideMat {
            fn $method(&mut self, rhs: &OxideMat) {
                let ctx = self.context.clone();
                // `self` is the destination, so `rhs` broadcasts into it
                ctx.assert_broadcastable_into(rhs.context.nbatch(), $label);
                let n = self.batch_len();
                let src = rhs.operand();
                let dest = self.operand_mut();
                launch_assign(&ctx, $op, dest, n, &src);
            }
        }
    };
}

impl_mat_assign!(AddAssign, add_assign, AssignOp::Add, "add_assign");
impl_mat_assign!(SubAssign, sub_assign, AssignOp::Sub, "sub_assign");

impl Add<&OxideMat> for OxideMat {
    type Output = OxideMat;
    fn add(mut self, rhs: &OxideMat) -> Self::Output {
        AddAssign::add_assign(&mut self, rhs);
        self
    }
}

impl Sub<&OxideMat> for OxideMat {
    type Output = OxideMat;
    fn sub(mut self, rhs: &OxideMat) -> Self::Output {
        SubAssign::sub_assign(&mut self, rhs);
        self
    }
}

// ============================================================
// DenseMatrix
// ============================================================

impl DenseMatrix for OxideMat {
    fn resize_cols(&mut self, new_ncols: IndexType) {
        if new_ncols == self.ncols {
            return;
        }
        let nbatch = self.context.nbatch();
        let nrows = self.nrows;
        let old_ncols = self.ncols;
        let cols_to_copy = old_ncols.min(new_ncols);
        let old_batch_elems = nrows * old_ncols;
        let new_batch_elems = nrows * new_ncols;
        // zeroed, so the columns beyond `cols_to_copy` need no separate memset
        let new_data = DeviceBuffer::zeroed(&self.context.stream, new_batch_elems * nbatch)
            .expect("Failed to allocate resized matrix");
        let elements_per_batch = nrows * cols_to_copy;
        if elements_per_batch > 0 {
            for b in 0..nbatch {
                copy_at(
                    &self.context.stream,
                    &new_data,
                    b * new_batch_elems,
                    &self.data,
                    b * old_batch_elems,
                    elements_per_batch,
                )
                .expect("Failed to copy data during resize_cols");
            }
        }
        self.data = new_data;
        self.ncols = new_ncols;
    }

    fn from_vec(nrows: IndexType, ncols: IndexType, data: Vec<Self::T>, ctx: Self::C) -> Self {
        assert_eq!(data.len(), nrows * ncols * ctx.nbatch());
        let device_data =
            DeviceBuffer::from_host(&ctx.stream, &data).expect("Failed to copy host to device");
        Self {
            nrows,
            ncols,
            data: device_data,
            context: ctx,
        }
    }

    fn get_index(&self, i: IndexType, j: IndexType) -> Self::T {
        assert_eq!(
            self.context.nbatch(),
            1,
            "get_index not supported for batched matrices"
        );
        self.get_index_impl(i, j)
    }

    fn set_index(&mut self, i: IndexType, j: IndexType, value: Self::T) {
        assert_eq!(
            self.context.nbatch(),
            1,
            "set_index not supported for batched matrices"
        );
        self.set_index_impl(i, j, value);
    }

    fn set_index_batch(&mut self, batch: IndexType, i: IndexType, j: IndexType, value: Self::T) {
        // batches are contiguous blocks of `ncols` columns, so batch `batch`'s column `j` is
        // physical column `batch * ncols + j` (see `Matrix::zeros`)
        let j = batch * self.ncols + j;
        self.set_index_impl(i, j, value);
    }

    fn column(&self, i: usize) -> <Self::V as Vector>::View<'_> {
        OxideVecRef {
            data: &self.data,
            context: self.context.clone(),
            nstates: self.nrows,
            stride: self.batch_len(),
            col_offset: i * self.nrows,
        }
    }

    fn column_mut(&mut self, i: usize) -> <Self::V as Vector>::ViewMut<'_> {
        let context = self.context.clone();
        let nrows = self.nrows;
        let stride = self.batch_len();
        OxideVecMut {
            data: &mut self.data,
            context,
            nstates: nrows,
            stride,
            col_offset: i * nrows,
        }
    }

    fn mul_cols_by(&mut self, ncols: IndexType, rhs: &[Self::T]) {
        assert!(
            ncols <= self.ncols,
            "mul_cols_by: column range out of bounds"
        );
        assert_eq!(
            rhs.len(),
            ncols * ncols,
            "mul_cols_by: rhs must hold ncols * ncols values"
        );
        assert!(
            ncols <= MAX_SMALL_COLS,
            "mul_cols_by: ncols exceeds MAX_SMALL_COLS"
        );
        let nrows = self.nrows;
        let nbatch = self.context.nbatch();
        if ncols == 0 || nrows == 0 || nbatch == 0 {
            // a launch with a zero grid dimension is an error, where the CPU
            // backends simply iterate over nothing
            return;
        }
        let ctx = self.context.clone();
        let mut kernel_rhs = [0.0f64; MAX_SMALL_COLS * MAX_SMALL_COLS];
        kernel_rhs[..rhs.len()].copy_from_slice(rhs);
        let mut dest = self.operand_mut();
        let dest_stride = dest.stride;
        let n = (nrows * nbatch) as u32;
        let cfg = OxideContext::config_1d(n);
        let m = &ctx.module;
        let p = m.prepare_mul_cols_by(cfg).expect("prepare mul_cols_by");
        m.mul_cols_by(
            &ctx.stream,
            &p,
            &mut dest.window,
            kernel_rhs,
            n,
            ncols as u32,
            nrows as u32,
            dest_stride,
        )
        .expect("launch mul_cols_by");
    }

    fn update_backward_diff(&mut self, order: IndexType, d: &Self::V) {
        assert!(order + 2 < self.ncols, "order out of bounds");
        let ctx = self.context.clone();
        ctx.assert_broadcastable_into(d.context.nbatch(), "update_backward_diff");
        let nrows = self.nrows;
        let nbatch = ctx.nbatch();
        if nrows == 0 {
            return;
        }
        let src = d.operand();
        let mut dest = self.operand_mut();
        let dest_stride = dest.stride;
        let n = (nrows * nbatch) as u32;
        let cfg = OxideContext::config_1d(n);
        let m = &ctx.module;
        let p = m
            .prepare_backward_diff_update(cfg)
            .expect("prepare backward_diff_update");
        m.backward_diff_update(
            &ctx.stream,
            &p,
            &mut dest.window,
            &src.window,
            order as u32,
            n,
            nrows as u32,
            dest_stride,
            src.stride,
            src.nbatch,
            nbatch as u32,
        )
        .expect("launch backward_diff_update");
    }

    fn gemv_cols(
        &self,
        start: IndexType,
        end: IndexType,
        alpha: Self::T,
        x: &[Self::T],
        beta: Self::T,
        y: &mut Self::V,
    ) {
        assert!(start <= end, "gemv_cols: column range start > end");
        assert!(end <= self.ncols, "gemv_cols: column range out of bounds");
        let nc = end - start;
        assert!(
            nc <= MAX_SMALL_COLS,
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
            if beta == 0.0 {
                y.fill(0.0);
            } else if beta != 1.0 {
                y.mul_assign(Scale(beta));
            }
            return;
        }
        let nrows = self.nrows;
        if nrows == 0 {
            return;
        }
        let ctx = self.context.clone();
        let y_nbatch = y.context.nbatch();
        let mut weights = [0.0f64; MAX_SMALL_COLS];
        weights[..nc].copy_from_slice(&x[..nc]);
        // the column range's start rides in the window, so the kernel indexes
        // from column zero
        let mat = self.cols_operand(start, nc);
        let mut dest = y.operand_mut();
        let y_nstates = dest.stride;
        let n = y_nstates * dest.nbatch;
        let cfg = OxideContext::config_1d(n);
        let m = &ctx.module;
        let p = m.prepare_gemv_cols(cfg).expect("prepare gemv_cols");
        m.gemv_cols(
            &ctx.stream,
            &p,
            &mut dest.window,
            &mat.window,
            weights,
            nc as u32,
            alpha,
            beta,
            n,
            y_nstates,
            nrows as u32,
            mat.stride,
            mat.nbatch,
            y_nbatch as u32,
        )
        .expect("launch gemv_cols");
    }
}

// ============================================================
// Matrix
// ============================================================

impl Matrix for OxideMat {
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

    fn zeros(nrows: IndexType, ncols: IndexType, ctx: Self::C) -> Self {
        let data = DeviceBuffer::zeroed(&ctx.stream, nrows * ncols * ctx.nbatch())
            .expect("Failed to allocate device memory");
        Self {
            data,
            context: ctx,
            nrows,
            ncols,
        }
    }

    fn new_from_sparsity(
        nrows: IndexType,
        ncols: IndexType,
        _sparsity: Option<Self::Sparsity>,
        ctx: Self::C,
    ) -> Self {
        Self::zeros(nrows, ncols, ctx)
    }

    fn copy_from(&mut self, other: &Self) {
        let ctx = self.context.clone();
        ctx.assert_broadcastable_into(other.context.nbatch(), "copy_from");
        let n = self.batch_len();
        let src = other.operand();
        let dest = self.operand_mut();
        launch_assign(&ctx, AssignOp::Copy, dest, n, &src);
    }

    fn gather(&mut self, other: &Self, indices: &<Self::V as Vector>::Index) {
        let ctx = self.context.clone();
        ctx.assert_broadcastable_into(other.context.nbatch(), "gather");
        let nindices = indices.len();
        if nindices == 0 {
            return;
        }
        let src = other.operand();
        let mut dest = self.operand_mut();
        let nindices_u32 = nindices as u32;
        let (nbatch_u32, dest_stride) = (dest.nbatch, dest.stride);
        let n = nindices_u32 * nbatch_u32;
        let cfg = OxideContext::config_1d(n);
        let m = &ctx.module;
        let p = m.prepare_vec_gather(cfg).expect("prepare vec_gather");
        m.vec_gather(
            &ctx.stream,
            &p,
            &mut dest.window,
            &src.window,
            &indices.data,
            n,
            nindices_u32,
            dest_stride,
            src.stride,
            src.nbatch,
            nbatch_u32,
        )
        .expect("launch vec_gather");
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
        let ctx = self.context.clone();
        ctx.assert_broadcastable_into(data.context.nbatch(), "set_data_with_indices");
        let nindices = dst_indices.len();
        if nindices == 0 {
            return;
        }
        let src = data.operand();
        let mut dest = self.operand_mut();
        let nindices_u32 = nindices as u32;
        let (nbatch_u32, dest_stride) = (dest.nbatch, dest.stride);
        let n = nindices_u32 * nbatch_u32;
        let cfg = OxideContext::config_1d(n);
        let m = &ctx.module;
        let p = m
            .prepare_mat_set_data_with_indices(cfg)
            .expect("prepare mat_set_data_with_indices");
        m.mat_set_data_with_indices(
            &ctx.stream,
            &p,
            &mut dest.window,
            &src.window,
            &dst_indices.data,
            &src_indices.data,
            n,
            nindices_u32,
            dest_stride,
            src.stride,
            src.nbatch,
            nbatch_u32,
        )
        .expect("launch mat_set_data_with_indices");
    }

    fn add_column_to_vector(&self, j: IndexType, v: &mut Self::V) {
        let ctx = v.context.clone();
        // `v` is the destination, so it carries the launch's batch count and
        // the matrix broadcasts over it (as on the CPU backends)
        ctx.assert_broadcastable_into(self.context.nbatch(), "add_column_to_vector");
        let nrows = self.nrows;
        // column `j` as an operand is just a window into the matrix, so this is
        // a plain axpy
        let col = self.cols_operand(j, 1);
        let dest = v.operand_mut();
        launch_axpy(&ctx, dest, nrows, 1.0, &col, 1.0);
    }

    fn add_columns_to_batched_vector(&self, v: &mut Self::V) {
        let nrows = self.nrows;
        let ncols = self.ncols;
        let nbatch = self.context.nbatch();
        assert_eq!(v.len(), nrows, "row count mismatch");
        assert_eq!(
            v.context.nbatch(),
            nbatch * ncols,
            "batch count mismatch: the destination holds one lane per (batch, column)"
        );
        // batch `b` column `j` lives at offset `(b * ncols + j) * nrows`, which is exactly the
        // destination lane's offset, so the matrix *is* a vector of `nbatch * ncols` lanes and
        // this is a plain batched axpy over the whole buffer
        let columns_as_lanes = OxideVecRef {
            data: &self.data,
            context: v.context.clone(),
            nstates: nrows,
            stride: nrows,
            col_offset: 0,
        };
        v.axpy_v(1.0, &columns_as_lanes, 1.0);
    }

    fn gemv(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V) {
        let nbatch = self.context.nbatch();
        let x_nbatch = x.context.nbatch();
        let y_nbatch = y.context.nbatch();
        // `y` is the destination, so it carries the batch count of the result
        y.context.assert_broadcastable_into(nbatch, "gemv");
        y.context.assert_broadcastable_into(x_nbatch, "gemv");
        if y_nbatch == 0 {
            return;
        }
        let (nrows, ncols, batch_len) = (self.nrows, self.ncols, self.batch_len());
        let (a, xp, yp) = (
            self.data.cu_deviceptr(),
            x.data.cu_deviceptr(),
            y.data.cu_deviceptr(),
        );
        let elem = size_of::<f64>() as u64;
        match GemvPlan::choose(nrows, ncols, nbatch, x_nbatch, y_nbatch) {
            GemvPlan::Strided {
                k,
                count,
                stride_a,
                stride_x,
            } => {
                for j in 0..k {
                    // sub-batch `j` starts at result lane `j`, so each operand starts at the lane
                    // that one reads
                    let a_off = broadcast_batch(j, nbatch, y_nbatch) * batch_len;
                    let x_off = broadcast_batch(j, x_nbatch, y_nbatch) * ncols;
                    self.context.gemv_batched(
                        nrows,
                        ncols,
                        alpha,
                        beta,
                        a + a_off as u64 * elem,
                        stride_a,
                        xp + x_off as u64 * elem,
                        stride_x,
                        yp + (j * nrows) as u64 * elem,
                        (k * nrows) as i64,
                        count,
                    );
                }
            }
            GemvPlan::Gemm { k } => self.context.gemm_batched(
                nrows,
                k,
                ncols,
                alpha,
                beta,
                a,
                batch_len as i64,
                xp,
                (k * ncols) as i64,
                yp,
                (k * nrows) as i64,
                nbatch,
            ),
        }
    }

    fn from_diagonal(v: &Self::V) -> Self {
        let ctx = v.context.clone();
        let nbatch = ctx.nbatch();
        let n = v.len();
        let mut ret = Self::zeros(n, n, ctx.clone());
        if n == 0 {
            return ret;
        }
        let diag = v.operand();
        let mut dest = ret.operand_mut();
        let dest_stride = dest.stride;
        let total = (n * nbatch) as u32;
        let cfg = OxideContext::config_1d(total);
        let m = &ctx.module;
        let p = m
            .prepare_mat_from_diagonal(cfg)
            .expect("prepare mat_from_diagonal");
        m.mat_from_diagonal(
            &ctx.stream,
            &p,
            &mut dest.window,
            &diag.window,
            total,
            n as u32,
            dest_stride,
            diag.stride,
            diag.nbatch,
            nbatch as u32,
        )
        .expect("launch mat_from_diagonal");
        ret
    }

    fn set_column(&mut self, j: IndexType, v: &Self::V) {
        let ctx = self.context.clone();
        ctx.assert_broadcastable_into(v.context.nbatch(), "set_column");
        let nrows = self.nrows;
        assert_eq!(
            v.len(),
            nrows,
            "Column length mismatch: {} != {}",
            v.len(),
            nrows
        );
        // column `j` is a window, so this is a plain copy
        let src = v.operand();
        let dest = self.cols_operand_mut(j, 1);
        launch_assign(&ctx, AssignOp::Copy, dest, nrows, &src);
    }

    /// `self = x + beta * y`
    fn scale_add_and_assign(&mut self, x: &Self, beta: Self::T, y: &Self) {
        let ctx = self.context.clone();
        ctx.assert_broadcastable_into(x.context.nbatch(), "scale_add_and_assign_x");
        ctx.assert_broadcastable_into(y.context.nbatch(), "scale_add_and_assign_y");
        let n = self.batch_len();
        if n == 0 {
            return;
        }
        let x_op = x.operand();
        let y_op = y.operand();
        let mut dest = self.operand_mut();
        let nbatch_u32 = dest.nbatch;
        let total = n as u32 * nbatch_u32;
        let cfg = OxideContext::config_1d(total);
        let m = &ctx.module;
        let p = m
            .prepare_mat_scale_add_assign(cfg)
            .expect("prepare mat_scale_add_assign");
        m.mat_scale_add_assign(
            &ctx.stream,
            &p,
            &mut dest.window,
            &x_op.window,
            &y_op.window,
            beta,
            total,
            n as u32,
            x_op.stride,
            x_op.nbatch,
            y_op.stride,
            y_op.nbatch,
            nbatch_u32,
        )
        .expect("launch mat_scale_add_assign");
    }

    fn partition_indices_by_zero_diagonal(
        &self,
    ) -> (<Self::V as Vector>::Index, <Self::V as Vector>::Index) {
        let diagonal = self.diagonal().clone_as_vec();
        let (zero_indices, nonzero_indices) = (0..self.nrows).fold(
            (Vec::new(), Vec::new()),
            |(mut zero_indices, mut nonzero_indices), i| {
                if diagonal[i] == 0.0 {
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

    fn triplet_iter(
        &self,
    ) -> (
        impl Iterator<Item = (IndexType, IndexType)> + '_,
        impl Iterator<Item = Self::T> + '_,
    ) {
        let nrows = self.nrows;
        let ncols = self.ncols;
        let nbatch = self.context.nbatch();
        let data = self
            .data
            .to_host_vec(&self.context.stream)
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
        values: Vec<Self::T>,
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
        let mut m = vec![0.0f64; nrows * ncols * nbatch];
        for b in 0..nbatch {
            let batch_offset = b * nrows * ncols;
            for (k, &(i, j)) in indices.iter().enumerate() {
                if i >= nrows || j >= ncols {
                    return Err(matrix_error!(IndexOutOfBounds));
                }
                m[batch_offset + i + j * nrows] = values[b * nnz + k];
            }
        }
        Ok(<Self as DenseMatrix>::from_vec(nrows, ncols, m, ctx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shapes `GemvPlan` has to tell apart. Correctness of each variant is covered by the
    /// generated `gemv` tests below; this pins which one a shape picks.
    #[test]
    fn test_gemv_plan_cuda_oxide() {
        let strided = |k, count, stride_a, stride_x| GemvPlan::Strided {
            k,
            count,
            stride_a,
            stride_x,
        };
        // 4 lanes of a 200x200 matrix against 4 lanes of x: one launch, a lane per result
        assert_eq!(
            GemvPlan::choose(200, 200, 4, 4, 4),
            strided(1, 4, 200 * 200, 200)
        );
        // one matrix, one x, broadcast over 4 result lanes: one launch, nothing striding
        assert_eq!(GemvPlan::choose(200, 200, 1, 1, 4), strided(1, 4, 0, 0));
        // grouped matrix: 2 lanes over 8 results, x per lane
        assert_eq!(GemvPlan::choose(100, 100, 2, 8, 8), GemvPlan::Gemm { k: 4 });
        // the same grouping, but a matrix too large for a 4-column gemm tile: 4 launches of 2
        // lanes, the matrix striding once per sub-batch lane and x four times
        assert_eq!(
            GemvPlan::choose(500, 500, 2, 8, 8),
            strided(4, 2, 500 * 500, 4 * 500)
        );
        // wide groups always go through the gemm
        assert_eq!(
            GemvPlan::choose(500, 500, 2, 32, 32),
            GemvPlan::Gemm { k: 16 }
        );
        // grouped x against a matrix lane per result: the mirror image, 4 launches of 2 lanes
        assert_eq!(
            GemvPlan::choose(500, 500, 8, 2, 8),
            strided(4, 2, 4 * 500 * 500, 500)
        );
        // grouped matrix, x broadcast from one lane: no gemm (x has no columns to multiply), so
        // 4 sub-batches of 2 lanes with x held still
        assert_eq!(
            GemvPlan::choose(500, 500, 2, 1, 8),
            strided(4, 2, 500 * 500, 0)
        );
        // both grouped, different widths: `lcm(4, 2) = 4` sub-batches of 2 lanes
        assert_eq!(
            GemvPlan::choose(500, 500, 2, 4, 8),
            strided(4, 2, 500 * 500, 2 * 500)
        );
    }

    super::super::generate_matrix_tests_nonbatched!(cuda_oxide, OxideMat);

    super::super::generate_matrix_tests_batched!(
        cuda_oxide,
        OxideMat,
        OxideContext::default(),
        OxideContext::default().with_nbatch(2)
    );

    super::super::generate_dense_matrix_tests_nonbatched!(cuda_oxide, OxideMat);

    super::super::generate_dense_matrix_tests_batched!(
        cuda_oxide,
        OxideMat,
        OxideContext::default(),
        OxideContext::default().with_nbatch(2)
    );
}
