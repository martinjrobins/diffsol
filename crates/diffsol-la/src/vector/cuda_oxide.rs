//! Vectors for the `cuda-oxide` backend.
//!
//! The layout, batching and broadcast rules are the same as the `cuda`
//! backend's [`CudaVec`](crate::vector::cuda::CudaVec); what differs is that
//! device memory is a [`cuda_core::DeviceBuffer`] and the kernels are the Rust
//! ones in [`crate::cuda_oxide_kernels`].
//!
//! The backend is `f64`-only, as the `cuda` one effectively is: its kernels are
//! all `_f64` and `ScalarCuda` has a single implementation. Dropping the type
//! parameter also keeps the launch calls free of turbofish, because
//! `#[cuda_module]` generates one host method per kernel rather than a generic
//! one.

use std::fmt::{self, Debug};
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Sub, SubAssign};

use cuda_core::{
    simt::memory::{memcpy_dtoh_async, memset_d8_async},
    DeviceBuffer, LaunchConfig1D,
};
use cuda_device::atomic::DeviceAtomicU64;
use cuda_host::RowWidth;

use super::{VectorIndex, VectorView, VectorViewMut};
use crate::context::broadcast_batch;
use crate::context::cuda_oxide::{copy_at, read_at, write_at};
use crate::cuda_oxide_kernels::{BLOCK_SIZE, SMALL_NSTATES};
use crate::{
    Context, DefaultDenseMatrix, IndexType, OxideContext, OxideMat, Scale, Vector, VectorCommon,
};

/// Dense vector in GPU memory.
///
/// # Data layout with batching
///
/// When `nbatch > 1`, data is a flat contiguous array of `nstates * nbatch`
/// elements. Batch *b* occupies `[b * nstates, (b+1) * nstates)`. [`len`](Vector::len)
/// returns the per-batch length `nstates`.
pub struct OxideVec {
    pub(crate) data: DeviceBuffer<f64>,
    pub(crate) context: OxideContext,
}

/// Integer indices in GPU memory, for gather/scatter. Shared across batches.
pub struct OxideIndex {
    pub(crate) data: DeviceBuffer<i32>,
    pub(crate) context: OxideContext,
}

/// Immutable, possibly strided view of a vector's device memory.
///
/// `stride` is the distance between batches and `col_offset` where each
/// batch's `nstates` elements start within it, which is what makes a column of
/// a batched matrix a zero-copy vector view.
pub struct OxideVecRef<'a> {
    pub(crate) data: &'a DeviceBuffer<f64>,
    pub(crate) context: OxideContext,
    pub(crate) nstates: IndexType,
    pub(crate) stride: IndexType,
    pub(crate) col_offset: IndexType,
}

/// Mutable counterpart of [`OxideVecRef`].
pub struct OxideVecMut<'a> {
    pub(crate) data: &'a mut DeviceBuffer<f64>,
    pub(crate) context: OxideContext,
    pub(crate) nstates: IndexType,
    pub(crate) stride: IndexType,
    pub(crate) col_offset: IndexType,
}

// `DeviceBuffer` is not `Debug`, and `VectorCommon` requires it. Print the
// shape rather than the contents: reading the contents would need a device
// copy.
impl Debug for OxideVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OxideVec")
            .field("nstates", &self.len())
            .field("nbatch", &self.context.nbatch())
            .finish()
    }
}
impl Debug for OxideIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OxideIndex")
            .field("len", &self.data.len())
            .finish()
    }
}
impl Debug for OxideVecRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OxideVecRef")
            .field("nstates", &self.nstates)
            .field("stride", &self.stride)
            .field("col_offset", &self.col_offset)
            .field("nbatch", &self.context.nbatch())
            .finish()
    }
}
impl Debug for OxideVecMut<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OxideVecMut")
            .field("nstates", &self.nstates)
            .field("stride", &self.stride)
            .field("col_offset", &self.col_offset)
            .field("nbatch", &self.context.nbatch())
            .finish()
    }
}

impl Clone for OxideVec {
    fn clone(&self) -> Self {
        let mut data = DeviceBuffer::zeroed(&self.context.stream, self.data.len())
            .expect("Failed to allocate device memory");
        data.copy_from_device_async(&self.data, &self.context.stream)
            .expect("Failed to copy device to device");
        Self {
            data,
            context: self.context.clone(),
        }
    }
}

impl Clone for OxideIndex {
    fn clone(&self) -> Self {
        let mut data = DeviceBuffer::zeroed(&self.context.stream, self.data.len())
            .expect("Failed to allocate device memory");
        data.copy_from_device_async(&self.data, &self.context.stream)
            .expect("Failed to copy device to device");
        Self {
            data,
            context: self.context.clone(),
        }
    }
}

// ============================================================
// Windows: the (offset, stride, nstates, nbatch) view of a buffer that a
// kernel launch actually needs.
// ============================================================

/// The handle a launch takes for `nbatch` batches of `nstates` elements
/// `stride` apart, starting at element `col_offset` of `data`.
///
/// It has to be a `DeviceBuffer`: the generated sync launchers take slice
/// parameters as `&DeviceBuffer<T>` / `&mut DeviceBuffer<T>`, and the
/// destination kernels index `DisjointSlice` by thread, which is only in
/// bounds because the handle starts at the operand's first element.
///
/// It owns nothing -- the caller's [`Operand`] / [`OperandMut`] holds the
/// borrow of the buffer it points into, and `ManuallyDrop` is what keeps the
/// ownership transfer [`DeviceBuffer::from_raw_parts`] documents from turning
/// into a double free.
fn launch_window(
    data: &DeviceBuffer<f64>,
    col_offset: IndexType,
    stride: IndexType,
    nstates: IndexType,
    nbatch: IndexType,
) -> ManuallyDrop<DeviceBuffer<f64>> {
    // the highest element any lane touches is `(nbatch - 1) * stride + nstates - 1`
    let len = (nbatch - 1) * stride + nstates;
    assert!(
        col_offset + len <= data.len(),
        "operand {col_offset}..{} out of bounds for buffer of {}",
        col_offset + len,
        data.len()
    );
    let ptr = data.cu_deviceptr() + (col_offset * std::mem::size_of::<f64>()) as u64;
    // SAFETY: `ptr` is inside `data`'s allocation, so it is a `cuMemAlloc*`
    // pointer in `data`'s context with at least `len * size_of::<f64>()` bytes
    // behind it.
    ManuallyDrop::new(unsafe { DeviceBuffer::from_raw_parts(ptr, len, data.context().clone()) })
}

/// One source operand of a launch: where its data starts, how far apart its
/// batches are, and how many batches it has.
pub(crate) struct Operand<'a> {
    pub(crate) window: ManuallyDrop<DeviceBuffer<f64>>,
    _base: PhantomData<&'a DeviceBuffer<f64>>,
    pub(crate) stride: u32,
    pub(crate) nbatch: u32,
}

/// The destination operand of a launch.
///
/// Same handle as [`Operand`], built from `&'a mut DeviceBuffer<f64>`: the
/// launchers take a destination as `&mut DeviceBuffer<T>` and the device write
/// wants exclusive authority over the range, so the borrow it is built from
/// says so too.
pub(crate) struct OperandMut<'a> {
    pub(crate) window: ManuallyDrop<DeviceBuffer<f64>>,
    _base: PhantomData<&'a mut DeviceBuffer<f64>>,
    pub(crate) stride: u32,
    pub(crate) nbatch: u32,
}

impl<'a> Operand<'a> {
    pub(crate) fn new(
        data: &'a DeviceBuffer<f64>,
        col_offset: IndexType,
        stride: IndexType,
        nstates: IndexType,
        nbatch: IndexType,
    ) -> Self {
        Self {
            window: launch_window(data, col_offset, stride, nstates, nbatch),
            _base: PhantomData,
            stride: stride as u32,
            nbatch: nbatch as u32,
        }
    }
}

impl<'a> OperandMut<'a> {
    pub(crate) fn new(
        data: &'a mut DeviceBuffer<f64>,
        col_offset: IndexType,
        stride: IndexType,
        nstates: IndexType,
        nbatch: IndexType,
    ) -> Self {
        Self {
            window: launch_window(data, col_offset, stride, nstates, nbatch),
            _base: PhantomData,
            stride: stride as u32,
            nbatch: nbatch as u32,
        }
    }
}

impl OxideVec {
    /// This vector as a launch source.
    pub(crate) fn operand(&self) -> Operand<'_> {
        let nstates = self.len();
        Operand::new(&self.data, 0, nstates, nstates, self.context.nbatch())
    }
    /// This vector as a launch destination.
    pub(crate) fn operand_mut(&mut self) -> OperandMut<'_> {
        let (nstates, nbatch) = (self.len(), self.context.nbatch());
        OperandMut::new(&mut self.data, 0, nstates, nstates, nbatch)
    }
}

impl OxideVecRef<'_> {
    pub(crate) fn len(&self) -> IndexType {
        self.nstates
    }
    pub(crate) fn operand(&self) -> Operand<'_> {
        Operand::new(
            self.data,
            self.col_offset,
            self.stride,
            self.nstates,
            self.context.nbatch(),
        )
    }
}

impl OxideVecMut<'_> {
    pub(crate) fn len(&self) -> IndexType {
        self.nstates
    }
    pub(crate) fn operand(&self) -> Operand<'_> {
        Operand::new(
            self.data,
            self.col_offset,
            self.stride,
            self.nstates,
            self.context.nbatch(),
        )
    }
    pub(crate) fn operand_mut(&mut self) -> OperandMut<'_> {
        let (col_offset, stride, nstates) = (self.col_offset, self.stride, self.nstates);
        let nbatch = self.context.nbatch();
        OperandMut::new(&mut *self.data, col_offset, stride, nstates, nbatch)
    }
}

// ============================================================
// Kernel launches. Every launch in this module goes through one of these, so
// the `prepare_*` / block-shape bookkeeping lives in one place.
//
// `prepare_*` is called per launch rather than cached. That looks wasteful --
// it re-runs 11 driver attribute queries (8 `cuDeviceGetAttribute`, 3
// `cuFuncGetAttribute`), `cuda-core` memoises none of them, and its own docs
// recommend reusing a `PreparedLaunch`. It was measured before assuming so, at
// `nstates = 3, nbatch = 1000` (`timing_breakdown` in the tests below):
//
//        debug          release
//        1.45us          285ns   prepare alone
//        3.19us         2.38us   launch alone, prepared hoisted out of the loop
//        4.49us         2.4us    the whole public path
//
// In release it is ~12% of the op and hoisting it makes no measurable
// difference -- run-to-run variance on the full path is ~+/-0.35us, several
// times the 285ns at stake. The 1.45us that made it look worth fixing was
// debug-build wrapper overhead, not the driver calls.
//
// A cache would have to be per `OxideContext` (augmented equations run at a
// different `nbatch`, so a device-global one would thrash) and several ways
// wide per kernel (`nout`-length vectors give a second config within one
// context), with a fresh cache on `clone_with_nbatch` but a shared one on
// `Clone`. That is a lot of invariant for 285ns.
//
// ponytail: no prepared-launch cache; revisit only if a profile shows
// `prepare_*` mattering, and prefer memoising the attribute queries upstream in
// `cuda-core` -- they are device and function constants.
// ============================================================

/// The `dest op= rhs` family, all sharing one argument list.
#[derive(Copy, Clone)]
pub(crate) enum AssignOp {
    Copy,
    Add,
    Sub,
    SubRev,
    Mul,
    Div,
}

/// `dest op= rhs`, over `nstates` elements of `dest.nbatch` batches.
///
/// `Copy`/`Add`/`Sub` may be given a matrix column as the destination, so they
/// take its stride and write through a raw pointer; the other three can only
/// ever have a contiguous destination and stay on the checked path. See the
/// tier discussion in [`crate::cuda_oxide_kernels`].
pub(crate) fn launch_assign(
    ctx: &OxideContext,
    op: AssignOp,
    mut dest: OperandMut<'_>,
    nstates: IndexType,
    rhs: &Operand<'_>,
) {
    if nstates == 0 {
        return;
    }
    let nstates = nstates as u32;
    let (nbatch, stride) = (dest.nbatch, dest.stride);
    let n = nstates * nbatch;
    let cfg = OxideContext::config_1d(n);
    let m = &ctx.module;
    let stream = &ctx.stream;
    let d = &mut *dest.window;
    macro_rules! contiguous {
        ($kernel:ident, $prepare:ident) => {{
            let p = m
                .$prepare(cfg)
                .expect(concat!("prepare ", stringify!($kernel)));
            m.$kernel(
                stream,
                &p,
                d,
                &rhs.window,
                n,
                nstates,
                rhs.stride,
                rhs.nbatch,
                nbatch,
            )
            .expect(concat!("launch ", stringify!($kernel)))
        }};
    }
    macro_rules! strided {
        ($kernel:ident, $prepare:ident) => {{
            let p = m
                .$prepare(cfg)
                .expect(concat!("prepare ", stringify!($kernel)));
            m.$kernel(
                stream,
                &p,
                d,
                &rhs.window,
                n,
                nstates,
                stride,
                rhs.stride,
                rhs.nbatch,
                nbatch,
            )
            .expect(concat!("launch ", stringify!($kernel)))
        }};
    }
    match op {
        AssignOp::Copy => strided!(vec_copy, prepare_vec_copy),
        AssignOp::Add => strided!(vec_add_assign, prepare_vec_add_assign),
        AssignOp::Sub => strided!(vec_sub_assign, prepare_vec_sub_assign),
        AssignOp::SubRev => contiguous!(vec_sub_assign_rev, prepare_vec_sub_assign_rev),
        AssignOp::Mul => contiguous!(vec_mul_assign, prepare_vec_mul_assign),
        AssignOp::Div => contiguous!(vec_div_assign, prepare_vec_div_assign),
    }
}

/// `ret = lhs + rhs` or `ret = lhs - rhs`, allocating into `ret`.
pub(crate) fn launch_binary(
    ctx: &OxideContext,
    add: bool,
    mut ret: OperandMut<'_>,
    nstates: IndexType,
    lhs: &Operand<'_>,
    rhs: &Operand<'_>,
) {
    if nstates == 0 {
        return;
    }
    let nstates = nstates as u32;
    let nbatch = ret.nbatch;
    let n = nstates * nbatch;
    let cfg = OxideContext::config_1d(n);
    let m = &ctx.module;
    let stream = &ctx.stream;
    let d = &mut *ret.window;
    if add {
        let p = m.prepare_vec_add(cfg).expect("prepare vec_add");
        m.vec_add(
            stream,
            &p,
            d,
            &lhs.window,
            &rhs.window,
            n,
            nstates,
            lhs.stride,
            lhs.nbatch,
            rhs.stride,
            rhs.nbatch,
            nbatch,
        )
        .expect("launch vec_add");
    } else {
        let p = m.prepare_vec_sub(cfg).expect("prepare vec_sub");
        m.vec_sub(
            stream,
            &p,
            d,
            &lhs.window,
            &rhs.window,
            n,
            nstates,
            lhs.stride,
            lhs.nbatch,
            rhs.stride,
            rhs.nbatch,
            nbatch,
        )
        .expect("launch vec_sub");
    }
}

/// `dest = alpha * x + beta * dest`
pub(crate) fn launch_axpy(
    ctx: &OxideContext,
    mut dest: OperandMut<'_>,
    nstates: IndexType,
    alpha: f64,
    x: &Operand<'_>,
    beta: f64,
) {
    if nstates == 0 {
        return;
    }
    let nstates = nstates as u32;
    let (nbatch, stride) = (dest.nbatch, dest.stride);
    let n = nstates * nbatch;
    let cfg = OxideContext::config_1d(n);
    let m = &ctx.module;
    let p = m.prepare_vec_axpy(cfg).expect("prepare vec_axpy");
    m.vec_axpy(
        &ctx.stream,
        &p,
        &mut dest.window,
        &x.window,
        alpha,
        beta,
        n,
        nstates,
        stride,
        x.stride,
        x.nbatch,
        nbatch,
    )
    .expect("launch vec_axpy");
}

/// `dest *= scalar`
pub(crate) fn launch_mul_assign_scalar(
    ctx: &OxideContext,
    mut dest: OperandMut<'_>,
    nstates: IndexType,
    scalar: f64,
) {
    if nstates == 0 {
        return;
    }
    let nstates = nstates as u32;
    let (nbatch, stride) = (dest.nbatch, dest.stride);
    let n = nstates * nbatch;
    let cfg = OxideContext::config_1d(n);
    let m = &ctx.module;
    let p = m
        .prepare_vec_mul_assign_scalar(cfg)
        .expect("prepare vec_mul_assign_scalar");
    m.vec_mul_assign_scalar(
        &ctx.stream,
        &p,
        &mut dest.window,
        scalar,
        n,
        nstates,
        stride,
        nbatch,
    )
    .expect("launch vec_mul_assign_scalar");
}

/// `ret = scalar * src`
pub(crate) fn launch_mul_scalar(
    ctx: &OxideContext,
    mut ret: OperandMut<'_>,
    nstates: IndexType,
    src: &Operand<'_>,
    scalar: f64,
) {
    if nstates == 0 {
        return;
    }
    let nstates = nstates as u32;
    let nbatch = ret.nbatch;
    let n = nstates * nbatch;
    let cfg = OxideContext::config_1d(n);
    let m = &ctx.module;
    let p = m
        .prepare_vec_mul_scalar(cfg)
        .expect("prepare vec_mul_scalar");
    m.vec_mul_scalar(
        &ctx.stream,
        &p,
        &mut ret.window,
        &src.window,
        scalar,
        n,
        nstates,
        src.stride,
        src.nbatch,
        nbatch,
    )
    .expect("launch vec_mul_scalar");
}

/// `dest = value`
pub(crate) fn launch_fill(
    ctx: &OxideContext,
    mut dest: OperandMut<'_>,
    nstates: IndexType,
    value: f64,
) {
    if nstates == 0 {
        return;
    }
    let n = nstates as u32 * dest.nbatch;
    let cfg = OxideContext::config_1d(n);
    let m = &ctx.module;
    let p = m.prepare_vec_fill(cfg).expect("prepare vec_fill");
    m.vec_fill(&ctx.stream, &p, &mut dest.window, value, n)
        .expect("launch vec_fill");
}

/// Where one of the reduction kernels writes, and the geometry it runs at.
///
/// Which arm a reduction gets is decided by `nstates` against
/// [`SMALL_NSTATES`]; see the section comment above the kernels.
enum ReduceOut<'a> {
    /// A block owns `cols_per_block` whole lanes, so a lane sum is complete
    /// inside one block and the cross-lane maximum is taken on the device.
    Small {
        out: &'a DeviceBuffer<DeviceAtomicU64>,
        cfg: LaunchConfig1D,
        cols_per_block: u32,
    },
    /// One block per lane, or `blocks_per_lane` of them sharing a lane when
    /// there are too few lanes to fill the device. A block that owns its whole
    /// lane folds into `out` itself; the shared-lane case writes `partials` for
    /// `lane_sum_max`, so the kernel takes both.
    Large {
        out: &'a DeviceBuffer<DeviceAtomicU64>,
        partials: &'a mut DeviceBuffer<f64>,
        cfg: LaunchConfig1D,
        blocks_per_lane: u32,
    },
}

/// Runs one of the reductions and returns the largest per-lane sum.
///
/// Both shapes leave the maximum in a single device `u64` -- the IEEE bit
/// pattern, which is monotonic for the non-negative sums these kernels produce
/// -- so the host reads back 8 bytes whatever `nbatch` is. `sqrt`, `powf(1/k)`
/// and `/nstates` are all monotonic on non-negative values, so callers apply
/// theirs to this maximum instead of per lane.
///
/// The buffers come from the context's [`ReduceScratch`] rather than a fresh
/// allocation, which is what makes this cost the kernel and one 8-byte readback
/// instead of a `cuMemAlloc`/`cuMemFree` pair.
fn reduce<F>(ctx: &OxideContext, nstates: IndexType, nbatch: IndexType, launch: F) -> f64
where
    F: FnOnce(ReduceOut<'_>),
{
    let stream = &ctx.stream;
    let mut guard = ctx.scratch.lock().expect("Reduction scratch poisoned");
    let scratch = &mut *guard;
    // SAFETY: `out` is one live 8-byte cell, and zero is the identity for the
    // `atom.max` the kernels fold into it.
    unsafe {
        memset_d8_async(
            scratch.out.cu_deviceptr(),
            0,
            scratch.out.num_bytes(),
            stream.cu_stream(),
        )
    }
    .expect("Failed to clear reduction output");
    let out = &scratch.out;

    if nstates as u32 <= SMALL_NSTATES {
        let cols_per_block = BLOCK_SIZE / nstates as u32;
        launch(ReduceOut::Small {
            out,
            cfg: OxideContext::config_1d_blocks((nbatch as u32).div_ceil(cols_per_block)),
            cols_per_block,
        });
    } else {
        // Fill the device: one block per lane when there are enough lanes,
        // several sharing a lane when there are not -- `nbatch = 1` with a long
        // state vector would otherwise reduce on a single SM. Capped at the
        // blocks a lane can actually keep busy.
        let blocks_per_lane = (ctx.target_blocks / nbatch as u32)
            .max(1)
            .min((nstates as u32).div_ceil(BLOCK_SIZE));
        let lanes_per_pass = (nbatch as u32).min(ctx.target_blocks.div_ceil(blocks_per_lane));
        // The kernels write `blocks_per_lane * nbatch` slots, which is at most
        // `max(nbatch, target_blocks)` -- how `ReduceScratch` sizes `partials`.
        let partials = &mut scratch.partials;
        launch(ReduceOut::Large {
            out,
            partials: &mut *partials,
            cfg: OxideContext::config_1d_blocks(blocks_per_lane * lanes_per_pass),
            blocks_per_lane,
        });
        if blocks_per_lane > 1 {
            // lanes spanned several blocks, so no block wrote a whole lane sum
            let m = &ctx.module;
            let cfg = OxideContext::config_1d(nbatch as u32);
            let p = m.prepare_lane_sum_max(cfg).expect("prepare lane_sum_max");
            m.lane_sum_max(stream, &p, out, partials, nbatch as u32, blocks_per_lane)
                .expect("launch lane_sum_max");
        }
    }

    let mut bits = [0u64];
    // SAFETY: `out` is one `u64`-sized cell and `bits` is one `u64`; the
    // synchronize completes the copy before it is read.
    unsafe {
        memcpy_dtoh_async(
            bits.as_mut_ptr(),
            scratch.out.cu_deviceptr(),
            scratch.out.num_bytes(),
            stream.cu_stream(),
        )
    }
    .and_then(|()| stream.synchronize())
    .expect("Failed to copy reduction output");
    f64::from_bits(bits[0])
}

impl OxideContext {
    /// `max_b (sum_i |x_i|^k)^(1/k)`, over the batches of `x`.
    fn lk_norm(&self, x: &Operand<'_>, nstates: IndexType, nbatch: IndexType, k: i32) -> f64 {
        let n = nstates as u32;
        let nb = nbatch as u32;
        let m = &self.module;
        let max_sum = reduce(self, nstates, nbatch, |out| match out {
            ReduceOut::Small {
                out,
                cfg,
                cols_per_block,
            } => {
                if k == 2 {
                    let p = m
                        .prepare_vec_norm_small(cfg)
                        .expect("prepare vec_norm_small");
                    m.vec_norm_small(
                        &self.stream,
                        &p,
                        out,
                        &x.window,
                        n,
                        nb,
                        x.stride,
                        cols_per_block,
                    )
                    .expect("launch vec_norm_small");
                } else {
                    let p = m
                        .prepare_vec_norm_lk_small(cfg)
                        .expect("prepare vec_norm_lk_small");
                    m.vec_norm_lk_small(
                        &self.stream,
                        &p,
                        out,
                        &x.window,
                        n,
                        nb,
                        x.stride,
                        cols_per_block,
                        k,
                    )
                    .expect("launch vec_norm_lk_small");
                }
            }
            ReduceOut::Large {
                out,
                partials,
                cfg,
                blocks_per_lane,
            } => {
                if k == 2 {
                    let p = m.prepare_vec_norm(cfg).expect("prepare vec_norm");
                    m.vec_norm(
                        &self.stream,
                        &p,
                        out,
                        partials,
                        &x.window,
                        n,
                        nb,
                        x.stride,
                        blocks_per_lane,
                    )
                    .expect("launch vec_norm");
                } else {
                    let p = m.prepare_vec_norm_lk(cfg).expect("prepare vec_norm_lk");
                    m.vec_norm_lk(
                        &self.stream,
                        &p,
                        out,
                        partials,
                        &x.window,
                        n,
                        nb,
                        x.stride,
                        blocks_per_lane,
                        k,
                    )
                    .expect("launch vec_norm_lk");
                }
            }
        });
        if k == 2 {
            max_sum.sqrt()
        } else {
            max_sum.powf(1.0 / k as f64)
        }
    }
}

impl DefaultDenseMatrix for OxideVec {
    type M = OxideMat;
}

macro_rules! impl_vector_common {
    ($vec:ty) => {
        impl VectorCommon for $vec {
            type T = f64;
            type C = OxideContext;
            type Inner = DeviceBuffer<f64>;
            fn inner(&self) -> &Self::Inner {
                self.data
            }
        }
    };
    (owned $vec:ty) => {
        impl VectorCommon for $vec {
            type T = f64;
            type C = OxideContext;
            type Inner = DeviceBuffer<f64>;
            fn inner(&self) -> &Self::Inner {
                &self.data
            }
        }
    };
}
impl_vector_common!(owned OxideVec);
impl_vector_common!(OxideVecRef<'_>);
impl_vector_common!(OxideVecMut<'_>);

// ============================================================
// Operators, fanned out over the operand flavours. Mirrors the `cuda`
// backend's macro set, minus the scalar type parameter.
// ============================================================

/// Allocating binary op: `&lhs + &rhs -> OxideVec`.
macro_rules! impl_binary_ref_ref {
    ([$($g:tt)*], $Op:ident, $method:ident, $add:expr, $label:expr, $Lhs:ty, $Rhs:ty) => {
        impl<$($g)*> $Op<$Rhs> for $Lhs {
            type Output = OxideVec;
            fn $method(self, rhs: $Rhs) -> OxideVec {
                let ctx = self.context.clone();
                // neither operand is owned, so the result carries the left-hand side's batch
                // count and `rhs` broadcasts into it
                ctx.assert_broadcastable_into(rhs.context.nbatch(), $label);
                let nstates = self.len();
                let mut ret = OxideVec::zeros(nstates, ctx.clone());
                let ret_op = ret.operand_mut();
                launch_binary(&ctx, $add, ret_op, nstates, &self.operand(), &rhs.operand());
                ret
            }
        }
    };
}

/// In-place assign: `lhs += rhs` / `lhs -= rhs`.
macro_rules! impl_assign {
    ([$($g:tt)*], $Op:ident, $method:ident, $op:expr, $label:expr,
     $Lhs:ty, $RhsRef:ty, $RhsOwned:ty) => {
        impl<$($g)*> $Op<$RhsOwned> for $Lhs {
            fn $method(&mut self, rhs: $RhsOwned) {
                self.$method(&rhs);
            }
        }
        impl<$($g)*> $Op<$RhsRef> for $Lhs {
            fn $method(&mut self, rhs: $RhsRef) {
                let ctx = self.context.clone();
                ctx.assert_broadcastable_into(rhs.context.nbatch(), $label);
                let nstates = self.len();
                let rhs_op = rhs.operand();
                let dest = self.operand_mut();
                launch_assign(&ctx, $op, dest, nstates, &rhs_op);
            }
        }
    };
}

/// `self` is the owned operand, so it is the destination and the in-place op is
/// the whole implementation.
macro_rules! impl_binary_owned_lhs {
    ($Op:ident, $method:ident, $AssignOp:ident, $assign:ident, $Rhs:ty) => {
        impl<'a> $Op<$Rhs> for OxideVec {
            type Output = OxideVec;
            fn $method(mut self, rhs: $Rhs) -> Self::Output {
                $AssignOp::$assign(&mut self, rhs);
                self
            }
        }
    };
}

/// `rhs` is the owned operand, so it is the destination. A commutative op is
/// the in-place op with the operands swapped; a non-commutative one launches
/// the reversed assign kernel (`dest = src - dest`) instead.
macro_rules! impl_binary_owned_rhs {
    (commutes, $Op:ident, $method:ident, $AssignOp:ident, $assign:ident, $Lhs:ty, $label:expr) => {
        impl<'a> $Op<OxideVec> for $Lhs {
            type Output = OxideVec;
            fn $method(self, mut rhs: OxideVec) -> Self::Output {
                $AssignOp::$assign(&mut rhs, self);
                rhs
            }
        }
    };
    (noncommutes, $Op:ident, $method:ident, $AssignOp:ident, $assign:ident, $Lhs:ty, $label:expr) => {
        impl<'a> $Op<OxideVec> for $Lhs {
            type Output = OxideVec;
            fn $method(self, mut rhs: OxideVec) -> Self::Output {
                let ctx = rhs.context.clone();
                ctx.assert_broadcastable_into(self.context.nbatch(), $label);
                let nstates = self.len();
                let lhs_op = self.operand();
                let dest = rhs.operand_mut();
                launch_assign(&ctx, AssignOp::SubRev, dest, nstates, &lhs_op);
                rhs
            }
        }
    };
}

/// Every operand flavour of one operator.
macro_rules! impl_binary_set {
    ($Op:ident, $method:ident, $AssignOp:ident, $assign:ident, $commutes:ident, $add:expr, $label:expr) => {
        impl_binary_owned_lhs!($Op, $method, $AssignOp, $assign, OxideVec);
        impl_binary_owned_lhs!($Op, $method, $AssignOp, $assign, &OxideVec);
        impl_binary_owned_lhs!($Op, $method, $AssignOp, $assign, OxideVecRef<'a>);
        impl_binary_owned_lhs!($Op, $method, $AssignOp, $assign, &OxideVecRef<'a>);
        impl_binary_owned_rhs!($commutes, $Op, $method, $AssignOp, $assign, OxideVecRef<'a>, $label);
        impl_binary_owned_rhs!($commutes, $Op, $method, $AssignOp, $assign, &OxideVec, $label);
        impl_binary_ref_ref!(['a], $Op, $method, $add, $label, OxideVecRef<'a>, &OxideVec);
        impl_binary_ref_ref!(['a, 'b], $Op, $method, $add, $label, OxideVecRef<'a>, OxideVecRef<'b>);
        impl_binary_ref_ref!(['a, 'b], $Op, $method, $add, $label, OxideVecRef<'a>, &OxideVecRef<'b>);
        impl_binary_ref_ref!([], $Op, $method, $add, $label, &OxideVec, &OxideVec);
        impl_binary_ref_ref!(['a], $Op, $method, $add, $label, &OxideVec, OxideVecRef<'a>);
        impl_binary_ref_ref!(['a], $Op, $method, $add, $label, &OxideVec, &OxideVecRef<'a>);
    };
}

/// Both destinations (an owned vector and a mutable view) against both source
/// flavours.
macro_rules! impl_assign_set {
    ($Op:ident, $method:ident, $op:expr, $label:expr) => {
        impl_assign!([], $Op, $method, $op, $label, OxideVec, &OxideVec, OxideVec);
        impl_assign!(['a], $Op, $method, $op, $label, OxideVec, &OxideVecRef<'a>, OxideVecRef<'a>);
        impl_assign!(['a], $Op, $method, $op, $label, OxideVecMut<'a>, &OxideVec, OxideVec);
        impl_assign!(
            ['a, 'b], $Op, $method, $op, $label,
            OxideVecMut<'a>, &OxideVecRef<'b>, OxideVecRef<'b>
        );
    };
}

impl_binary_set!(Add, add, AddAssign, add_assign, commutes, true, "add");
impl_binary_set!(Sub, sub, SubAssign, sub_assign, noncommutes, false, "sub");
impl_assign_set!(AddAssign, add_assign, AssignOp::Add, "add_assign");
impl_assign_set!(SubAssign, sub_assign, AssignOp::Sub, "sub_assign");

// ============================================================
// Scalar multiply / divide
// ============================================================

macro_rules! impl_mul_scalar_alloc {
    ([$($g:tt)*], $lhs:ty) => {
        impl<$($g)*> Mul<Scale<f64>> for $lhs {
            type Output = OxideVec;
            fn mul(self, rhs: Scale<f64>) -> Self::Output {
                let ctx = self.context.clone();
                let nstates = self.len();
                let mut ret = OxideVec::zeros(nstates, ctx.clone());
                let src = self.operand();
                let ret_op = ret.operand_mut();
                launch_mul_scalar(&ctx, ret_op, nstates, &src, rhs.value());
                ret
            }
        }
    };
}

impl_mul_scalar_alloc!([], &OxideVec);
impl_mul_scalar_alloc!(['a], OxideVecRef<'a>);
impl_mul_scalar_alloc!(['a], OxideVecMut<'a>);

macro_rules! impl_mul_assign_scalar {
    ([$($g:tt)*], $ty:ty) => {
        impl<$($g)*> MulAssign<Scale<f64>> for $ty {
            fn mul_assign(&mut self, rhs: Scale<f64>) {
                let ctx = self.context.clone();
                let nstates = self.len();
                let dest = self.operand_mut();
                launch_mul_assign_scalar(&ctx, dest, nstates, rhs.value());
            }
        }
    };
}

impl_mul_assign_scalar!([], OxideVec);
impl_mul_assign_scalar!(['a], OxideVecMut<'a>);

impl Mul<Scale<f64>> for OxideVec {
    type Output = OxideVec;
    fn mul(mut self, rhs: Scale<f64>) -> Self::Output {
        MulAssign::mul_assign(&mut self, rhs);
        self
    }
}

impl Div<Scale<f64>> for OxideVec {
    type Output = OxideVec;
    fn div(self, rhs: Scale<f64>) -> Self::Output {
        self.mul(Scale(1.0 / rhs.value()))
    }
}

// ============================================================
// VectorIndex
// ============================================================

impl VectorIndex for OxideIndex {
    type C = OxideContext;
    fn context(&self) -> &Self::C {
        &self.context
    }
    fn len(&self) -> IndexType {
        self.data.len() as IndexType
    }
    fn zeros(len: IndexType, ctx: Self::C) -> Self {
        let data =
            DeviceBuffer::zeroed(&ctx.stream, len).expect("Failed to allocate device memory");
        Self { data, context: ctx }
    }
    fn clone_as_vec(&self) -> Vec<IndexType> {
        self.data
            .to_host_vec(&self.context.stream)
            .expect("Failed to copy data from device to host")
            .into_iter()
            .map(|x| x as IndexType)
            .collect()
    }
    fn from_vec(v: Vec<IndexType>, ctx: Self::C) -> Self {
        let v: Vec<i32> = v.into_iter().map(|x| x as i32).collect();
        let data = DeviceBuffer::from_host(&ctx.stream, &v).expect("Failed to copy host to device");
        Self { data, context: ctx }
    }
}

// ============================================================
// Vector
// ============================================================

impl Vector for OxideVec {
    type View<'a> = OxideVecRef<'a>;
    type ViewMut<'a> = OxideVecMut<'a>;
    type Index = OxideIndex;

    fn context(&self) -> &Self::C {
        &self.context
    }
    fn inner_mut(&mut self) -> &mut Self::Inner {
        &mut self.data
    }
    fn len(&self) -> IndexType {
        self.data.len() as IndexType / self.context.nbatch()
    }
    fn get_index(&self, index: IndexType) -> Self::T {
        assert_eq!(
            self.context.nbatch(),
            1,
            "get_index not supported for batched vectors"
        );
        let mut out = [0.0f64];
        read_at(&self.context.stream, &self.data, index, &mut out)
            .expect("Failed to copy data from device to host");
        out[0]
    }
    fn set_index(&mut self, index: IndexType, value: Self::T) {
        assert_eq!(
            self.context.nbatch(),
            1,
            "set_index not supported for batched vectors, use fill_index"
        );
        self.fill_index(index, value);
    }
    fn fill_index(&mut self, index: IndexType, value: Self::T) {
        let nbatch = self.context.nbatch();
        let nstates = self.len();
        assert!(index < nstates, "Index out of bounds");
        for b in 0..nbatch {
            write_at(
                &self.context.stream,
                &self.data,
                b * nstates + index,
                &[value],
            )
            .expect("Failed to copy data from host to device");
        }
    }
    fn norm(&self, k: i32) -> Self::T {
        let nbatch = self.context.nbatch();
        let nstates = self.len();
        if nstates == 0 {
            return 0.0;
        }
        self.context.lk_norm(&self.operand(), nstates, nbatch, k)
    }
    fn squared_norm(&self, y: &Self, atol: &Self, rtol: Self::T) -> Self::T {
        self.as_view().squared_norm(y, atol, rtol)
    }

    fn from_vec(v: Vec<Self::T>, ctx: Self::C) -> Self {
        Self::from_slice(&v, ctx)
    }
    fn from_slice(slice: &[Self::T], ctx: Self::C) -> Self {
        let nbatch = ctx.nbatch();
        assert!(
            slice.len() % nbatch == 0,
            "slice length {} must be divisible by nbatch {}",
            slice.len(),
            nbatch
        );
        let data =
            DeviceBuffer::from_host(&ctx.stream, slice).expect("Failed to copy host to device");
        Self { data, context: ctx }
    }
    fn from_element(nstates: usize, value: Self::T, ctx: Self::C) -> Self {
        let mut ret = Self::zeros(nstates, ctx);
        ret.fill(value);
        ret
    }
    fn zeros(nstates: usize, ctx: Self::C) -> Self {
        let total = nstates * ctx.nbatch();
        let data =
            DeviceBuffer::zeroed(&ctx.stream, total).expect("Failed to allocate device memory");
        Self { data, context: ctx }
    }

    fn fill(&mut self, value: Self::T) {
        let ctx = self.context.clone();
        let nstates = self.len();
        let dest = self.operand_mut();
        launch_fill(&ctx, dest, nstates, value);
    }
    fn as_view(&self) -> Self::View<'_> {
        let nstates = self.len();
        OxideVecRef {
            data: &self.data,
            context: self.context.clone(),
            nstates,
            stride: nstates,
            col_offset: 0,
        }
    }
    fn as_view_mut(&mut self) -> Self::ViewMut<'_> {
        let nstates = self.len();
        let context = self.context.clone();
        OxideVecMut {
            data: &mut self.data,
            context,
            nstates,
            stride: nstates,
            col_offset: 0,
        }
    }
    fn get_batch(&self, batch: usize) -> Self::View<'_> {
        let nstates = self.len();
        assert!(batch < self.context.nbatch(), "Batch index out of bounds");
        OxideVecRef {
            data: &self.data,
            context: self.context.clone_with_nbatch(1).unwrap(),
            nstates,
            stride: nstates,
            col_offset: batch * nstates,
        }
    }
    fn get_batch_mut(&mut self, batch: usize) -> Self::ViewMut<'_> {
        let nstates = self.len();
        assert!(batch < self.context.nbatch(), "Batch index out of bounds");
        let context = self.context.clone_with_nbatch(1).unwrap();
        OxideVecMut {
            data: &mut self.data,
            context,
            nstates,
            stride: nstates,
            col_offset: batch * nstates,
        }
    }
    fn copy_from(&mut self, other: &Self) {
        self.copy_from_view(&other.as_view());
    }
    fn copy_from_view(&mut self, other: &Self::View<'_>) {
        let ctx = self.context.clone();
        ctx.assert_broadcastable_into(other.context.nbatch(), "copy_from_view");
        let nstates = self.len();
        let src = other.operand();
        let dest = self.operand_mut();
        launch_assign(&ctx, AssignOp::Copy, dest, nstates, &src);
    }
    fn axpy(&mut self, alpha: Self::T, x: &Self, beta: Self::T) {
        self.axpy_v(alpha, &x.as_view(), beta);
    }
    fn axpy_v(&mut self, alpha: Self::T, x: &Self::View<'_>, beta: Self::T) {
        let ctx = self.context.clone();
        ctx.assert_broadcastable_into(x.context.nbatch(), "axpy_v");
        let nstates = self.len();
        let src = x.operand();
        let dest = self.operand_mut();
        launch_axpy(&ctx, dest, nstates, alpha, &src, beta);
    }
    fn batched_axpy(&mut self, alpha: &Self, x: &Self, beta: Self::T) {
        let ctx = self.context.clone();
        let nbatch = ctx.nbatch();
        assert_eq!(
            alpha.len(),
            1,
            "batched_axpy: alpha must be a batched scalar, with len() == 1"
        );
        assert_eq!(
            alpha.context.nbatch(),
            nbatch,
            "batched_axpy: alpha nbatch must equal self.nbatch()"
        );
        ctx.assert_broadcastable_into(x.context.nbatch(), "batched_axpy");
        let nstates = self.len();
        if nstates == 0 {
            return;
        }
        let nstates_u32 = nstates as u32;
        let src = x.operand();
        // `alpha` is already one contiguous value per lane on the device, which
        // is exactly what the kernel reads -- no staging buffer needed
        let alpha_op = alpha.operand();
        let mut dest = self.operand_mut();
        let nbatch_u32 = dest.nbatch;
        let n = nstates_u32 * nbatch_u32;
        let cfg = OxideContext::config_1d(n);
        let m = &ctx.module;
        let p = m
            .prepare_vec_batched_axpy(cfg)
            .expect("prepare vec_batched_axpy");
        m.vec_batched_axpy(
            &ctx.stream,
            &p,
            &mut dest.window,
            &src.window,
            &alpha_op.window,
            beta,
            n,
            nstates_u32,
            src.stride,
            src.nbatch,
            nbatch_u32,
        )
        .expect("launch vec_batched_axpy");
    }
    fn clone_as_vec(&self) -> Vec<Self::T> {
        self.data
            .to_host_vec(&self.context.stream)
            .expect("Failed to copy data from device to host")
    }
    fn component_mul_assign(&mut self, other: &Self) {
        let ctx = self.context.clone();
        ctx.assert_broadcastable_into(other.context.nbatch(), "component_mul_assign");
        let nstates = self.len();
        let src = other.operand();
        let dest = self.operand_mut();
        launch_assign(&ctx, AssignOp::Mul, dest, nstates, &src);
    }
    fn component_div_assign(&mut self, other: &Self) {
        let ctx = self.context.clone();
        ctx.assert_broadcastable_into(other.context.nbatch(), "component_div_assign");
        let nstates = self.len();
        let src = other.operand();
        let dest = self.operand_mut();
        launch_assign(&ctx, AssignOp::Div, dest, nstates, &src);
    }
    fn root_finding(&self, g1: &Self) -> (bool, Self::T, i32) {
        let nstates = self.len();
        if nstates == 0 {
            return (false, 0.0, -1);
        }
        let ctx = self.context.clone();
        ctx.assert_broadcastable_into(g1.context.nbatch(), "root_finding");
        // as on the CPU backends, the reduction runs over `self`'s batches
        let nbatch = ctx.nbatch();
        assert_eq!(
            nstates,
            g1.len(),
            "Vector length mismatch: {} != {}",
            nstates,
            g1.len()
        );

        let n = nstates as u32;
        let blocks_per_batch = n.div_ceil(BLOCK_SIZE);
        let total = blocks_per_batch as usize * nbatch;
        let stream = &ctx.stream;
        let mut max_vals =
            DeviceBuffer::<f64>::zeroed(stream, total).expect("Failed to allocate max_vals");
        let mut max_idxs =
            DeviceBuffer::<i32>::zeroed(stream, total).expect("Failed to allocate max_idxs");
        let mut flags =
            DeviceBuffer::<i32>::zeroed(stream, total).expect("Failed to allocate flags");
        let g0_op = self.operand();
        let g1_op = g1.operand();
        {
            let cfg = OxideContext::config_2d(n, nbatch as u32);
            let m = &ctx.module;
            let p = m
                .prepare_vec_root_finding(cfg)
                .expect("prepare vec_root_finding");
            m.vec_root_finding(
                stream,
                &p,
                RowWidth::new(&mut max_vals, blocks_per_batch),
                RowWidth::new(&mut max_idxs, blocks_per_batch),
                RowWidth::new(&mut flags, blocks_per_batch),
                &g0_op.window,
                &g1_op.window,
                n,
                nbatch as u32,
                g0_op.stride,
                g1_op.stride,
                g1_op.nbatch,
            )
            .expect("launch vec_root_finding");
        }
        let h_max_vals = max_vals.to_host_vec(stream).expect("copy max_vals");
        let h_max_idxs = max_idxs.to_host_vec(stream).expect("copy max_idxs");
        let h_flags = flags.to_host_vec(stream).expect("copy flags");

        let mut first_result: Option<(bool, f64, i32)> = None;
        for b in 0..nbatch {
            let start = b * blocks_per_batch as usize;
            let end = start + blocks_per_batch as usize;
            let found_root = h_flags[start..end].iter().any(|&f| f != 0);
            let mut max_val = 0.0;
            let mut max_idx = -1;
            for i in start..end {
                if h_max_vals[i] > max_val {
                    max_val = h_max_vals[i];
                    max_idx = h_max_idxs[i];
                }
            }
            let result = (found_root, max_val, max_idx);
            if let Some(ref first) = first_result {
                if first.0 != result.0 || first.2 != result.2 {
                    panic!(
                        "Root finding results differ across batches: batch 0 = {:?}, batch {} = {:?}",
                        first, b, result
                    );
                }
            } else {
                first_result = Some(result);
            }
        }
        first_result.unwrap()
    }
    fn assign_at_indices(&mut self, indices: &Self::Index, value: Self::T) {
        let ctx = self.context.clone();
        let nindices = indices.len();
        if nindices == 0 {
            return;
        }
        let mut dest = self.operand_mut();
        let nindices_u32 = nindices as u32;
        let (nbatch_u32, dest_stride) = (dest.nbatch, dest.stride);
        let n = nindices_u32 * nbatch_u32;
        let cfg = OxideContext::config_1d(n);
        let m = &ctx.module;
        let p = m
            .prepare_vec_assign_at_indices(cfg)
            .expect("prepare vec_assign_at_indices");
        m.vec_assign_at_indices(
            &ctx.stream,
            &p,
            &mut dest.window,
            &indices.data,
            value,
            n,
            nindices_u32,
            dest_stride,
        )
        .expect("launch vec_assign_at_indices");
    }
    fn copy_from_indices(&mut self, other: &Self, indices: &Self::Index) {
        let ctx = self.context.clone();
        ctx.assert_broadcastable_into(other.context.nbatch(), "copy_from_indices");
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
        let p = m
            .prepare_vec_copy_from_indices(cfg)
            .expect("prepare vec_copy_from_indices");
        m.vec_copy_from_indices(
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
        .expect("launch vec_copy_from_indices");
    }
    fn gather(&mut self, other: &Self, indices: &Self::Index) {
        // as on the CPU backends: the destination is exactly the indices long
        assert_eq!(self.len(), indices.len());
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
    fn scatter(&self, indices: &Self::Index, other: &mut Self) {
        let ctx = other.context.clone();
        ctx.assert_broadcastable_into(self.context.nbatch(), "scatter");
        let nindices = indices.len();
        if nindices == 0 {
            return;
        }
        let src = self.operand();
        let mut dest = other.operand_mut();
        let nindices_u32 = nindices as u32;
        let (nbatch_u32, dest_stride) = (dest.nbatch, dest.stride);
        let n = nindices_u32 * nbatch_u32;
        let cfg = OxideContext::config_1d(n);
        let m = &ctx.module;
        let p = m.prepare_vec_scatter(cfg).expect("prepare vec_scatter");
        m.vec_scatter(
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
        .expect("launch vec_scatter");
    }
    /// The closure is host code, so every operand is staged through host
    /// memory.
    ///
    /// TODO: cuda-oxide can compile a host closure into a generic kernel, which would
    /// let this run on the device; not attempted yet.
    fn for_each_batch_mut<const M: usize, const N: usize>(
        mut mut_args: [&mut Self; M],
        args: [&Self; N],
        mut f: impl FnMut([&mut [Self::T]; M], [&[Self::T]; N], usize),
    ) {
        assert!(M > 0, "for_each_batch needs at least one mutable operand");
        let nbatch = mut_args[0].context.nbatch();
        {
            let ctx = &mut_args[0].context;
            for arg in mut_args.iter() {
                ctx.assert_broadcastable_into(arg.context.nbatch(), "for_each_batch");
            }
            for arg in args.iter() {
                ctx.assert_broadcastable_into(arg.context.nbatch(), "for_each_batch");
            }
        }
        // The mutable operands carry their own (nstates, nbatch) so the lane
        // slices need no index into `mut_args`.
        let mut host: [(Vec<Self::T>, usize, usize); M] = std::array::from_fn(|i| {
            (
                mut_args[i].clone_as_vec(),
                mut_args[i].len(),
                mut_args[i].context.nbatch(),
            )
        });
        let arg_host: [Vec<Self::T>; N] = args.map(|a| a.clone_as_vec());
        for b in 0..nbatch {
            let ins = std::array::from_fn(|i| {
                let n = args[i].len();
                let ab = broadcast_batch(b, args[i].context.nbatch(), nbatch);
                &arg_host[i][ab * n..(ab + 1) * n]
            });
            let outs = host.each_mut().map(|(h, n, arg_nbatch)| {
                let mb = broadcast_batch(b, *arg_nbatch, nbatch);
                &mut h[mb * *n..(mb + 1) * *n]
            });
            f(outs, ins, b);
        }
        for (v, (h, _, _)) in mut_args.iter_mut().zip(host.iter()) {
            let stream = v.context.stream.clone();
            v.data
                .copy_from_host(&stream, h)
                .expect("Failed to copy data from host to device");
        }
    }
}

// ============================================================
// VectorView / VectorViewMut
// ============================================================

impl VectorView<'_> for OxideVecRef<'_> {
    type Owned = OxideVec;
    fn get_index(&self, index: IndexType) -> Self::T {
        assert_eq!(
            self.context.nbatch(),
            1,
            "get_index not supported for batched views"
        );
        let mut out = [0.0f64];
        read_at(
            &self.context.stream,
            self.data,
            self.col_offset + index,
            &mut out,
        )
        .expect("Failed to copy data from device to host");
        out[0]
    }
    fn into_owned(self) -> Self::Owned {
        let nbatch = self.context.nbatch();
        let ret = OxideVec::zeros(self.nstates, self.context.clone());
        for b in 0..nbatch {
            copy_at(
                &self.context.stream,
                &ret.data,
                b * self.nstates,
                self.data,
                b * self.stride + self.col_offset,
                self.nstates,
            )
            .expect("Failed to copy device to device");
        }
        ret
    }
    fn squared_norm(&self, y: &Self::Owned, atol: &Self::Owned, rtol: Self::T) -> Self::T {
        let nstates = self.nstates;
        if nstates == 0 {
            return 0.0;
        }
        let ctx = self.context.clone();
        ctx.assert_broadcastable_into(y.context.nbatch(), "squared_norm");
        ctx.assert_broadcastable_into(atol.context.nbatch(), "squared_norm");
        // as on the CPU backends, the reduction runs over `self`'s batches and
        // the kernel broadcasts `y` and `atol` over them
        let nbatch = ctx.nbatch();
        let n = nstates as u32;
        let nb = nbatch as u32;
        let self_op = self.operand();
        let y_op = y.operand();
        let atol_op = atol.operand();
        let m = &ctx.module;
        let max_sum = reduce(&ctx, nstates, nbatch, |out| match out {
            ReduceOut::Small {
                out,
                cfg,
                cols_per_block,
            } => {
                let p = m
                    .prepare_vec_squared_norm_small(cfg)
                    .expect("prepare vec_squared_norm_small");
                m.vec_squared_norm_small(
                    &ctx.stream,
                    &p,
                    out,
                    &self_op.window,
                    &y_op.window,
                    &atol_op.window,
                    rtol,
                    n,
                    nb,
                    self_op.stride,
                    y_op.stride,
                    y_op.nbatch,
                    atol_op.stride,
                    atol_op.nbatch,
                    cols_per_block,
                )
                .expect("launch vec_squared_norm_small");
            }
            ReduceOut::Large {
                out,
                partials,
                cfg,
                blocks_per_lane,
            } => {
                let p = m
                    .prepare_vec_squared_norm(cfg)
                    .expect("prepare vec_squared_norm");
                m.vec_squared_norm(
                    &ctx.stream,
                    &p,
                    out,
                    partials,
                    &self_op.window,
                    &y_op.window,
                    &atol_op.window,
                    rtol,
                    n,
                    nb,
                    self_op.stride,
                    y_op.stride,
                    y_op.nbatch,
                    atol_op.stride,
                    atol_op.nbatch,
                    blocks_per_lane,
                )
                .expect("launch vec_squared_norm");
            }
        });
        max_sum / nstates as f64
    }
}

impl<'a> VectorViewMut<'a> for OxideVecMut<'a> {
    type Owned = OxideVec;
    type View = OxideVecRef<'a>;
    type Index = OxideIndex;
    fn copy_from(&mut self, other: &Self::Owned) {
        let ctx = self.context.clone();
        ctx.assert_broadcastable_into(other.context.nbatch(), "copy_from");
        let nstates = self.nstates;
        let src = other.operand();
        let dest = self.operand_mut();
        launch_assign(&ctx, AssignOp::Copy, dest, nstates, &src);
    }
    fn copy_from_view(&mut self, other: &Self::View) {
        let ctx = self.context.clone();
        ctx.assert_broadcastable_into(other.context.nbatch(), "copy_from_view");
        let nstates = self.nstates;
        let src = other.operand();
        let dest = self.operand_mut();
        launch_assign(&ctx, AssignOp::Copy, dest, nstates, &src);
    }
    fn set_index(&mut self, index: IndexType, value: Self::T) {
        assert_eq!(
            self.context.nbatch(),
            1,
            "set_index not supported for batched vectors, use fill_index"
        );
        self.fill_index(index, value);
    }
    fn fill_index(&mut self, index: IndexType, value: Self::T) {
        let nbatch = self.context.nbatch();
        assert!(index < self.nstates, "Index out of bounds");
        for b in 0..nbatch {
            let offset = b * self.stride + self.col_offset + index;
            write_at(&self.context.stream, self.data, offset, &[value])
                .expect("Failed to copy data from host to device");
        }
    }
    fn axpy(&mut self, alpha: Self::T, x: &Self::Owned, beta: Self::T) {
        let ctx = self.context.clone();
        ctx.assert_broadcastable_into(x.context.nbatch(), "axpy");
        let nstates = self.nstates;
        let src = x.operand();
        let dest = self.operand_mut();
        launch_axpy(&ctx, dest, nstates, alpha, &src, beta);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Launch-shape timing, for comparing the 2D and flat-1D launch shapes.
    ///
    /// Run with `just oxide-test -- --ignored --nocapture timing_axpy`.
    #[test]
    #[ignore = "timing only"]
    fn timing_axpy() {
        use std::time::Instant;
        const REPS: u32 = 2000;
        // fixed total element count, varying how it is split -- isolates the
        // launch shape from the amount of work
        for (nstates, nbatch) in [
            (5usize, 60_000usize),
            (30, 10_000),
            (300, 1_000),
            (3_000, 100),
            (300_000, 1),
        ] {
            let ctx = OxideContext::default().with_nbatch(nbatch);
            let mut y = OxideVec::from_element(nstates, 1.0, ctx.clone());
            let x = OxideVec::from_element(nstates, 2.0, ctx.clone());
            for _ in 0..50 {
                y.axpy(1.0, &x, 1.0);
            }
            ctx.stream.synchronize().unwrap();
            let start = Instant::now();
            for _ in 0..REPS {
                y.axpy(1.0, &x, 1.0);
            }
            ctx.stream.synchronize().unwrap();
            let per = start.elapsed() / REPS;
            println!("axpy nstates={nstates:>6} nbatch={nbatch:>5}: {per:?} per launch");
        }
    }

    /// `gridDim.y` caps at 65535, so the 2-D launch shape this backend used to
    /// have could not run more than that many batches at all -- `prepare_*`
    /// rejected it with `DeviceDimensionExceeded`. The flat shape puts the
    /// batch on `grid.x`, which caps at 2^31 - 1.
    #[test]
    fn nbatch_above_grid_y_limit() {
        const NSTATES: usize = 2;
        const NBATCH: usize = 70_000;
        let ctx = OxideContext::default().with_nbatch(NBATCH);
        let mut y = OxideVec::from_element(NSTATES, 1.0, ctx.clone());
        let x = OxideVec::from_element(NSTATES, 2.0, ctx.clone());
        y.axpy(3.0, &x, 10.0);
        let host = y.clone_as_vec();
        assert_eq!(host.len(), NSTATES * NBATCH);
        // 3 * 2 + 10 * 1
        assert!(host.iter().all(|&v| v == 16.0));
    }

    /// `norm(2)`, `norm(1)` and `squared_norm` against host arithmetic.
    ///
    /// The lanes get different magnitudes, so a reduction that dropped the
    /// cross-lane maximum, or took lane 0's, fails instead of passing by
    /// symmetry -- which is the part of these kernels the shared vector suites
    /// do not pin down.
    fn check_reductions(nstates: usize, nbatch: usize) {
        const RTOL: f64 = 0.1;
        const ATOL: f64 = 0.25;
        let ctx = OxideContext::default().with_nbatch(nbatch);
        let host: Vec<f64> = (0..nstates * nbatch)
            .map(|i| (1 + i / nstates) as f64 * (1.0 + (i % nstates % 7) as f64))
            .collect();
        let x = OxideVec::from_vec(host.clone(), ctx.clone());
        let atol = OxideVec::from_element(nstates, ATOL, ctx.clone());

        let want = |f: &dyn Fn(&[f64]) -> f64| {
            (0..nbatch)
                .map(|b| f(&host[b * nstates..(b + 1) * nstates]))
                .fold(0.0, f64::max)
        };
        let expected = [
            want(&|l| l.iter().map(|v| v * v).sum::<f64>().sqrt()),
            want(&|l| l.iter().map(|v| v.abs()).sum::<f64>()),
            want(&|l| {
                l.iter()
                    .map(|v| {
                        let r = v / (v.abs() * RTOL + ATOL);
                        r * r
                    })
                    .sum::<f64>()
            }) / nstates as f64,
        ];
        let got = [x.norm(2), x.norm(1), x.squared_norm(&x, &atol, RTOL)];
        for (name, (got, want)) in ["norm(2)", "norm(1)", "squared_norm"]
            .iter()
            .zip(got.iter().zip(expected.iter()))
        {
            // the device sums per warp then across warps, the host
            // sequentially
            assert!(
                ((got - want) / want).abs() < 1e-10,
                "{name} at nstates={nstates} nbatch={nbatch}: {got} != {want}"
            );
        }
    }

    /// [`SMALL_NSTATES`] is 85, so 85 takes the several-lanes-per-block
    /// kernels and 86 the one-block-per-lane ones.
    #[test]
    fn reductions_across_the_nstates_threshold() {
        for nstates in [1, 3, 64, 85, 86, 300] {
            for nbatch in [1, 2, 7] {
                check_reductions(nstates, nbatch);
            }
        }
    }

    /// A long lane with only one of them: `blocks_per_lane > 1`, so several
    /// blocks share the lane and `lane_sum_max` folds a run longer than one.
    /// This is the shape that would run on a single SM without it.
    #[test]
    fn reduction_with_several_blocks_per_lane() {
        check_reductions(300_000, 1);
    }

    /// Past the 65535 `gridDim.y` the reductions used to be capped by, on the
    /// large path (`nstates > SMALL_NSTATES`) with `nbatch` far above
    /// `target_blocks`, so the lane loop runs many passes.
    #[test]
    fn reduction_above_grid_y_limit() {
        check_reductions(129, 70_000);
    }

    /// Whether the large kernel's cost is its lane loop or its bytes.
    ///
    /// Every shape is 6.5M elements on the large path, so the bytes moved are
    /// identical; what changes is how many lane iterations the grid runs, and so
    /// how many times a thread issues one load, stalls on it, and waits out the
    /// block reduction's barriers before it may issue the next. Bandwidth-bound
    /// would be flat; loop-bound falls as lanes get longer and fewer.
    ///
    /// Run with
    /// `just oxide-test --release -- --ignored --nocapture timing_lane_loop`.
    #[test]
    #[ignore = "timing only"]
    fn timing_lane_loop() {
        use std::time::Instant;
        const REPS: u32 = 50;
        const ELEMS: usize = 6_500_000;
        for nstates in [130usize, 260, 650, 1_300, 6_500, 65_000] {
            let nbatch = ELEMS / nstates;
            let ctx = OxideContext::default().with_nbatch(nbatch);
            let x = OxideVec::from_element(nstates, 2.0, ctx.clone());
            for _ in 0..5 {
                let _ = x.norm(2);
            }
            let start = Instant::now();
            for _ in 0..REPS {
                let _ = x.norm(2);
            }
            let per = start.elapsed() / REPS;
            let gbs = (ELEMS * 8) as f64 / per.as_secs_f64() / 1e9;
            println!("nstates={nstates:>6} nbatch={nbatch:>6}: {per:?}  {gbs:.0} GB/s");
        }
    }

    /// One `norm(2)` on each side of the `SMALL_NSTATES` crossing, for `ncu`.
    ///
    /// One launch per shape, because a profiled launch costs ~100ms. Run with
    /// `ncu --section SpeedOfLight --section WarpStateStats -k regex:vec_norm`
    /// against the release test binary.
    ///
    /// Needs host-level `CAP_SYS_ADMIN`: NVIDIA restricts the counters to
    /// admins, and a namespaced capability does not satisfy that check, so this
    /// cannot be profiled from inside a rootless container. Lifting it means
    /// `NVreg_RestrictProfilingToAdminUsers=0` on the host and a reboot.
    #[test]
    #[ignore = "profiling only"]
    fn ncu_shapes() {
        const NBATCH: usize = 100_000;
        for nstates in [85usize, 86] {
            let ctx = OxideContext::default().with_nbatch(NBATCH);
            let x = OxideVec::from_element(nstates, 2.0, ctx.clone());
            println!("nstates={nstates} norm={}", x.norm(2));
        }
    }

    /// `norm(2)` across the `SMALL_NSTATES` crossing, which is what calibrated
    /// the constant.
    ///
    /// One curve per run: this shows whichever kernel the current threshold
    /// picks. To re-tune it, flip `SMALL_NSTATES` to 256 (the largest
    /// `nstates` the small kernels can take, since `cols_per_block` floors to
    /// 1 there), run this again, and compare the two curves -- the table in
    /// that constant's docs was built exactly that way.
    ///
    /// Run with
    /// `just oxide-test --release -- --ignored --nocapture timing_threshold`.
    #[test]
    #[ignore = "timing only"]
    fn timing_threshold() {
        use std::time::Instant;
        const REPS: u32 = 200;
        const NBATCH: usize = 10_000;
        for nstates in [64usize, 80, 85, 86, 128, 200] {
            let ctx = OxideContext::default().with_nbatch(NBATCH);
            let x = OxideVec::from_element(nstates, 2.0, ctx.clone());
            for _ in 0..20 {
                let _ = x.norm(2);
            }
            let start = Instant::now();
            for _ in 0..REPS {
                let _ = x.norm(2);
            }
            println!(
                "nbatch={NBATCH} nstates={nstates:>4}: {:?}",
                start.elapsed() / REPS
            );
        }
    }

    /// Reduction cost against launch shape, the same sweep as `timing_axpy`.
    ///
    /// Run with `just oxide-test -- --release --ignored --nocapture timing_norm`.
    #[test]
    #[ignore = "timing only"]
    fn timing_norm() {
        use std::time::Instant;
        const REPS: u32 = 500;
        for (nstates, nbatch) in [
            (5usize, 60_000usize),
            (30, 10_000),
            (100, 10_000),
            (300, 1_000),
            (3_000, 100),
            (300_000, 1),
        ] {
            let ctx = OxideContext::default().with_nbatch(nbatch);
            let x = OxideVec::from_element(nstates, 2.0, ctx.clone());
            let atol = OxideVec::from_element(nstates, 1e-6, ctx.clone());
            for _ in 0..20 {
                let _ = x.norm(2);
            }
            let start = Instant::now();
            for _ in 0..REPS {
                let _ = x.norm(2);
            }
            let norm = start.elapsed() / REPS;
            let start = Instant::now();
            for _ in 0..REPS {
                let _ = x.squared_norm(&x, &atol, 1e-6);
            }
            let sq = start.elapsed() / REPS;
            println!(
                "nstates={nstates:>6} nbatch={nbatch:>5}: norm(2) {norm:?}, squared_norm {sq:?}"
            );
        }
    }

    /// Where the per-op cost goes: `prepare_*` versus the driver launch.
    ///
    /// Run with `just oxide-test -- --release --ignored --nocapture timing_breakdown`.
    #[test]
    #[ignore = "timing only"]
    fn timing_breakdown() {
        use std::time::Instant;
        const REPS: u32 = 2000;
        let (nstates, nbatch) = (3usize, 1000usize);
        let ctx = OxideContext::default().with_nbatch(nbatch);
        let mut y = OxideVec::from_element(nstates, 1.0, ctx.clone());
        let x = OxideVec::from_element(nstates, 2.0, ctx.clone());
        let n = (nstates * nbatch) as u32;
        let cfg = OxideContext::config_1d(n);
        let m = &ctx.module;

        for _ in 0..50 {
            let _ = m.prepare_vec_axpy(cfg).unwrap();
        }
        let start = Instant::now();
        for _ in 0..REPS {
            std::hint::black_box(m.prepare_vec_axpy(cfg).unwrap());
        }
        println!("prepare only:        {:?}", start.elapsed() / REPS);

        let prepared = m.prepare_vec_axpy(cfg).unwrap();
        let mut dest = y.operand_mut();
        let src = x.operand();
        let (stride, nb) = (dest.stride, dest.nbatch);
        macro_rules! launch {
            () => {
                m.vec_axpy(
                    &ctx.stream,
                    &prepared,
                    &mut dest.window,
                    &src.window,
                    1.0,
                    1.0,
                    n,
                    nstates as u32,
                    stride,
                    src.stride,
                    src.nbatch,
                    nb,
                )
                .unwrap()
            };
        }
        for _ in 0..50 {
            launch!();
        }
        ctx.stream.synchronize().unwrap();
        let start = Instant::now();
        for _ in 0..REPS {
            launch!();
        }
        ctx.stream.synchronize().unwrap();
        println!("launch only:         {:?}", start.elapsed() / REPS);

        for _ in 0..50 {
            y.axpy(1.0, &x, 1.0);
        }
        ctx.stream.synchronize().unwrap();
        let start = Instant::now();
        for _ in 0..REPS {
            y.axpy(1.0, &x, 1.0);
        }
        ctx.stream.synchronize().unwrap();
        println!("Vector::axpy (both): {:?}", start.elapsed() / REPS);
    }

    super::super::generate_vector_tests_nonbatched!(cuda_oxide, OxideVec);

    super::super::generate_vector_tests_batched!(
        cuda_oxide,
        OxideVec,
        OxideContext::default().with_nbatch(2),
        OxideContext::default().with_nbatch(3)
    );
}
