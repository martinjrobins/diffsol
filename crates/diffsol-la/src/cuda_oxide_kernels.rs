//! GPU kernels for the `cuda-oxide` backend.
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU64};
use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_device::{DisjointSlice, Runtime2DIndex, SharedArray};

use crate::matrix::MAX_SMALL_COLS;

const MAX_SMALL_COLS_SQ: usize = MAX_SMALL_COLS * MAX_SMALL_COLS;
pub(crate) const BLOCK_SIZE: u32 = 256;

/// Warps in a block. Every kernel's launch contract pins the block to
/// `(BLOCK_SIZE, 1, 1)`, so a block is always this many *whole* warps -- which
/// is what lets the block reductions shuffle with the full-warp mask.
const WARPS_PER_BLOCK: usize = BLOCK_SIZE as usize / 32;

/// Above this `nstates`, the reductions switch from several-lanes-per-block to
/// one-block-per-lane.
///
/// the crossing moves with the block shape and with how many SMs the
/// device has. `timing_threshold` in `crate::vector::cuda_oxide`'s tests
/// re-derives it; this is the only number to change.
///
/// 85 is where `cols_per_block = BLOCK_SIZE / nstates` falls from 3 to 2, and
/// measurement lands on the same step: at `nbatch = 10_000` on an A40 the small
/// kernels take 61us at 85 against the large path's 80us, and 85us at 86
/// against its 80us.
pub(crate) const SMALL_NSTATES: u32 = 85;

/// Batch lane and element for flat work item `i`.
/// Note: The host guarantees `nstates > 0`.
#[inline(always)]
fn split(i: usize, nstates: u32) -> (usize, usize) {
    let nstates = nstates as usize;
    (i / nstates, i % nstates)
}

/// Source batch feeding destination batch `b`, for a source holding
/// `src_nbatch` batches, resolved to a flat index.
///
/// Device mirror of [`crate::context::broadcast_batch`]
#[inline(always)]
fn broadcast_src(b: usize, src_stride: u32, src_nbatch: u32, nbatch: u32, elem: usize) -> usize {
    let src_b = b * src_nbatch as usize / nbatch as usize;
    src_b * src_stride as usize + elem
}

/// Stride between the elements one thread visits in a grid-stride loop.
#[inline(always)]
fn grid_stride() -> usize {
    (thread::blockDim_x() * thread::gridDim_x()) as usize
}

/// Slot in a per-block output array for this block: one per `(batch, block)`.
#[inline(always)]
fn block_slot(b: usize) -> usize {
    b * thread::gridDim_x() as usize + thread::blockIdx_x() as usize
}

/// Lane geometry of a small reduction's block: the first lane it owns,
/// `nstates` as a `usize`, and how many lanes fit in a block.
#[inline(always)]
fn lane_block(nstates: u32, cols_per_block: u32) -> (usize, usize, usize) {
    let cols = cols_per_block as usize;
    (thread::blockIdx_x() as usize * cols, nstates as usize, cols)
}

/// The `(lane, element)` thread `tid` loads in a small reduction's first phase,
/// or `None` when it has none: `BLOCK_SIZE % nstates` threads are spare, and
/// the last block can reach past `nbatch`.
#[inline(always)]
fn lane_element(
    first: usize,
    nstates: usize,
    cols: usize,
    nbatch: u32,
    tid: usize,
) -> Option<(usize, usize)> {
    if tid >= cols * nstates {
        return None;
    }
    let b = first + tid / nstates;
    if b >= nbatch as usize {
        return None;
    }
    Some((b, tid % nstates))
}

/// How a large reduction's block walks the lanes:
/// `(first lane, this block's slice of a lane, lanes per pass, element stride)`.
///
/// The grid is `blocks_per_lane * (lanes per pass)` blocks, so a block covers
/// slice `blockIdx.x % blocks_per_lane` of every lane congruent to
/// `blockIdx.x / blocks_per_lane`.
#[inline(always)]
fn lane_loop(blocks_per_lane: u32) -> (usize, usize, usize, usize) {
    let bpl = blocks_per_lane as usize;
    let block = thread::blockIdx_x() as usize;
    (
        block / bpl,
        block % bpl,
        thread::gridDim_x() as usize / bpl,
        bpl * thread::blockDim_x() as usize,
    )
}

/// First element of a lane covered by this thread of slice `blk`.
#[inline(always)]
fn lane_start(blk: usize) -> usize {
    blk * thread::blockDim_x() as usize + thread::threadIdx_x() as usize
}

/// Device addresses and lane geometry of the read-only operands of one
/// [`kernels::vec_for_each_batch`] launch.
#[derive(Clone, Copy)]
pub struct LaneArgs<const K: usize> {
    pub ptr: [*const f64; K],
    pub nstates: [u32; K],
    pub nbatch: [u32; K],
}

/// Mutable counterpart of [`LaneArgs`], for the operands the closure writes.
#[derive(Clone, Copy)]
pub struct LaneArgsMut<const K: usize> {
    pub ptr: [*mut f64; K],
    pub nstates: [u32; K],
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Ordering for the reductions' one atomic. The backend needs a constant
    /// here, and a `const` item reads as one in MIR where an inline variant
    /// path does not.
    const RELAXED: AtomicOrdering = AtomicOrdering::Relaxed;

    // ========================================================================
    // Elementwise, contiguous destination (Tier 1)
    // ========================================================================
    //
    // Elementwise kernels are launched **flat**: one thread per work item over
    // `nstates * nbatch` items, `grid.y = 1`. Thread `i` splits into a batch lane
    // and an element with [`split`].

    /// `lhs[b, elem] = value`
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), requires = (lhs.len() == n))]
    pub fn vec_fill(mut lhs: DisjointSlice<f64>, value: f64, n: u32) {
        let idx = thread::index_1d();
        if idx.get() < n as usize {
            if let Some(elem) = lhs.get_mut(idx) {
                *elem = value;
            }
        }
    }

    /// `lhs[b, elem] = rhs[..] - lhs[b, elem]`
    ///
    /// The reversed form of [`vec_sub_assign`], for `&lhs - rhs` where `rhs` is
    /// the owned operand and therefore the destination.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), requires = (lhs.len() == n))]
    #[allow(clippy::too_many_arguments)]
    pub fn vec_sub_assign_rev(
        mut lhs: DisjointSlice<f64>,
        rhs: &[f64],
        n: u32,
        nstates: u32,
        rhs_stride: u32,
        rhs_nbatch: u32,
        nbatch: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < n as usize {
            let (b, elem) = split(i, nstates);
            let value = rhs[broadcast_src(b, rhs_stride, rhs_nbatch, nbatch, elem)];
            if let Some(elem) = lhs.get_mut(idx) {
                *elem = value - *elem;
            }
        }
    }

    /// `lhs[b, elem] *= rhs[..]`
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), requires = (lhs.len() == n))]
    #[allow(clippy::too_many_arguments)]
    pub fn vec_mul_assign(
        mut lhs: DisjointSlice<f64>,
        rhs: &[f64],
        n: u32,
        nstates: u32,
        rhs_stride: u32,
        rhs_nbatch: u32,
        nbatch: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < n as usize {
            let (b, elem) = split(i, nstates);
            let value = rhs[broadcast_src(b, rhs_stride, rhs_nbatch, nbatch, elem)];
            if let Some(elem) = lhs.get_mut(idx) {
                *elem *= value;
            }
        }
    }

    /// `lhs[b, elem] /= rhs[..]`
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), requires = (lhs.len() == n))]
    #[allow(clippy::too_many_arguments)]
    pub fn vec_div_assign(
        mut lhs: DisjointSlice<f64>,
        rhs: &[f64],
        n: u32,
        nstates: u32,
        rhs_stride: u32,
        rhs_nbatch: u32,
        nbatch: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < n as usize {
            let (b, elem) = split(i, nstates);
            let value = rhs[broadcast_src(b, rhs_stride, rhs_nbatch, nbatch, elem)];
            if let Some(elem) = lhs.get_mut(idx) {
                *elem /= value;
            }
        }
    }

    /// `ret[b, elem] = lhs[..] + rhs[..]`, for the allocating `&a + &b`.
    ///
    /// `ret` is freshly allocated at the launch's batch count, so it needs no
    /// broadcast of its own.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), requires = (ret.len() == n))]
    #[allow(clippy::too_many_arguments)]
    pub fn vec_add(
        mut ret: DisjointSlice<f64>,
        lhs: &[f64],
        rhs: &[f64],
        n: u32,
        nstates: u32,
        lhs_stride: u32,
        lhs_nbatch: u32,
        rhs_stride: u32,
        rhs_nbatch: u32,
        nbatch: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < n as usize {
            let (b, elem) = split(i, nstates);
            let a = lhs[broadcast_src(b, lhs_stride, lhs_nbatch, nbatch, elem)];
            let c = rhs[broadcast_src(b, rhs_stride, rhs_nbatch, nbatch, elem)];
            if let Some(elem) = ret.get_mut(idx) {
                *elem = a + c;
            }
        }
    }

    /// `ret[b, elem] = lhs[..] - rhs[..]`, for the allocating `&a - &b`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), requires = (ret.len() == n))]
    #[allow(clippy::too_many_arguments)]
    pub fn vec_sub(
        mut ret: DisjointSlice<f64>,
        lhs: &[f64],
        rhs: &[f64],
        n: u32,
        nstates: u32,
        lhs_stride: u32,
        lhs_nbatch: u32,
        rhs_stride: u32,
        rhs_nbatch: u32,
        nbatch: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < n as usize {
            let (b, elem) = split(i, nstates);
            let a = lhs[broadcast_src(b, lhs_stride, lhs_nbatch, nbatch, elem)];
            let c = rhs[broadcast_src(b, rhs_stride, rhs_nbatch, nbatch, elem)];
            if let Some(elem) = ret.get_mut(idx) {
                *elem = a - c;
            }
        }
    }

    /// `ret[b, elem] = scalar * src[..]`
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), requires = (ret.len() == n))]
    #[allow(clippy::too_many_arguments)]
    pub fn vec_mul_scalar(
        mut ret: DisjointSlice<f64>,
        src: &[f64],
        scalar: f64,
        n: u32,
        nstates: u32,
        src_stride: u32,
        src_nbatch: u32,
        nbatch: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < n as usize {
            let (b, elem) = split(i, nstates);
            let value = src[broadcast_src(b, src_stride, src_nbatch, nbatch, elem)];
            if let Some(elem) = ret.get_mut(idx) {
                *elem = scalar * value;
            }
        }
    }

    /// `y[b, elem] = alpha[b] * x[..] + beta * y[b, elem]`, one `alpha` per
    /// batch lane.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), requires = (y.len() == n))]
    #[allow(clippy::too_many_arguments)]
    pub fn vec_batched_axpy(
        mut y: DisjointSlice<f64>,
        x: &[f64],
        alpha: &[f64],
        beta: f64,
        n: u32,
        nstates: u32,
        x_stride: u32,
        x_nbatch: u32,
        nbatch: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < n as usize {
            let (b, elem) = split(i, nstates);
            let xv = x[broadcast_src(b, x_stride, x_nbatch, nbatch, elem)];
            let a = alpha[b];
            if let Some(elem) = y.get_mut(idx) {
                *elem = a * xv + beta * *elem;
            }
        }
    }

    // ========================================================================
    // Elementwise, destination may be a matrix column (Tier 2)
    // ========================================================================
    //
    // These five are the ops reachable with a strided destination, via
    // `Matrix::set_column` or a `DenseMatrix::column_mut()` view. The element
    // lives at `b * dest_stride + elem`, which the `index_1d` witness cannot
    // name, so the write goes through the raw pointer. The disjointness
    // argument is the same in all five and is stated once here: distinct
    // threads hold distinct `i`, hence distinct `(b, elem)` pairs, and with
    // `elem < nstates <= dest_stride` distinct pairs give distinct
    // `b * dest_stride + elem`. Bounds are checked per write, and the
    // `requires` clause rejects an undersized destination on the host.

    /// `lhs[b, elem] = rhs[..]`
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1),
                      requires = (lhs.len() >= (nbatch - 1) * lhs_stride + nstates))]
    #[allow(clippy::too_many_arguments)]
    pub fn vec_copy(
        mut lhs: DisjointSlice<f64>,
        rhs: &[f64],
        n: u32,
        nstates: u32,
        lhs_stride: u32,
        rhs_stride: u32,
        rhs_nbatch: u32,
        nbatch: u32,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        let (b, elem) = split(i, nstates);
        let value = rhs[broadcast_src(b, rhs_stride, rhs_nbatch, nbatch, elem)];
        let li = b * lhs_stride as usize + elem;
        if li < lhs.len() {
            // SAFETY: see the section comment above.
            unsafe {
                *lhs.as_mut_ptr().add(li) = value;
            }
        }
    }

    /// `lhs[b, elem] += rhs[..]`
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1),
                      requires = (lhs.len() >= (nbatch - 1) * lhs_stride + nstates))]
    #[allow(clippy::too_many_arguments)]
    pub fn vec_add_assign(
        mut lhs: DisjointSlice<f64>,
        rhs: &[f64],
        n: u32,
        nstates: u32,
        lhs_stride: u32,
        rhs_stride: u32,
        rhs_nbatch: u32,
        nbatch: u32,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        let (b, elem) = split(i, nstates);
        let value = rhs[broadcast_src(b, rhs_stride, rhs_nbatch, nbatch, elem)];
        let li = b * lhs_stride as usize + elem;
        if li < lhs.len() {
            // SAFETY: see the section comment above.
            unsafe {
                let p = lhs.as_mut_ptr().add(li);
                *p += value;
            }
        }
    }

    /// `lhs[b, elem] -= rhs[..]`
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1),
                      requires = (lhs.len() >= (nbatch - 1) * lhs_stride + nstates))]
    #[allow(clippy::too_many_arguments)]
    pub fn vec_sub_assign(
        mut lhs: DisjointSlice<f64>,
        rhs: &[f64],
        n: u32,
        nstates: u32,
        lhs_stride: u32,
        rhs_stride: u32,
        rhs_nbatch: u32,
        nbatch: u32,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        let (b, elem) = split(i, nstates);
        let value = rhs[broadcast_src(b, rhs_stride, rhs_nbatch, nbatch, elem)];
        let li = b * lhs_stride as usize + elem;
        if li < lhs.len() {
            // SAFETY: see the section comment above.
            unsafe {
                let p = lhs.as_mut_ptr().add(li);
                *p -= value;
            }
        }
    }

    /// `lhs[b, elem] *= scalar`
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1),
                      requires = (lhs.len() >= (nbatch - 1) * lhs_stride + nstates))]
    // `nbatch` is read by the `requires` clause above, which the host evaluates;
    // the body has no source operand to broadcast, so it never needs it.
    #[allow(unused_variables)]
    pub fn vec_mul_assign_scalar(
        mut lhs: DisjointSlice<f64>,
        scalar: f64,
        n: u32,
        nstates: u32,
        lhs_stride: u32,
        nbatch: u32,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        let (b, elem) = split(i, nstates);
        let li = b * lhs_stride as usize + elem;
        if li < lhs.len() {
            // SAFETY: see the section comment above.
            unsafe {
                let p = lhs.as_mut_ptr().add(li);
                *p *= scalar;
            }
        }
    }

    /// `y[b, elem] = alpha * x[..] + beta * y[b, elem]`
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1),
                      requires = (y.len() >= (nbatch - 1) * y_stride + nstates))]
    #[allow(clippy::too_many_arguments)]
    pub fn vec_axpy(
        mut y: DisjointSlice<f64>,
        x: &[f64],
        alpha: f64,
        beta: f64,
        n: u32,
        nstates: u32,
        y_stride: u32,
        x_stride: u32,
        x_nbatch: u32,
        nbatch: u32,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        let (b, elem) = split(i, nstates);
        let xv = x[broadcast_src(b, x_stride, x_nbatch, nbatch, elem)];
        let yi = b * y_stride as usize + elem;
        if yi < y.len() {
            // SAFETY: see the section comment above.
            unsafe {
                let p = y.as_mut_ptr().add(yi);
                *p = alpha * xv + beta * *p;
            }
        }
    }

    // ========================================================================
    // Index-driven copies (Tier 2)
    // ========================================================================
    //
    // The destination position comes from an index array rather than from the
    // thread's own coordinate (gather is odd-one out, see docstring).

    /// `dest[b, j] = src[.., indices[j]]`
    ///
    /// `Matrix::gather` keeps it Tier 2, the destination is a
    /// whole matrix, stride `nrows * ncols`, and the trait does not require the
    /// indices to cover it so writes as not contiguous.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    #[allow(clippy::too_many_arguments)]
    pub fn vec_gather(
        mut dest: DisjointSlice<f64>,
        src: &[f64],
        indices: &[i32],
        n: u32,
        nindices: u32,
        dest_stride: u32,
        src_stride: u32,
        src_nbatch: u32,
        nbatch: u32,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        let (b, j) = split(i, nindices);
        let value = src[broadcast_src(b, src_stride, src_nbatch, nbatch, indices[j] as usize)];
        let di = b * dest_stride as usize + j;
        if di < dest.len() {
            // SAFETY: bounds checked above; distinct threads hold distinct
            // `(b, j)` and `j < nindices <= dest_stride`, so `di` is distinct.
            unsafe {
                *dest.as_mut_ptr().add(di) = value;
            }
        }
    }

    /// `dest[b, indices[j]] = src[.., j]`
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    #[allow(clippy::too_many_arguments)]
    pub fn vec_scatter(
        mut dest: DisjointSlice<f64>,
        src: &[f64],
        indices: &[i32],
        n: u32,
        nindices: u32,
        dest_stride: u32,
        src_stride: u32,
        src_nbatch: u32,
        nbatch: u32,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        let (b, j) = split(i, nindices);
        let value = src[broadcast_src(b, src_stride, src_nbatch, nbatch, j)];
        let di = b * dest_stride as usize + indices[j] as usize;
        if di < dest.len() {
            // SAFETY: see the section comment above.
            unsafe {
                *dest.as_mut_ptr().add(di) = value;
            }
        }
    }

    /// `dest[b, indices[j]] = src[.., indices[j]]`
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    #[allow(clippy::too_many_arguments)]
    pub fn vec_copy_from_indices(
        mut dest: DisjointSlice<f64>,
        src: &[f64],
        indices: &[i32],
        n: u32,
        nindices: u32,
        dest_stride: u32,
        src_stride: u32,
        src_nbatch: u32,
        nbatch: u32,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        let (b, j) = split(i, nindices);
        let index = indices[j] as usize;
        let value = src[broadcast_src(b, src_stride, src_nbatch, nbatch, index)];
        let di = b * dest_stride as usize + index;
        if di < dest.len() {
            // SAFETY: see the section comment above.
            unsafe {
                *dest.as_mut_ptr().add(di) = value;
            }
        }
    }

    /// `dest[b, indices[j]] = value`
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn vec_assign_at_indices(
        mut dest: DisjointSlice<f64>,
        indices: &[i32],
        value: f64,
        n: u32,
        nindices: u32,
        dest_stride: u32,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        let (b, j) = split(i, nindices);
        let di = b * dest_stride as usize + indices[j] as usize;
        if di < dest.len() {
            // SAFETY: see the section comment above. Repeated indices would
            // have two threads write the same slot, but with the same `value`.
            unsafe {
                *dest.as_mut_ptr().add(di) = value;
            }
        }
    }

    // ========================================================================
    // Reductions (flat launch)
    // ========================================================================
    //
    // Each of `norm`, `norm_lk` and `squared_norm` comes in two shapes, because
    // a batch lane either fits inside a block or it does not. The host picks by
    // `nstates` against `SMALL_NSTATES`.
    //
    // *Small* `nstates`: threads in block assigned like column major indexing
    // where nstates are the rows and nmbatches are the cols
    // A block owns `cols_per_block = BLOCK_SIZE / nstates`
    // whole lanes.
    //
    // *Large* `nstates` (single block): one block per lane, block-striding over the lane's
    // states into a register and reducing that with [`block_sum`]'s warp
    // shuffles, then the block jumps `gridDim.x / blocks_per_lane` lanes along
    // and repeats. Grid is sized by the device rather than by `nbatch`.
    //
    // *Large* `nstates (multi block): if too few lanes to fill the device with
    // single block, then `blocks_per_lane > 1` and several blocks share a lane,
    // each block sum go to `partials` for a second `lane_sum_max` pass
    // to add up.

    /// Sum of `x[b, i]^2` per block, for the 2-norm.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1),
                      requires = (out.len() == 1,
                                  partials.len() >= blocks_per_lane * nbatch,
                                  x.len() >= (nbatch - 1) * x_stride + nstates))]
    pub fn vec_norm(
        out: &[DeviceAtomicU64],
        mut partials: DisjointSlice<f64>,
        x: &[f64],
        nstates: u32,
        nbatch: u32,
        x_stride: u32,
        blocks_per_lane: u32,
    ) {
        let (mut b, blk, lane_step, step) = lane_loop(blocks_per_lane);
        while b < nbatch as usize {
            let mut local = 0.0f64;
            let mut i = lane_start(blk);
            while i < nstates as usize {
                let v = x[b * x_stride as usize + i];
                local += v * v;
                i += step;
            }
            publish_block_sum(
                out,
                &mut partials,
                blocks_per_lane,
                b * blocks_per_lane as usize + blk,
                local,
            );
            b += lane_step;
        }
    }

    /// `max_b sum_i x[b, i]^2`, for lanes short enough to fit in a block.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1),
                      requires = (out.len() == 1,
                                  x.len() >= (nbatch - 1) * x_stride + nstates))]
    pub fn vec_norm_small(
        out: &[DeviceAtomicU64],
        x: &[f64],
        nstates: u32,
        nbatch: u32,
        x_stride: u32,
        cols_per_block: u32,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let (first, nstates, cols) = lane_block(nstates, cols_per_block);
        let term = match lane_element(first, nstates, cols, nbatch, tid) {
            Some((b, elem)) => {
                let v = x[b * x_stride as usize + elem];
                v * v
            }
            None => 0.0,
        };
        block_max_into(out, small_lane_sum(term, first, nstates, cols, nbatch));
    }

    /// Sum of `|x[b, i]|^k` per block, for the k-norm.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1),
                      requires = (out.len() == 1,
                                  partials.len() >= blocks_per_lane * nbatch,
                                  x.len() >= (nbatch - 1) * x_stride + nstates))]
    #[allow(clippy::too_many_arguments)]
    pub fn vec_norm_lk(
        out: &[DeviceAtomicU64],
        mut partials: DisjointSlice<f64>,
        x: &[f64],
        nstates: u32,
        nbatch: u32,
        x_stride: u32,
        blocks_per_lane: u32,
        k: i32,
    ) {
        let k_f64 = k as f64;
        let (mut b, blk, lane_step, step) = lane_loop(blocks_per_lane);
        while b < nbatch as usize {
            let mut local = 0.0f64;
            let mut i = lane_start(blk);
            while i < nstates as usize {
                local += x[b * x_stride as usize + i].abs().powf(k_f64);
                i += step;
            }
            publish_block_sum(
                out,
                &mut partials,
                blocks_per_lane,
                b * blocks_per_lane as usize + blk,
                local,
            );
            b += lane_step;
        }
    }

    /// `max_b sum_i |x[b, i]|^k`, for lanes short enough to fit in a block.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1),
                      requires = (out.len() == 1,
                                  x.len() >= (nbatch - 1) * x_stride + nstates))]
    #[allow(clippy::too_many_arguments)]
    pub fn vec_norm_lk_small(
        out: &[DeviceAtomicU64],
        x: &[f64],
        nstates: u32,
        nbatch: u32,
        x_stride: u32,
        cols_per_block: u32,
        k: i32,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let (first, nstates, cols) = lane_block(nstates, cols_per_block);
        let term = match lane_element(first, nstates, cols, nbatch, tid) {
            Some((b, elem)) => x[b * x_stride as usize + elem].abs().powf(k as f64),
            None => 0.0,
        };
        block_max_into(out, small_lane_sum(term, first, nstates, cols, nbatch));
    }

    /// Sum of `(y / (|y0| * rtol + atol))^2` per block, the BDF/RK error norm.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1),
                      requires = (out.len() == 1,
                                  partials.len() >= blocks_per_lane * nbatch,
                                  y.len() >= (nbatch - 1) * y_stride + nstates))]
    #[allow(clippy::too_many_arguments)]
    pub fn vec_squared_norm(
        out: &[DeviceAtomicU64],
        mut partials: DisjointSlice<f64>,
        y: &[f64],
        y0: &[f64],
        atol: &[f64],
        rtol: f64,
        nstates: u32,
        nbatch: u32,
        y_stride: u32,
        y0_stride: u32,
        y0_nbatch: u32,
        atol_stride: u32,
        atol_nbatch: u32,
        blocks_per_lane: u32,
    ) {
        let (mut b, blk, lane_step, step) = lane_loop(blocks_per_lane);
        while b < nbatch as usize {
            let mut local = 0.0f64;
            let mut i = lane_start(blk);
            while i < nstates as usize {
                let denom = y0[broadcast_src(b, y0_stride, y0_nbatch, nbatch, i)].abs() * rtol
                    + atol[broadcast_src(b, atol_stride, atol_nbatch, nbatch, i)];
                let ratio = y[b * y_stride as usize + i] / denom;
                local += ratio * ratio;
                i += step;
            }
            publish_block_sum(
                out,
                &mut partials,
                blocks_per_lane,
                b * blocks_per_lane as usize + blk,
                local,
            );
            b += lane_step;
        }
    }

    /// The error norm's sum, for lanes short enough to fit in a block.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1),
                      requires = (out.len() == 1,
                                  y.len() >= (nbatch - 1) * y_stride + nstates))]
    #[allow(clippy::too_many_arguments)]
    pub fn vec_squared_norm_small(
        out: &[DeviceAtomicU64],
        y: &[f64],
        y0: &[f64],
        atol: &[f64],
        rtol: f64,
        nstates: u32,
        nbatch: u32,
        y_stride: u32,
        y0_stride: u32,
        y0_nbatch: u32,
        atol_stride: u32,
        atol_nbatch: u32,
        cols_per_block: u32,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let (first, nstates_u, cols) = lane_block(nstates, cols_per_block);
        let term = match lane_element(first, nstates_u, cols, nbatch, tid) {
            Some((b, elem)) => {
                let denom = y0[broadcast_src(b, y0_stride, y0_nbatch, nbatch, elem)].abs() * rtol
                    + atol[broadcast_src(b, atol_stride, atol_nbatch, nbatch, elem)];
                let ratio = y[b * y_stride as usize + elem] / denom;
                ratio * ratio
            }
            None => 0.0,
        };
        block_max_into(out, small_lane_sum(term, first, nstates_u, cols, nbatch));
    }

    /// Folds the large kernels' per-lane partial sums into the one maximum.
    ///
    /// `partials` is `nbatch` runs of `blocks_per_lane` values. A thread sums
    /// its lane's run in index order -- `blocks_per_lane` is bounded by the
    /// host's occupancy target, so the chain is short -- and grid-strides over
    /// lanes, so one launch covers any `nbatch`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1),
                      requires = (out.len() == 1,
                                  partials.len() >= blocks_per_lane * nbatch))]
    pub fn lane_sum_max(
        out: &[DeviceAtomicU64],
        partials: &[f64],
        nbatch: u32,
        blocks_per_lane: u32,
    ) {
        let bpl = blocks_per_lane as usize;
        let step = grid_stride();
        let mut b = thread::index_1d().get();
        let mut local = 0.0f64;
        while b < nbatch as usize {
            let mut sum = 0.0f64;
            let base = b * bpl;
            for j in 0..bpl {
                sum += partials[base + j];
            }
            if sum > local {
                local = sum;
            }
            b += step;
        }
        block_max_into(out, local);
    }

    /// Root search between two g-vectors, per block.
    ///
    /// For every `i` where `g0[i] * g1[i] < 0`, tracks
    /// `max |g1[i] / (g1[i] - g0[i])|` and the `i` attaining it; `flags` reports
    /// whether any `g1[i]` is exactly zero. The host reduces the per-block
    /// results.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 2, block = (256, 1, 1))]
    #[allow(clippy::too_many_arguments)]
    pub fn vec_root_finding(
        mut max_vals: DisjointSlice<f64, Runtime2DIndex>,
        mut max_idxs: DisjointSlice<i32, Runtime2DIndex>,
        mut flags: DisjointSlice<i32, Runtime2DIndex>,
        g0: &[f64],
        g1: &[f64],
        nstates: u32,
        nbatch: u32,
        g0_stride: u32,
        g1_stride: u32,
        g1_nbatch: u32,
    ) {
        static mut SVALS: SharedArray<f64, { BLOCK_SIZE as usize }> = SharedArray::UNINIT;
        static mut SIDXS: SharedArray<i32, { BLOCK_SIZE as usize }> = SharedArray::UNINIT;
        static mut SFLAGS: SharedArray<i32, { BLOCK_SIZE as usize }> = SharedArray::UNINIT;

        let b = thread::index_2d_row();
        let mut local_max = 0.0f64;
        let mut local_idx = -1i32;
        let mut local_flag = 0i32;
        let mut i = thread::index_2d_col();
        let step = grid_stride();
        while i < nstates as usize {
            let v0 = g0[b * g0_stride as usize + i];
            let v1 = g1[broadcast_src(b, g1_stride, g1_nbatch, nbatch, i)];
            if v1 == 0.0 {
                local_flag = 1;
            }
            if v0 * v1 < 0.0 {
                let val = (v1 / (v1 - v0)).abs();
                if val > local_max {
                    local_max = val;
                    local_idx = i as i32;
                }
            }
            i += step;
        }

        let tid = thread::threadIdx_x() as usize;
        // SAFETY: each thread writes only its own slot, and the barrier below
        // separates these writes from any other thread's reads.
        unsafe {
            SVALS[tid] = local_max;
            SIDXS[tid] = local_idx;
            SFLAGS[tid] = local_flag;
        }
        thread::sync_threads();

        // argmax tree reduction; `BLOCK_SIZE` is a power of two so the halving
        // covers the whole block
        let mut s = BLOCK_SIZE as usize / 2;
        while s > 0 {
            if tid < s {
                // SAFETY: only threads below `s` touch shared memory in this
                // round, each at its own `tid` and at `tid + s` which no other
                // active thread owns. The barrier below closes the round before
                // the next one reads.
                unsafe {
                    if SVALS[tid] < SVALS[tid + s] {
                        SVALS[tid] = SVALS[tid + s];
                        SIDXS[tid] = SIDXS[tid + s];
                    }
                    if SFLAGS[tid + s] != 0 {
                        SFLAGS[tid] = 1;
                    }
                }
            }
            thread::sync_threads();
            s /= 2;
        }

        if tid == 0 {
            let slot = block_slot(b);
            // SAFETY: one slot per (batch, block), written by thread 0 only, so
            // no two threads in the grid write the same element. Bounds hold
            // because the host sizes all three arrays as `nbatch * gridDim.x`.
            unsafe {
                *max_vals.as_mut_ptr().add(slot) = SVALS[0];
                *max_idxs.as_mut_ptr().add(slot) = SIDXS[0];
                *flags.as_mut_ptr().add(slot) = SFLAGS[0];
            }
        }
    }

    // ========================================================================
    // Matrix
    // ========================================================================
    //
    // A matrix is column-major with whole batches contiguous, so its per-batch
    // stride is `nrows * ncols` and column `j` of batch `b` starts at
    // `b * stride + j * nrows`. Operations on a single column therefore reuse
    // the vector kernels against a window that starts at `j * nrows`; only the
    // ones that touch several columns per thread need a kernel of their own.

    /// `diag[b, row] = mat[.., row * nrows + row]`
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), requires = (diag.len() == n))]
    #[allow(clippy::too_many_arguments)]
    pub fn mat_get_diagonal(
        mut diag: DisjointSlice<f64>,
        mat: &[f64],
        n: u32,
        nrows: u32,
        mat_stride: u32,
        mat_nbatch: u32,
        nbatch: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < n as usize {
            let (b, row) = split(i, nrows);
            let mi = row * nrows as usize + row;
            let value = mat[broadcast_src(b, mat_stride, mat_nbatch, nbatch, mi)];
            if let Some(elem) = diag.get_mut(idx) {
                *elem = value;
            }
        }
    }

    /// `mat[b, row * nrows + row] = diag[.., row]`, on a matrix already zeroed.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    #[allow(clippy::too_many_arguments)]
    pub fn mat_from_diagonal(
        mut mat: DisjointSlice<f64>,
        diag: &[f64],
        n: u32,
        nrows: u32,
        mat_stride: u32,
        diag_stride: u32,
        diag_nbatch: u32,
        nbatch: u32,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        let (b, row) = split(i, nrows);
        let value = diag[broadcast_src(b, diag_stride, diag_nbatch, nbatch, row)];
        let mi = b * mat_stride as usize + row * nrows as usize + row;
        if mi < mat.len() {
            // SAFETY: in bounds by the check above. The destination is a
            // diagonal element, one per thread's own `(b, row)`, so no two
            // threads write the same slot.
            unsafe {
                *mat.as_mut_ptr().add(mi) = value;
            }
        }
    }

    /// `dest[b, dst_indices[j]] = src[.., src_indices[j]]`
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    #[allow(clippy::too_many_arguments)]
    pub fn mat_set_data_with_indices(
        mut dest: DisjointSlice<f64>,
        src: &[f64],
        dst_indices: &[i32],
        src_indices: &[i32],
        n: u32,
        nindices: u32,
        dest_stride: u32,
        src_stride: u32,
        src_nbatch: u32,
        nbatch: u32,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        let (b, j) = split(i, nindices);
        let value = src[broadcast_src(b, src_stride, src_nbatch, nbatch, src_indices[j] as usize)];
        let di = b * dest_stride as usize + dst_indices[j] as usize;
        if di < dest.len() {
            // SAFETY: in bounds by the check above, and distinct per thread for
            // distinct `dst_indices` -- see `vec_scatter`.
            unsafe {
                *dest.as_mut_ptr().add(di) = value;
            }
        }
    }

    /// `dest[b, elem] = x[..] + beta * y[..]`
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), requires = (dest.len() == n))]
    #[allow(clippy::too_many_arguments)]
    pub fn mat_scale_add_assign(
        mut dest: DisjointSlice<f64>,
        x: &[f64],
        y: &[f64],
        beta: f64,
        n: u32,
        nstates: u32,
        x_stride: u32,
        x_nbatch: u32,
        y_stride: u32,
        y_nbatch: u32,
        nbatch: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < n as usize {
            let (b, elem) = split(i, nstates);
            let xv = x[broadcast_src(b, x_stride, x_nbatch, nbatch, elem)];
            let yv = y[broadcast_src(b, y_stride, y_nbatch, nbatch, elem)];
            if let Some(elem) = dest.get_mut(idx) {
                *elem = xv + beta * yv;
            }
        }
    }

    /// `y[b, row] = alpha * sum_k w[k] * mat[.., row + k * nrows] + beta * y[b, row]`
    /// over `nc` columns of `mat`.
    ///
    /// This is a kernel rather than a `cublasDgemv` call because `w` comes from
    /// a small fixed *host* slice. The `mat` window starts at the beginning of
    /// the column range.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), requires = (y.len() == n))]
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_cols(
        mut y: DisjointSlice<f64>,
        mat: &[f64],
        w: [f64; MAX_SMALL_COLS],
        nc: u32,
        alpha: f64,
        beta: f64,
        n: u32,
        nstates: u32,
        nrows: u32,
        mat_stride: u32,
        mat_nbatch: u32,
        nbatch: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= n as usize {
            return;
        }
        let (b, row) = split(i, nstates);
        if row >= nrows as usize {
            return;
        }
        let base = broadcast_src(b, mat_stride, mat_nbatch, nbatch, row);
        // consecutive threads are consecutive rows within a column, so each
        // read is coalesced
        let mut acc = 0.0f64;
        for k in 0..nc as usize {
            acc += w[k] * mat[base + k * nrows as usize];
        }
        if let Some(elem) = y.get_mut(idx) {
            // beta == 0 must not read y: it may hold uninitialised values
            *elem = if beta == 0.0 {
                alpha * acc
            } else {
                alpha * acc + beta * *elem
            };
        }
    }

    ///  in-place gemm `C = C * B` where B is small.
    /// `mat[b, 0..ncols] = mat[b, 0..ncols] * rhs[0..ncols, 0..ncols]`
    /// in place where ncols is small, `rhs` column-major.
    ///
    /// Each thread owns one `(row, batch)` element, reads that row's `ncols`
    /// values into registers and writes the results back
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    #[allow(clippy::needless_range_loop)]
    pub fn mul_cols_by(
        mut mat: DisjointSlice<f64>,
        rhs: [f64; MAX_SMALL_COLS_SQ],
        n: u32,
        ncols: u32,
        nrows: u32,
        mat_stride: u32,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        let ncols = ncols as usize;
        if ncols == 0 {
            return;
        }
        let (b, row) = split(i, nrows);
        let base = b * mat_stride as usize + row;
        let stride = nrows as usize;
        if base + (ncols - 1) * stride >= mat.len() {
            return;
        }
        let ptr = mat.as_mut_ptr();

        let mut old = [0.0f64; MAX_SMALL_COLS];
        // SAFETY: bounds checked above. Every access below is at
        // `base + l * nrows` for `l < ncols`, and `base` is unique to this
        // thread's `(b, row)`, so the columns this thread reads and writes are
        // its own.
        unsafe {
            for l in 0..ncols {
                old[l] = *ptr.add(base + l * stride);
            }
            for j in 0..ncols {
                let mut acc = 0.0f64;
                for l in 0..ncols {
                    acc += old[l] * rhs[j * ncols + l];
                }
                *ptr.add(base + j * stride) = acc;
            }
        }
    }

    /// Fuses the BDF backward-difference table update into one launch:
    ///
    /// ```text
    /// diff[:, order+2] = d - diff[:, order+1]
    /// for i in (order+1 .. 0].rev(): diff[:, i] += diff[:, i+1]
    /// ```
    ///
    /// Each thread owns a `(row, batch)` element and loops over the
    /// columns of the difference table doing the loop described above
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    #[allow(clippy::too_many_arguments)]
    pub fn backward_diff_update(
        mut diff: DisjointSlice<f64>,
        d: &[f64],
        order: u32,
        n: u32,
        nrows: u32,
        diff_stride: u32,
        d_stride: u32,
        d_nbatch: u32,
        nbatch: u32,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        let (b, elem) = split(i, nrows);
        let dv = d[broadcast_src(b, d_stride, d_nbatch, nbatch, elem)];
        let base = b * diff_stride as usize + elem;
        let stride = nrows as usize;
        let order = order as usize;
        if base + (order + 2) * stride >= diff.len() {
            return;
        }
        let ptr = diff.as_mut_ptr();
        // SAFETY: bounds checked above, and every access is at
        // `base + i * nrows` for `i <= order + 2`, with `base` unique to this
        // thread's `(b, elem)`.
        unsafe {
            let mut carry = dv - *ptr.add(base + (order + 1) * stride);
            *ptr.add(base + (order + 2) * stride) = carry;
            let mut i = order + 1;
            loop {
                carry += *ptr.add(base + i * stride);
                *ptr.add(base + i * stride) = carry;
                if i == 0 {
                    break;
                }
                i -= 1;
            }
        }
    }

    /// Folds a non-negative `value` into `out[0]`, the reductions' one output.
    ///
    /// `out` is a single `f64` reinterpreted as a `u64`: for non-negative
    /// doubles the IEEE bit pattern is monotonic in the value, so an unsigned
    /// atomic max is an `f64` max. That detour exists because the float atomics
    /// have no `fetch_max` -- only load, store, `fetch_add`, `fetch_sub` and
    /// `swap`. Every value reaching here is a sum of squares or of `|x|^k`, so
    /// it is `>= 0` and never `-0.0`.
    ///
    /// The `> 0.0` guard skips both the identity and NaN, which is what the
    /// host's `if norm > max_norm` starting from zero used to do.
    fn atomic_max_into(out: &[DeviceAtomicU64], value: f64) {
        if value > 0.0 {
            out[0].fetch_max(value.to_bits(), RELAXED);
        }
    }

    /// Reduces `value` to the block maximum and folds that into `out[0]`.
    ///
    /// For callers whose threads each hold a finished lane sum: the small
    /// kernels and `lane_sum_max`.
    fn block_max_into(out: &[DeviceAtomicU64], value: f64) {
        static mut SMAX: SharedArray<f64, { BLOCK_SIZE as usize }> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        // SAFETY: each thread writes only its own slot, and the barrier below
        // separates it from any other thread's read.
        unsafe {
            SMAX[tid] = value;
        }
        thread::sync_threads();

        let mut s = BLOCK_SIZE as usize / 2;
        while s > 0 {
            if tid < s {
                // SAFETY: one owner per slot per round -- `tid` and `tid + s`,
                // which no other active thread holds -- and barriers between
                // rounds.
                unsafe {
                    if SMAX[tid] < SMAX[tid + s] {
                        SMAX[tid] = SMAX[tid + s];
                    }
                }
            }
            thread::sync_threads();
            s /= 2;
        }

        if tid == 0 {
            // SAFETY: slot 0 is written only by this thread, and the barrier
            // above closed the last round that wrote it.
            atomic_max_into(out, unsafe { SMAX[0] });
        }
    }

    /// Sums `local` across the block and publishes the block total.
    ///
    /// When this block owns its whole lane -- `blocks_per_lane == 1`, which
    /// holds whenever `nstates <= BLOCK_SIZE` or there are enough lanes to fill
    /// the device -- the block sum *is* the lane sum, so it goes straight into
    /// the output and no second pass is needed. Otherwise it is one slice of a
    /// lane and has to wait for `lane_sum_max` to add it to the lane's others.
    ///
    /// The caller computes the slot: a block can visit several lanes, so it is
    /// not derivable from `blockIdx.x` alone.
    fn publish_block_sum(
        out: &[DeviceAtomicU64],
        partials: &mut DisjointSlice<f64>,
        blocks_per_lane: u32,
        slot: usize,
        local: f64,
    ) {
        if let Some(total) = block_sum(local) {
            if blocks_per_lane == 1 {
                atomic_max_into(out, total);
            } else {
                // SAFETY: one slot per (lane, slice), thread 0 only, so no two
                // threads in the grid write the same element; `slot` is below
                // `blocks_per_lane * nbatch`, which the launch contract checks
                // `partials.len()` covers.
                unsafe {
                    *partials.as_mut_ptr().add(slot) = total;
                }
            }
        }
    }

    /// Second phase of a small reduction: publishes every thread's `term` to
    /// shared memory, then gives thread `c` the sum of lane `first + c` in
    /// index order, or `0.0` when it owns no lane.
    ///
    /// Phase 1 is the load, which differs per kernel; the block geometry comes
    /// from [`lane_block`] and the `term` from [`lane_element`].
    fn small_lane_sum(term: f64, first: usize, nstates: usize, cols: usize, nbatch: u32) -> f64 {
        static mut SDATA: SharedArray<f64, { BLOCK_SIZE as usize }> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        // SAFETY: each thread writes only its own slot, and the barrier below
        // separates it from the segment reads.
        unsafe {
            SDATA[tid] = term;
        }
        thread::sync_threads();

        let mut sum = 0.0f64;
        if tid < cols && first + tid < nbatch as usize {
            let base = tid * nstates;
            for j in 0..nstates {
                // SAFETY: shared memory is read-only after the barrier, and
                // `base + j < cols * nstates <= BLOCK_SIZE`.
                sum += unsafe { SDATA[base + j] };
            }
        }
        sum
    }

    /// Sums `local` across the block, returning the total in thread 0 and
    /// `None` in every other thread.
    ///
    /// Each warp folds its 32 values, publishes one total, and warp
    /// 0 folds those the same way.
    ///
    /// Trailing barrier means that this is safe to call repeatedly in a lane loop
    fn block_sum(local: f64) -> Option<f64> {
        static mut SWARP: SharedArray<f64, WARPS_PER_BLOCK> = SharedArray::UNINIT;

        let lane = warp::lane_id() as usize;
        let w = warp::warp_id() as usize;

        let warp_total = warp::reduce_sum_f64(local);
        if lane == 0 {
            // SAFETY: one slot per warp, written by its lane 0 only, and the
            // barrier below separates it from warp 0's read.
            unsafe {
                SWARP[w] = warp_total;
            }
        }
        thread::sync_threads();

        let mut total = None;
        if w == 0 {
            // SAFETY: read-only after the barrier, and `lane` is in bounds
            // under the guard. The 32 - `WARPS_PER_BLOCK` lanes with no slot
            // still join the shuffle, with the identity.
            let slot = if lane < WARPS_PER_BLOCK {
                unsafe { SWARP[lane] }
            } else {
                0.0
            };
            let sum = warp::reduce_sum_f64(slot);
            if lane == 0 {
                total = Some(sum);
            }
        }
        thread::sync_threads();
        total
    }

    // ========================================================================
    // Caller-supplied lane closures
    // ========================================================================

    /// Run `f` once per batch lane, on the lane slices of `outs` and `ins`.
    ///
    /// One thread per lane, which is the only parallelism available when `f` is
    /// opaque. The host only launches this when every operand in `outs` has the
    /// full lane count, so the threads write disjoint ranges.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn vec_for_each_batch<const M: usize, const N: usize, F>(
        f: F,
        outs: LaneArgsMut<M>,
        ins: LaneArgs<N>,
        nbatch: u32,
    ) where
        F: Fn([&mut [f64]; M], [&[f64]; N], usize) + Copy,
    {
        let b = thread::index_1d().get();
        if b >= nbatch as usize {
            return;
        }
        // SAFETY: each pointer is read out of a `Copy` byval struct, so the `M`
        // mutable slices borrow no shared owner, and lane `b` of an operand
        // with the full lane count is touched by this thread alone.
        let o = core::array::from_fn(|i| unsafe {
            let n = outs.nstates[i] as usize;
            core::slice::from_raw_parts_mut(outs.ptr[i].add(b * n), n)
        });
        // SAFETY: as above; a read operand with a smaller lane count is
        // broadcast, so several threads may read the same lane.
        let a = core::array::from_fn(|i| unsafe {
            let n = ins.nstates[i] as usize;
            let base = broadcast_src(b, ins.nstates[i], ins.nbatch[i], nbatch, 0);
            core::slice::from_raw_parts(ins.ptr[i].add(base), n)
        });
        f(o, a, b);
    }
}
