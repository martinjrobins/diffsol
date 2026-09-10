use std::{
    collections::HashMap,
    sync::{Arc, LazyLock, Mutex},
};

use cuda_core::{
    simt::memory::{memcpy_dtod_async, memcpy_dtoh_async, memcpy_htod_async},
    sys::CUdeviceptr,
    CudaContext, CudaStream, DeviceBuffer, DriverError, LaunchConfig1D, LaunchConfig2D,
};
use cuda_device::atomic::DeviceAtomicU64;

use crate::{
    cuda_error,
    cuda_oxide_kernels::{kernels, BLOCK_SIZE},
    error::LaError,
};

/// A device, the kernel module loaded on it, and its SM count.
type DeviceEntry = (Arc<CudaContext>, Arc<kernels::LoadedModule>, u32);

/// Devices seen so far, with the kernel module loaded on each.
static DEVICES: LazyLock<Mutex<HashMap<usize, DeviceEntry>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Device scratch the reductions reuse, rather than allocating per call.
pub(crate) struct ReduceScratch {
    /// The one cell the cross-lane maximum lands in, as an IEEE bit pattern.
    pub(crate) out: DeviceBuffer<DeviceAtomicU64>,
    /// Per-lane partial sums, for the large kernels.
    pub(crate) partials: DeviceBuffer<f64>,
}

impl ReduceScratch {
    fn new(stream: &CudaStream, nbatch: usize, target_blocks: u32) -> Result<Self, LaError> {
        let fail = |e| cuda_error!(Other, format!("Failed to allocate scratch: {}", e));
        Ok(Self {
            out: DeviceBuffer::<u64>::zeroed(stream, 1)
                .map_err(fail)?
                .cast_elem(),
            partials: DeviceBuffer::zeroed(stream, nbatch.max(target_blocks as usize))
                .map_err(fail)?,
        })
    }
}

impl std::fmt::Debug for ReduceScratch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReduceScratch")
            .field("partials", &self.partials.len())
            .finish()
    }
}

/// Context for the `cuda-oxide` backend.
#[derive(Clone, Debug)]
pub struct OxideContext {
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) module: Arc<kernels::LoadedModule>,
    /// Blocks the large reductions aim to launch
    pub(crate) target_blocks: u32,
    /// Shared by both `Clone`` so two threads
    /// reducing through clones of one context serialize instead of clobbering
    /// each other's scratch.
    pub(crate) scratch: Arc<Mutex<ReduceScratch>>,
    nbatch: usize,
}

impl OxideContext {
    fn get_device_and_module(ordinal: usize) -> Result<DeviceEntry, LaError> {
        let mut devices = DEVICES.lock().unwrap();
        let (context, module, sms) = match devices.get(&ordinal) {
            Some(entry) => entry.clone(),
            None => {
                let context = CudaContext::new(ordinal)
                    .map_err(|e| cuda_error!(CudaInitializationError, e.to_string()))?;
                // SAFETY: this crate owns the device bundle embedded for
                // `cuda_oxide_kernels::kernels`, so the loaded artifact matches
                // the ABI the generated launch methods describe.
                let module = unsafe { kernels::load(&context) }.map_err(|e| {
                    cuda_error!(Other, format!("Failed to load kernel module: {}", e))
                })?;
                let sms = context
                    .multiprocessor_count()
                    .map_err(|e| cuda_error!(CudaInitializationError, e.to_string()))?;
                let entry = (context, Arc::new(module), sms);
                devices.insert(ordinal, entry.clone());
                entry
            }
        };
        context
            .bind_to_thread()
            .map_err(|e| cuda_error!(CudaInitializationError, e.to_string()))?;
        Ok((context, module, sms))
    }

    /// Creates a new context on the given device, using its default stream.
    pub fn new(ordinal: usize) -> Result<Self, LaError> {
        let (context, module, sms) = Self::get_device_and_module(ordinal)?;
        let stream = context.default_stream();
        let target_blocks = sms * 4;
        let scratch = ReduceScratch::new(&stream, 1, target_blocks)?;
        Ok(Self {
            stream,
            module,
            target_blocks,
            scratch: Arc::new(Mutex::new(scratch)),
            nbatch: 1,
        })
    }

    pub fn with_nbatch(self, nbatch: usize) -> Self {
        self.clone_with_nbatch_inner(nbatch)
            .expect("Failed to allocate reduction scratch")
    }

    /// `clone_with_nbatch` with the scratch resized, shared by the builder and
    /// the [`crate::Context`] method.
    fn clone_with_nbatch_inner(&self, nbatch: usize) -> Result<Self, LaError> {
        assert!(nbatch > 0, "nbatch must be > 0");
        let scratch = ReduceScratch::new(&self.stream, nbatch, self.target_blocks)?;
        Ok(Self {
            stream: self.stream.clone(),
            module: self.module.clone(),
            target_blocks: self.target_blocks,
            scratch: Arc::new(Mutex::new(scratch)),
            nbatch,
        })
    }

    /// Launch geometry for a flat kernel covering `n` work items, one thread
    /// each.
    pub(crate) fn config_1d(n: u32) -> LaunchConfig1D {
        Self::config_1d_blocks(n.div_ceil(BLOCK_SIZE))
    }

    /// Launch geometry for a flat kernel with an explicit block count, for the
    /// reductions.
    pub(crate) fn config_1d_blocks(nblocks: u32) -> LaunchConfig1D {
        LaunchConfig1D::new(nblocks, BLOCK_SIZE, 0)
    }

    /// Launch geometry for [`kernels::vec_root_finding`], the one kernel still
    /// on the 2-D shape: `grid.x` over the elements, `grid.y` over the batches,
    pub(crate) fn config_2d(nstates: u32, nbatch: u32) -> LaunchConfig2D {
        LaunchConfig2D::new((nstates.div_ceil(BLOCK_SIZE), nbatch), (BLOCK_SIZE, 1), 0)
    }
}

impl Default for OxideContext {
    fn default() -> Self {
        Self::new(0).unwrap()
    }
}

impl crate::Context for OxideContext {
    fn nbatch(&self) -> usize {
        self.nbatch
    }
    fn clone_with_nbatch(&self, nbatch: usize) -> Result<Self, LaError> {
        self.clone_with_nbatch_inner(nbatch)
    }
    fn synchronize(&self) {
        self.stream
            .synchronize()
            .expect("Failed to synchronize stream");
    }
}

/// Element `offset` of `buf` as a device pointer, checked to have `len`
/// elements of the allocation behind it.
///
/// (`cuda-core`'s copy methods all work on the whole buffer, so an offset copy
/// goes to the driver directly.)
fn offset_ptr<T>(buf: &DeviceBuffer<T>, offset: usize, len: usize) -> CUdeviceptr {
    assert!(
        offset + len <= buf.len(),
        "range {offset}..{} out of bounds for buffer of {}",
        offset + len,
        buf.len()
    );
    buf.cu_deviceptr() + (offset * std::mem::size_of::<T>()) as u64
}

/// Copies `dst.len()` elements from element `offset` of `buf` to the host,
/// synchronizing `stream` before returning, as `DeviceBuffer::copy_to_host`
/// does.
pub(crate) fn read_at<T>(
    stream: &CudaStream,
    buf: &DeviceBuffer<T>,
    offset: usize,
    dst: &mut [T],
) -> Result<(), DriverError> {
    let src = offset_ptr(buf, offset, dst.len());
    // SAFETY: `src` has `dst.len()` elements of `buf` behind it, and `dst` is a
    // host slice of that many. The synchronize below completes the copy before
    // the caller reads `dst`.
    unsafe {
        memcpy_dtoh_async(
            dst.as_mut_ptr(),
            src,
            std::mem::size_of_val(dst),
            stream.cu_stream(),
        )?;
    }
    stream.synchronize()
}

/// Copies `src` to `buf` starting at element `offset`, synchronizing `stream`
/// before returning so CUDA cannot still be reading the borrowed host slice
/// after the call, as `DeviceBuffer::copy_from_host` does.
pub(crate) fn write_at<T>(
    stream: &CudaStream,
    buf: &DeviceBuffer<T>,
    offset: usize,
    src: &[T],
) -> Result<(), DriverError> {
    let dst = offset_ptr(buf, offset, src.len());
    // SAFETY: `dst` has `src.len()` elements of `buf` behind it, and the
    // synchronize below keeps `src` alive for the whole copy.
    unsafe {
        memcpy_htod_async(
            dst,
            src.as_ptr(),
            std::mem::size_of_val(src),
            stream.cu_stream(),
        )?;
    }
    stream.synchronize()
}

/// Copies `len` elements from element `src_offset` of `src` to element
/// `dst_offset` of `dst`, enqueued on `stream`, as
/// `DeviceBuffer::copy_from_device_async` does.
///
/// The two ranges must not overlap.
pub(crate) fn copy_at<T>(
    stream: &CudaStream,
    dst: &DeviceBuffer<T>,
    dst_offset: usize,
    src: &DeviceBuffer<T>,
    src_offset: usize,
    len: usize,
) -> Result<(), DriverError> {
    if len == 0 {
        return Ok(());
    }
    let dst = offset_ptr(dst, dst_offset, len);
    let src = offset_ptr(src, src_offset, len);
    // SAFETY: both ranges are inside their allocations, and the callers copy
    // between two distinct buffers.
    unsafe { memcpy_dtod_async(dst, src, len * std::mem::size_of::<T>(), stream.cu_stream()) }
}
