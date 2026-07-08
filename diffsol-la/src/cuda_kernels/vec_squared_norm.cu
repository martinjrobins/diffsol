#include <cuda_runtime.h>
#include <math.h>

// CUDA kernel to compute the squared norm of a vector
// using a relative and absolute tolerance
// The squared norm is computed as:
// sum_i (y_i / (|y0_i| * rel_tol + abs_tol_i))^2
// where y0 is the reference vector, abs_tol is the absolute tolerance vector,
// and rel_tol is the relative tolerance scalar.
// The result is stored in the partial_sums array, which is assumed to be large enough
// to hold the results of all blocks.
__global__
void vec_squared_norm_f64(const double* __restrict__ y,
                          const double* __restrict__ y0,
                          const double* __restrict__ abs_tol,
                          double rel_tol,
                          int nstates, int nbatch,
                          int y_stride, int y_nbatch,
                          int y0_stride, int y0_nbatch,
                          int atol_stride, int atol_nbatch,
                          double* partial_sums) {
    extern __shared__ double sdata[];

    int b = blockIdx.y;
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    double local_sum = 0.0;

    for (int i = idx; i < nstates; i += blockDim.x * gridDim.x) {
        // TODO: do we need all these batch guards?
        int yi = (b % y_nbatch) * y_stride + i;
        int y0i = (b % y0_nbatch) * y0_stride + i;
        int ati = (b % atol_nbatch) * atol_stride + i;
        double denom = fabs(y0[y0i]) * rel_tol + abs_tol[ati];
        double ratio = y[yi] / denom;
        local_sum += ratio * ratio;
    }

    // Store local sum in shared memory
    sdata[tid] = local_sum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Write result of this block to global memory
    if (tid == 0)
        partial_sums[b * gridDim.x + blockIdx.x] = sdata[0];
}
