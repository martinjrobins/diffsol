#include <cuda_runtime.h>
#include <math.h>

// CUDA kernel to compute the L2 norm of a vector across batches.
//
// gridDim.y = nbatch — each batch is handled by its own row of blocks.
// Each thread computes x[i]*x[i] for its element, accumulates via a
// grid-stride loop, then a tree reduction in shared memory produces
// a per-block partial sum stored in partial_sums.
__global__
void vec_norm_f64(const double* __restrict__ x,
                  int nstates, int nbatch,
                  int x_stride,
                  double* partial_sums) {
    extern __shared__ double sdata[];

    int b = blockIdx.y;
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    double local_sum = 0.0;

    for (int i = idx; i < nstates; i += blockDim.x * gridDim.x) {
        int xi = b * x_stride + i;
        double val = x[xi];
        local_sum += val * val;
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        partial_sums[b * gridDim.x + blockIdx.x] = sdata[0];
}
