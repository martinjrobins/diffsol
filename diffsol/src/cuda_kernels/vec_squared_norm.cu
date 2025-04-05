#include <cuda_runtime.h>
#include <math.h>

__global__
void vec_squared_norm_f64(const double* __restrict__ y,
                          const double* __restrict__ y0,
                          const double* __restrict__ abs_tol,
                          double rel_tol,
                          int n,
                          double* partial_sums) {
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double local_sum = 0.0;

    // Grid-stride loop
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        double denom = fabs(y0[i]) * rel_tol + abs_tol[i];
        double ratio = y[i] / denom;
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
        partial_sums[blockIdx.x] = sdata[0];
}