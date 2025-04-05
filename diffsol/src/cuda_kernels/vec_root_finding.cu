#include <cuda_runtime.h>
#include <math.h>
#include <limits>

/// given two vectors `g0=self` and `g1`, return:
/// - `true` if a root is found in g1 (i.e. g1_i == 0)
/// - for all values of i where a zero crossing is found (i.e. g0_i * g1_i < 0), return:
///     - max_i(abs(g1_i / (g1_i - g0_i))), 0 otherwise
///     - the index i at the maximum value, -1 otherwise
__global__
void vec_root_finding_f64(const double* __restrict__ g0,
                                const double* __restrict__ g1,
                                int n,
                                int* root_flag,
                                double* max_vals,
                                int* max_idxs) {
    extern __shared__ double sdata[];

    double* svals = sdata;
    int* sidxs = (int*)&svals[blockDim.x];

    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    double local_max = 0.0;
    int local_index = -1;

    // Grid-stride loop
    for (int i = global_idx; i < n; i += stride) {
        double v0 = g0[i];
        double v1 = g1[i];

        if (v1 == 0.0) {
            atomicExch(root_flag, 1);
        }

        if (v0 * v1 < 0.0) {
            double val = fabs(v1 / (v1 - v0));
            if (val > local_max) {
                local_max = val;
                local_index = i;
            }
        }
    }

    // Store local results in shared memory
    svals[tid] = local_max;
    sidxs[tid] = local_index;
    __syncthreads();

    // In-block parallel reduction for max val and corresponding index
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (svals[tid] < svals[tid + s]) {
                svals[tid] = svals[tid + s];
                sidxs[tid] = sidxs[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        max_vals[blockIdx.x] = svals[0];
        max_idxs[blockIdx.x] = sidxs[0];
    }
}