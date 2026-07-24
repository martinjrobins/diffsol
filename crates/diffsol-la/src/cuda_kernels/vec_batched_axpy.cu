#include <cuda_runtime.h>
#include "cuda_kernels_common.h"

__global__ void vec_batched_axpy_f64(double* y, const double* __restrict__ x,
                                     const double* __restrict__ alpha,
                                     double beta,
                                     int nstates,
                                     int y_stride,
                                     int x_stride, int x_nbatch) {
    int elem, yi, xi_;
    if (!batch_binary_setup(&elem, nstates,
                            &yi, y_stride,
                            &xi_, x_stride, x_nbatch)) return;
    y[yi] = alpha[blockIdx.y] * x[xi_] + beta * y[yi];
}
