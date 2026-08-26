// Matrix-vector multiply over a range of columns, with host-side coefficients:
//   y = alpha * mat[:, start:start+nc] * x + beta * y
// Each thread owns one (row, batch) element and runs the short coefficient loop itself.
//
// This is a kernel rather than a cublasDgemv call because x is a *host* slice: cuBLAS requires
// device pointers for its vector arguments, so serving a host x through cuBLAS would mean a
// host-to-device copy on every Runge-Kutta stage. Passing the coefficients by value in the
// launch (at most MAX_SMALL_COLS of them, from cuda_kernels_common.h) avoids both the copy and
// any allocation.
struct GemvColsWeights_f64 {
    double w[MAX_SMALL_COLS];
};

__global__ void gemv_cols_f64(double* y, const double* __restrict__ mat,
                              struct GemvColsWeights_f64 x, int nc, double alpha, double beta,
                              int start, int nrows, int y_stride, int mat_stride,
                              int mat_nbatch) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;
    int b = blockIdx.y;
    int mat_base = (b % mat_nbatch) * mat_stride + start * nrows + row;
    int yi = b * y_stride + row;

    // consecutive threads are consecutive rows within a column, so each read is coalesced
    double acc = 0.0;
    for (int k = 0; k < nc; k++) {
        acc += x.w[k] * mat[mat_base + k * nrows];
    }
    // beta == 0 must not read y: it may hold uninitialised values.
    y[yi] = (beta == 0.0) ? alpha * acc : alpha * acc + beta * y[yi];
}
