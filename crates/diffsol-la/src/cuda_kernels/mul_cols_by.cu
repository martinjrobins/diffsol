// Right-multiplies a range of columns by a small matrix, in place:
//   mat[:, 0:ncols] = mat[:, 0:ncols] * rhs
// Each thread owns one (row, batch) element, reads that row's ncols values into registers and
// writes the ncols results back, so the update needs no scratch matrix. BLAS has no in-place
// C = C * B, which is why this is a kernel rather than a cublas call: as a gemm it would cost a
// second full-size device matrix plus a swap, purely to avoid aliasing.
//
// rhs is column-major and passed by value (at most MAX_SMALL_COLS x MAX_SMALL_COLS entries,
// from cuda_kernels_common.h), so the small host-computed coefficient block reaches the device
// without a copy per call.
struct MulColsByRhs_f64 {
    double m[MAX_SMALL_COLS * MAX_SMALL_COLS];
};

__global__ void mul_cols_by_f64(double* mat, struct MulColsByRhs_f64 rhs, int ncols, int nrows,
                                int mat_stride) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;
    int base = blockIdx.y * mat_stride + row;

    // consecutive threads are consecutive rows within a column, so each of these is coalesced
    double old[MAX_SMALL_COLS];
    for (int l = 0; l < ncols; l++) {
        old[l] = mat[base + l * nrows];
    }
    for (int j = 0; j < ncols; j++) {
        double acc = 0.0;
        for (int l = 0; l < ncols; l++) {
            acc += old[l] * rhs.m[j * ncols + l];
        }
        mat[base + j * nrows] = acc;
    }
}
