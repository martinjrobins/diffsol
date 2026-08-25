// Fuses a weighted sum of a range of matrix columns into one kernel launch:
//   y = beta * y + sum_{k < nw} weights.w[k] * mat[:, start + k]
// Each thread owns one (row, batch) element and runs the short column loop
// itself, so a BDF predict/psi update is one launch instead of one per column.
//
// The weights are passed by value (at most WEIGHTED_COLUMN_SUM_MAX_WEIGHTS per
// launch) to avoid a host-to-device copy on every step; longer ranges are split
// into chunks by the caller, with beta = 1 for every chunk after the first.
#define WEIGHTED_COLUMN_SUM_MAX_WEIGHTS 8

struct WeightedColumnSumWeights_f64 {
    double w[WEIGHTED_COLUMN_SUM_MAX_WEIGHTS];
};

__global__ void weighted_column_sum_f64(double* y, const double* __restrict__ mat,
                                        struct WeightedColumnSumWeights_f64 weights,
                                        int nw, double beta, int start, int nrows,
                                        int y_stride, int mat_stride, int mat_nbatch) {
    int elem = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem >= nrows) return;
    int b = blockIdx.y;
    int mat_base = (b % mat_nbatch) * mat_stride + start * nrows + elem;
    int yi = b * y_stride + elem;

    double acc = 0.0;
    for (int k = 0; k < nw; k++) {
        acc += weights.w[k] * mat[mat_base + k * nrows];
    }
    // beta == 0 must not read y: it may hold uninitialised values.
    y[yi] = (beta == 0.0) ? acc : acc + beta * y[yi];
}
