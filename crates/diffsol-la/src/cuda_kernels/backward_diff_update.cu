// Fuses the BDF backward-difference table update into one kernel launch:
//   diff[:, order+2] = d - diff[:, order+1]
//   for i in (order+1 .. 0].rev(): diff[:, i] += diff[:, i+1]
// (each addition uses the just-updated value of column i+1). The recurrence is
// sequential per (row, batch) element but independent across elements, so each
// thread runs the short (order+2 <= 7) carry loop itself.
__global__ void backward_diff_update_f64(double* diff, const double* __restrict__ d,
                                          int order, int nrows,
                                          int diff_stride, int d_stride, int d_nbatch) {
    int elem = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem >= nrows) return;
    int b = blockIdx.y;
    int diff_base = b * diff_stride + elem;
    int d_idx = broadcast_batch(b, d_nbatch) * d_stride + elem;

    double carry = d[d_idx] - diff[diff_base + (order + 1) * nrows];
    diff[diff_base + (order + 2) * nrows] = carry;
    for (int i = order + 1; i >= 0; i--) {
        carry = diff[diff_base + i * nrows] + carry;
        diff[diff_base + i * nrows] = carry;
    }
}
