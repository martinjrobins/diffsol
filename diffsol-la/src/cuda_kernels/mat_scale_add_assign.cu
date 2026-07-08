__global__ void mat_scale_add_assign_f64(double* self,
                                          const double* __restrict__ x,
                                          const double* __restrict__ y,
                                          double beta,
                                          int nstates,
                                          int self_stride,
                                          int x_stride, int x_nbatch,
                                          int y_stride, int y_nbatch) {
    int elem, si, xi, yi;
    if (!batch_ternary_setup(&elem, nstates,
                             &si, self_stride,
                             &xi, x_stride, x_nbatch,
                             &yi, y_stride, y_nbatch)) return;
    self[si] = x[xi] + beta * y[yi];
}
