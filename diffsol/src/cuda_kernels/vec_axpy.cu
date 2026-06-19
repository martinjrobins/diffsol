__global__ void vec_axpy_f64(double* y, const double* __restrict__ x,
                             double alpha, double beta,
                             int nstates,
                             int y_stride,
                             int x_stride, int x_nbatch) {
    int elem, yi, xi_;
    if (!batch_binary_setup(&elem, nstates,
                            &yi, y_stride,
                            &xi_, x_stride, x_nbatch)) return;
    y[yi] = alpha * x[xi_] + beta * y[yi];
}
