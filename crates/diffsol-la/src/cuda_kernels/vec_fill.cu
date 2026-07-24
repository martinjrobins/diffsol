__global__ void vec_fill_f64(double* lhs, double value,
                             int nstates, int nbatch, int stride) {
    int b, elem;
    if (!batch_unary_setup(&b, &elem, nstates)) return;
    lhs[b * stride + elem] = value;
}
