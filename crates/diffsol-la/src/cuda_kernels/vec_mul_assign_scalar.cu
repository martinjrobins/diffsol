__global__ void vec_mul_assign_scalar_f64(double* vec, double scalar,
                                           int nstates, int nbatch, int stride) {
    int b, elem;
    if (!batch_unary_setup(&b, &elem, nstates)) return;
    vec[b * stride + elem] *= scalar;
}
