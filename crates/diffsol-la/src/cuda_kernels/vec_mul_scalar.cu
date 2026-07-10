__global__ void vec_mul_scalar_f64(double* vec, double scalar, double* res,
                                   int nstates,
                                   int res_stride,
                                   int vec_stride, int vec_nbatch) {
    int elem, li, ri;
    if (!batch_binary_setup(&elem, nstates,
                            &li, res_stride,
                            &ri, vec_stride, vec_nbatch)) return;
    res[li] = vec[ri] * scalar;
}
