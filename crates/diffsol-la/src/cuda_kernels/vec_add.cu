__global__ void vec_add_f64(double* lhs, double* rhs, double *ret,
                            int nstates,
                            int lhs_stride,
                            int rhs_stride, int rhs_nbatch,
                            int ret_stride, int ret_nbatch) {
    int elem, li, ri, oi;
    if (!batch_ternary_setup(&elem, nstates,
                             &li, lhs_stride,
                             &ri, rhs_stride, rhs_nbatch,
                             &oi, ret_stride, ret_nbatch)) return;
    ret[oi] = lhs[li] + rhs[ri];
}
