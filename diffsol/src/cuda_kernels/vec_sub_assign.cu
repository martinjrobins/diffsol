__global__ void vec_sub_assign_f64(double* lhs, double* rhs,
                                   int nstates,
                                   int lhs_stride,
                                   int rhs_stride, int rhs_nbatch) {
    int elem, li, ri;
    if (!batch_binary_setup(&elem, nstates,
                            &li, lhs_stride,
                            &ri, rhs_stride, rhs_nbatch)) return;
    lhs[li] -= rhs[ri];
}
