__global__
void vec_gather_f64(double* self,
                   const double* __restrict__ other,
                   const int* __restrict__ indices,
                   int nindices,
                   int self_stride, int self_nbatch,
                   int other_stride, int other_nbatch) {
    int j, si, oi;
    if (!batch_gather_scatter_setup(&j, nindices,
                                    &si, self_stride, self_nbatch,
                                    &oi, other_stride, other_nbatch,
                                    indices)) return;
    self[si] = other[oi];
}
