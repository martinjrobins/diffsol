__global__
void vec_scatter_f64(const double* __restrict__ self,
                    const int* __restrict__ indices,
                    double* other,
                    int nindices,
                    int self_stride, int self_nbatch,
                    int other_stride, int other_nbatch) {
    int j, si, oi;
    if (!batch_gather_scatter_setup(&j, nindices,
                                    &si, self_stride, self_nbatch,
                                    &oi, other_stride, other_nbatch,
                                    indices)) return;
    other[oi] = self[si];
}
