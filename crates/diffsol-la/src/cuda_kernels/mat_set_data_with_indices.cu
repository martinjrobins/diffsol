__global__
void mat_set_data_with_indices_f64(double* self,
                                     const double* __restrict__ other,
                                     const int* __restrict__ dst_indices,
                                     const int* __restrict__ src_indices,
                                     int n,
                                     int self_stride, int self_nbatch,
                                     int other_stride, int other_nbatch) {
    int j, si, oi;
    if (!batch_set_data_setup(&j, n,
                              &si, self_stride, self_nbatch,
                              &oi, other_stride, other_nbatch,
                              dst_indices, src_indices)) return;
    self[si] = other[oi];
}
