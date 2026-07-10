__global__
void vec_assign_at_indices_f64(double* self,
                                    const int* __restrict__ indices,
                                    double value,
                                    int nindices,
                                    int self_stride, int self_nbatch) {
    int j, si;
    if (!batch_assign_at_setup(&j, nindices,
                               &si, self_stride, self_nbatch,
                               indices)) return;
    self[si] = value;
}
