__global__
void mat_get_diagonal_f64(const double* __restrict__ self,
                           double* diagonal_vector,
                           int nrows,
                           int mat_stride, int mat_nbatch,
                           int diag_stride, int diag_nbatch) {
    int i, mi, di;
    if (!batch_diagonal_setup(&i, nrows,
                              &mi, mat_stride, mat_nbatch,
                              &di, diag_stride, diag_nbatch)) return;
    diagonal_vector[di] = self[mi];
}
