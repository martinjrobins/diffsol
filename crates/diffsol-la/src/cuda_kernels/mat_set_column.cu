__global__
void mat_set_column_f64(double* self,
                        const double* __restrict__ column_vector,
                        int column_index,
                        int nrows,
                        int mat_stride, int mat_nbatch,
                        int col_stride, int col_nbatch) {
    int i, mi, ci;
    if (!batch_set_column_setup(&i, nrows,
                                &mi, mat_stride, mat_nbatch,
                                &ci, col_stride, col_nbatch,
                                column_index)) return;
    self[mi] = column_vector[ci];
}
