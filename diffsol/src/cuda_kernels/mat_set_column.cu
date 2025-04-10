// copies from a diagonal vector to a matrix (self, initialised with zeros, column-major order)
__global__
void mat_set_column_f64(double* self,
                        const double* __restrict__ column_vector,
                        int column_index,
                        int nrows) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = tid; i < nrows; i += blockDim.x * gridDim.x) {
        int idx = column_index * nrows + i; // column-major index
        self[idx] = column_vector[i];
    }
}