// copies from a diagonal vector to a matrix (self, initialised with zeros, column-major order)
__global__
void mat_from_diagonal_f64(double* self_zeros,
                           const double* __restrict__ diagonal_vector,
                           int nrows) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = tid; i < nrows; i += blockDim.x * gridDim.x) {
        int idx = i * nrows + i; // column-major index
        self_zeros[idx] = diagonal_vector[i];
    }
}