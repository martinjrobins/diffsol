// copies from a diagonal vector to a matrix (self, initialised with zeros, column-major order)
__global__
void mat_get_diagonal_f64(const double* __restrict__ self,
                           double* diagonal_vector,
                           int nrows) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = tid; i < nrows; i += blockDim.x * gridDim.x) {
        int idx = i * nrows + i; // column-major index
        diagonal_vector[i] = self[idx];
    }
}