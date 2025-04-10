// assign the values in the `data` vector to the matrix at the indices in `dst_indices` from the indices in `src_indices`
// dst_index can be obtained using the `get_index` method on the Sparsity type
//      - for dense matrices, the dst_index is the data index in column-major order
__global__
void mat_set_data_with_indices_f64(double* self,
                                     const double* __restrict__ other,
                                     const int* __restrict__ dst_indices,
                                     const int* __restrict__ src_indices,
                                     int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        int dst_idx = dst_indices[i];
        int src_idx = src_indices[i];
        self[dst_idx] = other[src_idx];
    }
}