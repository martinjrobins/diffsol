/// scatter values from `self` to `other` at the indices specified by `indices`
/// i.e. `other[indices[i]] = self[i]` for all i
__global__
void vec_scatter_f64(const double* __restrict__ self,
                    const int* __restrict__ indices,
                    double* other,
                    int num_indices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Grid-stride loop
    for (int i = idx; i < num_indices; i += blockDim.x * gridDim.x) {
        other[indices[i]] = self[i];
    }
}