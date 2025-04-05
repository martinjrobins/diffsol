/// scatter values from `self` to `other` at the indices specified by `indices`
/// i.e. `other[indices[i]] = self[i]` for all i
__global__
void scatter_kernel(const double* __restrict__ self,
                    double* other,
                    const int* __restrict__ indices,
                    int num_indices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Grid-stride loop
    for (int i = idx; i < num_indices; i += blockDim.x * gridDim.x) {
        int target_idx = indices[i];
        other[target_idx] = self[i];
    }
}