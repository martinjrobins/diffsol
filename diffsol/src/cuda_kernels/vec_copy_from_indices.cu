
/// copy from `other` at the indices specified by `indices`
/// generaly `self` and `other` have the same length
/// i.e. `self[indices[i]] = other[indices[i]]` for all i
__global__
void vec_copy_from_indices_f64(double* self,
                            const double* __restrict__ other,
                            const int* __restrict__ indices,
                            int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Grid-stride loop
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        int target_idx = indices[i];
        self[target_idx] = other[target_idx];
    }
}