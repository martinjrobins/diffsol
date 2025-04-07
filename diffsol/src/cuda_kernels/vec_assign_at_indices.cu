
/// assign `value` to the elements of `self` at the indices specified by `indices`
/// i.e. `self[indices[i]] = value` for all i
__global__
void vec_assign_at_indices_f64(double* self,
                                    const int* __restrict__ indices,
                                    double value,
                                    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Grid-stride loop for flexibility
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        self[indices[i]] = value;
    }
}