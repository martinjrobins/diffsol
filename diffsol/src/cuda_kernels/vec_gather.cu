/// gather values from `other` at the indices specified by `indices`
/// i.e. `self[i] = other[indices[i]]` for all i
__global__
void vec_gather_f64(double* self,
                   const double* __restrict__ other,
                   const int* __restrict__ indices,
                   int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Grid-stride loop for flexibility
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        int src_idx = indices[i];
        self[i] = other[src_idx];
    }
}