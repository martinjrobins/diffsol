__global__ void vec_copy_f64(double* lhs, const double* rhs, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        lhs[i] = rhs[i];
    }
}