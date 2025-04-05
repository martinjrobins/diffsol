__global__ void vec_sub_assign_f64(double* lhs, double* rhs, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        lhs[i] -= rhs[i];
    }
}