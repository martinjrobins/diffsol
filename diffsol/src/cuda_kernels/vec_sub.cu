__global__ void vec_sub_f64(double* lhs, double* rhs, double *ret, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        ret[i] = lhs[i] - rhs[i];
    }
}