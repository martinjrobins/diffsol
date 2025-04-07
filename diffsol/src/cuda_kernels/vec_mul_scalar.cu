__global__ void vec_mul_scalar_f64(double* vec, double scalar, double* res, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        res[i] = vec[i] * scalar;
    }
}