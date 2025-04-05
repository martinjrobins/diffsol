__global__ void vec_mul_assign_scalar_f64(double* vec, double scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        vec[i] *= scalar;
    }
}