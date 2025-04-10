/// Perform the assignment self = x + beta * y where x and y are matrices and beta is a scalar
__global__ void mat_scale_add_assign_f64(double* self, 
                                         const double* __restrict__ x, 
                                         const double* __restrict__ y, 
                                         double beta, 
                                         int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        self[i] = x[i] + beta * y[i];
    }
}