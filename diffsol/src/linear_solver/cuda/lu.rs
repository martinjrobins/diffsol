use std::mem::MaybeUninit;

use crate::{CudaMat, CudaVec, LinearSolver, NonLinearOpJacobian, ScalarCuda, Matrix};
use cudarc::{driver::DevicePtrMut, cusolver::sys::{cusolverDnCreate, cusolverDnDestroy, cusolverDnDgetrf, cusolverDnDgetrf_bufferSize, cusolverDnDgetrs, cusolverDnHandle_t, cusolverStatus_t}, driver::CudaSlice};



pub struct CudaLu<T>
where
    T: ScalarCuda,
{
    lu: Option<FullPivLu<T>>,
    work: Option<CudaSlice<T>>,
    matrix: Option<CudaMat<T>>,
    handle: cusolverDnHandle_t,

}

impl<T> Default for CudaLu<T>
where
    T: ScalarCuda,
{
    fn default() -> Self {
        let handle = {
            let mut handle = MaybeUninit::uninit();
            unsafe {
                let stat = cusolverDnCreate(handle.as_mut_ptr());
                assert_eq!(stat, cusolverStatus_t::CUSOLVER_STATUS_SUCCESS);
                handle.assume_init()
            }
        };
        Self {
            lu: None,
            matrix: None,
            handle,
        }
    }
}

impl<T: ScalarCuda> LinearSolver<CudaMat<T>> for CudaLu<T> {
    fn set_linearisation<C: NonLinearOpJacobian<T = T, V = CudaVec<T>, M = CudaMat<T>>>(
        &mut self,
        op: &C,
        x: &CudaVec<T>,
        t: T,
    ) {
        let matrix = self.matrix.as_mut().expect("Matrix not set");
        op.jacobian_inplace(x, t, matrix);
        self.lu = Some(matrix.data.full_piv_lu());
    }

    fn solve_in_place(&self, x: &mut FaerVec<T>) -> Result<(), DiffsolError> {
        if self.lu.is_none() {
            return Err(linear_solver_error!(LuNotInitialized))?;
        }
        let lu = self.lu.as_ref().unwrap();
        lu.solve_in_place(x.data.as_mut());
        Ok(())
    }

    fn set_problem<
        C: NonLinearOpJacobian<T = T, V = CudaVec<T>, M = CudaMat<T>, C = CudaContext>,
    >(
        &mut self,
        op: &C,
    ) {
        let ncols = op.nstates();
        let nrows = op.nout();
        let matrix =
            C::M::new_from_sparsity(nrows, ncols, op.jacobian_sparsity(), op.context().clone());
        
        self.matrix = Some(matrix);
        
        /* step 3: query working space of getrf */
        let lwork = {
            let mut lwork = 0;
            let (a, _syn) = self.matrix.as_ref().unwrap().data.device_ptr_mut(op.context().stream);
            unsafe {
                cusolverDnDgetrf_bufferSize(self.handle, i32::try_from(nrows).unwrap(), i32::try_from(ncols).unwrap(), a as *mut f64, i32::try_from(nrows).unwrap(), &mut lwork);
            }
            lwork
        };
        self.work = Some(op.context().stream.alloc(lwork));
    }
}

//cusolverDnHandle_t cusolverH = NULL;
//cudaStream_t stream = NULL;
//
//const int m = 3;
//const int lda = m;
//const int ldb = m;
//
// *       | 1 2 3  |
// *   A = | 4 5 6  |
// *       | 7 8 10 |
// *
// * without pivoting: A = L*U
// *       | 1 0 0 |      | 1  2  3 |
// *   L = | 4 1 0 |, U = | 0 -3 -6 |
// *       | 7 2 1 |      | 0  0  1 |
// *
// * with pivoting: P*A = L*U
// *       | 0 0 1 |
// *   P = | 1 0 0 |
// *       | 0 1 0 |
// *
// *       | 1       0     0 |      | 7  8       10     |
// *   L = | 0.1429  1     0 |, U = | 0  0.8571  1.5714 |
// *       | 0.5714  0.5   1 |      | 0  0       -0.5   |
// */
//
//const std::vector<double> A = {1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 10.0};
//const std::vector<double> B = {1.0, 2.0, 3.0};
//std::vector<double> X(m, 0);
//std::vector<double> LU(lda * m, 0);
//std::vector<int> Ipiv(m, 0);
//int info = 0;
//
//double *d_A = nullptr; /* device copy of A */
//double *d_B = nullptr; /* device copy of B */
//int *d_Ipiv = nullptr; /* pivoting sequence */
//int *d_info = nullptr; /* error info */
//
//int lwork = 0;            /* size of workspace */
//double *d_work = nullptr; /* device workspace for getrf */
//
//const int pivot_on = 0;
//
//if (pivot_on) {
//    printf("pivot is on : compute P*A = L*U \n");
//} else {
//    printf("pivot is off: compute A = L*U (not numerically stable)\n");
//}
//
//printf("A = (matlab base-1)\n");
//print_matrix(m, m, A.data(), lda);
//printf("=====\n");
//
//printf("B = (matlab base-1)\n");
//print_matrix(m, 1, B.data(), ldb);
//printf("=====\n");
//
//CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
//
//CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
//CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));
//
//CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * A.size()));
//CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(double) * B.size()));
//CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Ipiv), sizeof(int) * Ipiv.size()));
//CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));
//
//CUDA_CHECK(
//    cudaMemcpyAsync(d_A, A.data(), sizeof(double) * A.size(), cudaMemcpyHostToDevice, stream));
//CUDA_CHECK(
//    cudaMemcpyAsync(d_B, B.data(), sizeof(double) * B.size(), cudaMemcpyHostToDevice, stream));
//
//CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(cusolverH, m, m, d_A, lda, &lwork));
//
//CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * lwork));
//
/* step 4: LU factorization */
//if (pivot_on) {
//    CUSOLVER_CHECK(cusolverDnDgetrf(cusolverH, m, m, d_A, lda, d_work, d_Ipiv, d_info));
//} else {
//    CUSOLVER_CHECK(cusolverDnDgetrf(cusolverH, m, m, d_A, lda, d_work, NULL, d_info));
//}
//
//if (pivot_on) {
//    CUDA_CHECK(cudaMemcpyAsync(Ipiv.data(), d_Ipiv, sizeof(int) * Ipiv.size(),
//                               cudaMemcpyDeviceToHost, stream));
//}
//CUDA_CHECK(
//    cudaMemcpyAsync(LU.data(), d_A, sizeof(double) * A.size(), cudaMemcpyDeviceToHost, stream));
//CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));
//
//CUDA_CHECK(cudaStreamSynchronize(stream));
//
//if (0 > info) {
//    printf("%d-th parameter is wrong \n", -info);
//    exit(1);
//}
//if (pivot_on) {
//    printf("pivoting sequence, matlab base-1\n");
//    for (int j = 0; j < m; j++) {
//        printf("Ipiv(%d) = %d\n", j + 1, Ipiv[j]);
//    }
//}
//printf("L and U = (matlab base-1)\n");
//print_matrix(m, m, LU.data(), lda);
//printf("=====\n");
//
//*
// * step 5: solve A*X = B
// *       | 1 |       | -0.3333 |
// *   B = | 2 |,  X = |  0.6667 |
// *       | 3 |       |  0      |
// *
// */
//if (pivot_on) {
//    CUSOLVER_CHECK(cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, m, 1, /* nrhs */
//                                    d_A, lda, d_Ipiv, d_B, ldb, d_info));
//} else {
//    CUSOLVER_CHECK(cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, m, 1, /* nrhs */
//                                    d_A, lda, NULL, d_B, ldb, d_info));
//}
//
//CUDA_CHECK(
//    cudaMemcpyAsync(X.data(), d_B, sizeof(double) * X.size(), cudaMemcpyDeviceToHost, stream));
//CUDA_CHECK(cudaStreamSynchronize(stream));
//
//printf("X = (matlab base-1)\n");
//print_matrix(m, 1, X.data(), ldb);
//printf("=====\n");
//
/* free resources */
//CUDA_CHECK(cudaFree(d_A));
//CUDA_CHECK(cudaFree(d_B));
//CUDA_CHECK(cudaFree(d_Ipiv));
//CUDA_CHECK(cudaFree(d_info));
//CUDA_CHECK(cudaFree(d_work));
//
//CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
//
//CUDA_CHECK(cudaStreamDestroy(stream));
//
//CUDA_CHECK(cudaDeviceReset());
//
//return EXIT_SUCCESS;
//}
