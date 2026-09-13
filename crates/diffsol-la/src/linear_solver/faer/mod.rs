pub mod lu;
pub mod sparse_lu;

#[cfg(test)]
mod tests {
    use crate::linear_solver::tests::diagonal_op_n;
    use crate::{
        Context, DenseMatrix, FaerContext, FaerMat, FaerSparseMat, FaerVec, LinearSolver, Matrix,
        Vector,
    };
    use faer::Par;

    /// Runs every faer kernel that can take a `Par`: dense/sparse `gemv`, LU factorisation and LU
    /// solve.
    fn exercise<M, LS>(ctx: FaerContext)
    where
        M: Matrix<T = f64, V = FaerVec<f64>, C = FaerContext>,
        LS: LinearSolver<M>,
    {
        let n = 64;
        let op = diagonal_op_n::<M>(n, 2.0, ctx);
        let mut s = LS::default();
        s.set_sparsity(&op);
        s.set_linearisation(&op);
        let b = ctx.vector_from_element::<FaerVec<f64>>(n, 1.0);
        let x = s.solve(&b).unwrap();
        let a = M::from_diagonal(&b);
        let mut y = ctx.vector_zeros::<FaerVec<f64>>(n);
        a.gemv(1.0, &x, 0.0, &mut y);
        y.assert_eq_st(&ctx.vector_from_element::<FaerVec<f64>>(n, 0.5), 1e-10);
    }

    /// `gemv_cols` is a second macro with its own `Par` argument, and dense only.
    fn exercise_dense(ctx: FaerContext) {
        let n = 64;
        let a = FaerMat::<f64>::from_diagonal(&ctx.vector_from_element::<FaerVec<f64>>(n, 2.0));
        let mut y = ctx.vector_zeros::<FaerVec<f64>>(n);
        let nc = crate::matrix::MAX_SMALL_COLS;
        a.gemv_cols(0, nc, 1.0, &vec![1.0; nc], 0.0, &mut y);
        let expected = (0..n).map(|i| if i < nc { 2.0 } else { 0.0 }).collect();
        y.assert_eq_st(&ctx.vector_from_vec::<FaerVec<f64>>(expected), 1e-10);
    }

    /// No faer call site may fall back on faer's process-global parallelism setting: they must all
    /// read [`FaerContext::par`].  `get_global_parallelism` panics once the global is disabled, so
    /// a call site that still queries it fails here.
    ///
    /// The global is process-wide, so this briefly affects any test running concurrently -- which
    /// is only safe because no diffsol code path reads it any more.  Restore it before returning.
    #[test]
    fn faer_kernels_never_query_the_global_parallelism() {
        for par in par_settings() {
            let mut ctx = FaerContext::default();
            ctx.par = par;
            faer::disable_global_parallelism();
            exercise::<FaerMat<f64>, super::lu::LU<f64>>(ctx);
            exercise::<FaerSparseMat<f64>, super::sparse_lu::FaerSparseLU<f64>>(ctx);
            exercise_dense(ctx);
            faer::set_global_parallelism(Par::Seq);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn par_settings() -> [Par; 2] {
        [Par::Seq, Par::rayon(2)]
    }

    #[cfg(target_arch = "wasm32")]
    fn par_settings() -> [Par; 1] {
        [Par::Seq]
    }
}
