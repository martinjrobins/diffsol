use sundials_sys::{sunindextype, SUNContext, SUNDenseMatrix, SUNMatDestroy, SUNMatrix};

#[derive(Debug)]
pub struct SundialsMatrix<'ctx> {
    sm: SUNMatrix,
    ctx: &'ctx SUNContext,
}

impl<'ctx> SundialsMatrix<'ctx> {
    pub fn new_dense(m: sunindextype, n: sunindextype, ctx: &'ctx SUNContext) -> Self {
        let sm = unsafe { SUNDenseMatrix(m, n, *ctx) };
        SundialsMatrix { sm, ctx }
    }
}

impl<'ctx> Drop for SundialsMatrix<'ctx> {
    fn drop(&mut self) {
        unsafe { SUNMatDestroy(self.sm) };
    }
}
