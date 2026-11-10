use std::sync::Arc;

use diffsol::{DefaultDenseMatrix, DenseMatrix, Solution as DiffsolSolution, Vector};

use crate::tensor::{shared_tensor, Tensor, TensorLayout};

/// A solution of a solve. It does not change after the solve, so the tensors that the methods
/// give share its data, with no copy.
///
/// Each method takes an `Arc` of the solution: the tensor holds it, so the data stays valid
/// while the tensor exists, also after the [crate::SolutionWrapper] is dropped.
pub(crate) trait Solution: Send + Sync {
    /// The solution values, as a `[nout, nt, nbatch]` array.
    fn get_ys(self: Arc<Self>) -> Tensor;
    /// The solution times, as a `[nt]` array. All the lanes use the same times.
    fn get_ts(self: Arc<Self>) -> Tensor;
    /// The sensitivities, as `nparams` arrays, each `[nout, nt, nbatch]`.
    fn get_sens(self: Arc<Self>) -> Vec<Tensor>;
}

impl<V> Solution for DiffsolSolution<V>
where
    V: Vector + DefaultDenseMatrix + Send + Sync + 'static,
    <V as DefaultDenseMatrix>::M: DenseMatrix + Send + Sync + TensorLayout,
    Vec<V::T>: TensorLayout,
{
    fn get_sens(self: Arc<Self>) -> Vec<Tensor> {
        (0..self.y_sens.len())
            .map(|i| {
                let nt = self.ts.len();
                // a matrix can have more columns than times: keep a column for each time
                shared_tensor(self.clone(), |s| &s.y_sens[i]).truncate_axis(1, nt)
            })
            .collect()
    }

    fn get_ts(self: Arc<Self>) -> Tensor {
        shared_tensor(self, |s| &s.ts)
    }

    fn get_ys(self: Arc<Self>) -> Tensor {
        let nt = self.ts.len();
        // a solve to a final time grows `ys` with spare columns: keep a column for each time
        shared_tensor(self, |s| &s.ys).truncate_axis(1, nt)
    }
}
