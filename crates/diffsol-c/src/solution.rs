use std::any::Any;

use diffsol::{DefaultDenseMatrix, DenseMatrix, Solution as DiffsolSolution, Vector};

use crate::host_array::{HostArray, ToHostArray};

pub(crate) trait Solution: Any + Send + Sync {
    /// The solution values, as a `[nout, nt, nbatch]` array.
    fn get_ys(&self) -> HostArray;
    /// The solution times, as a `[nt]` array. All the lanes use the same times.
    fn get_ts(&self) -> HostArray;
    /// The sensitivities, as `nparams` arrays, each `[nout, nt, nbatch]`.
    fn get_sens(&self) -> Vec<HostArray>;
}

impl<V> Solution for DiffsolSolution<V>
where
    V: Vector + DefaultDenseMatrix + Send + Sync + 'static,
    <V as DefaultDenseMatrix>::M: DenseMatrix + Clone + Send + Sync + ToHostArray,
    Vec<V::T>: ToHostArray,
{
    fn get_sens(&self) -> Vec<HostArray> {
        self.y_sens
            .iter()
            .map(|s| {
                let mut s = s.clone();
                s.resize_cols(self.ts.len());
                s.to_host_array()
            })
            .collect()
    }

    fn get_ts(&self) -> HostArray {
        self.ts.clone().to_host_array()
    }

    fn get_ys(&self) -> HostArray {
        let mut ys = self.ys.clone();
        ys.resize_cols(self.ts.len());
        ys.to_host_array()
    }
}
