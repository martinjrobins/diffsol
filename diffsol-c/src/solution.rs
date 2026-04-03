use std::any::Any;

use diffsol::{
    DefaultDenseMatrix, DenseMatrix, MatrixCommon, Solution as DiffsolSolution, Vector,
    VectorCommon,
};

use crate::host_array::{HostArray, ToHostArray};

pub(crate) trait Solution: Any + Send + Sync {
    fn get_ys(&self) -> HostArray;
    fn get_ts(&self) -> HostArray;
    fn get_sens(&self) -> Vec<HostArray>;
}

impl<V> Solution for DiffsolSolution<V>
where
    V: Vector + DefaultDenseMatrix + Send + Sync + 'static,
    <V as DefaultDenseMatrix>::M: DenseMatrix + Clone + Send + Sync,
    <V as VectorCommon>::Inner: ToHostArray<V::T> + Clone,
    <<V as DefaultDenseMatrix>::M as MatrixCommon>::Inner: ToHostArray<V::T> + Clone,
{
    fn get_sens(&self) -> Vec<HostArray> {
        self.y_sens
            .iter()
            .map(|s| {
                let mut s = s.clone();
                s.resize_cols(self.ts.len());
                (*s.inner()).clone().to_host_array()
            })
            .collect()
    }

    fn get_ts(&self) -> HostArray {
        (*V::from_slice(&self.ts, V::C::default()).inner())
            .clone()
            .to_host_array()
    }

    fn get_ys(&self) -> HostArray {
        let mut ys = self.ys.clone();
        ys.resize_cols(self.ts.len());
        (*ys.inner()).clone().to_host_array()
    }
}
