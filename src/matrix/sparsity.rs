use crate::{
    scalar::IndexType,
    vector::{Vector, VectorIndex},
};
use anyhow::Result;

use super::Matrix;

pub trait MatrixSparsity<M: Matrix>: Sized {
    fn nrows(&self) -> IndexType;
    fn ncols(&self) -> IndexType;
    fn is_sparse() -> bool;
    fn try_from_indices(
        nrows: IndexType,
        ncols: IndexType,
        indices: Vec<(IndexType, IndexType)>,
    ) -> Result<Self>;
    fn indices(&self) -> Vec<(IndexType, IndexType)>;
    fn union(self, other: M::SparsityRef<'_>) -> Result<M::Sparsity>;
    fn new_diagonal(n: IndexType) -> Self;
    fn as_ref(&self) -> M::SparsityRef<'_>;
}

pub trait MatrixSparsityRef<'a, M: Matrix> {
    fn nrows(&self) -> IndexType;
    fn ncols(&self) -> IndexType;
    fn is_sparse() -> bool;
    fn indices(&self) -> Vec<(IndexType, IndexType)>;
    fn to_owned(&self) -> M::Sparsity;
    fn get_index(&self, rows: &[IndexType], cols: &[IndexType]) -> <M::V as Vector>::Index;
}

pub struct Dense<M: Matrix> {
    nrows: IndexType,
    ncols: IndexType,
    _phantom: std::marker::PhantomData<M>,
}

pub struct DenseRef<'a, M: Matrix> {
    dense: &'a Dense<M>,
}

impl<M: Matrix> Dense<M> {
    pub fn new(nrows: IndexType, ncols: IndexType) -> Self {
        Dense {
            nrows,
            ncols,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, M: Matrix> DenseRef<'a, M> {
    pub fn new(dense: &'a Dense<M>) -> Self {
        DenseRef { dense }
    }
}

impl<M> MatrixSparsity<M> for Dense<M>
where
    for<'a> M: Matrix<Sparsity = Dense<M>, SparsityRef<'a> = DenseRef<'a, M>> + 'a,
{
    fn union(self, other: M::SparsityRef<'_>) -> Result<M::Sparsity> {
        if self.nrows() != other.nrows() || self.ncols() != other.ncols() {
            return Err(anyhow::anyhow!(
                "Cannot union matrices with different shapes"
            ));
        }
        Ok(Self::new(self.nrows(), self.ncols()))
    }

    fn as_ref(&self) -> M::SparsityRef<'_> {
        DenseRef::new(self)
    }

    fn nrows(&self) -> IndexType {
        self.nrows
    }

    fn ncols(&self) -> IndexType {
        self.ncols
    }

    fn is_sparse() -> bool {
        false
    }

    fn try_from_indices(
        nrows: IndexType,
        ncols: IndexType,
        _indices: Vec<(IndexType, IndexType)>,
    ) -> Result<Self> {
        if nrows == 0 || ncols == 0 {
            return Err(anyhow::anyhow!(
                "Cannot create a matrix with zero rows or columns"
            ));
        }
        Ok(Dense::new(nrows, ncols))
    }

    fn indices(&self) -> Vec<(IndexType, IndexType)> {
        Vec::new()
    }

    fn new_diagonal(n: IndexType) -> Self {
        Dense::new(n, n)
    }
}

impl<'a, M> MatrixSparsityRef<'a, M> for DenseRef<'a, M>
where
    M: Matrix<Sparsity = Dense<M>, SparsityRef<'a> = Self> + 'a,
{
    fn to_owned(&self) -> M::Sparsity {
        Dense::new(self.nrows(), self.ncols())
    }

    fn nrows(&self) -> IndexType {
        self.dense.nrows
    }

    fn ncols(&self) -> IndexType {
        self.dense.ncols
    }

    fn is_sparse() -> bool {
        false
    }

    fn get_index(&self, rows: &[IndexType], cols: &[IndexType]) -> <M::V as Vector>::Index {
        let indices: Vec<_> = rows
            .iter()
            .zip(cols.iter())
            .map(|(i, j)| {
                if i >= &self.nrows() || j >= &self.ncols() {
                    panic!("Index out of bounds")
                }
                j * self.nrows() + i
            })
            .collect();
        <M::V as Vector>::Index::from_slice(indices.as_slice())
    }

    fn indices(&self) -> Vec<(IndexType, IndexType)> {
        Vec::new()
    }
}
