use faer::{unzipped, zipped};

use crate::IndexType;

use super::{Vector, VectorCommon, VectorIndex, VectorView, VectorViewMut};
impl Vector for faer::Col<f64> {
    type View<'a> = faer::ColRef<'a, f64>;
    type ViewMut<'a> = faer::ColMut<'a, f64>;
    type Index = Vec<IndexType>;
    fn len(&self) -> IndexType {
        self.nrows()
    }
    fn norm(&self) -> f64 {
        self.norm_l2()
    }
    fn abs(&self) -> Self {
        zipped!(self).map(|unzipped!(xi)| xi.abs())
    }
    fn as_view(&self) -> Self::View<'_> {
        self.as_ref()
    }
    fn as_view_mut(&mut self) -> Self::ViewMut<'_> {
        self.as_mut()
    }
    fn copy_from(&mut self, other: &Self) {
        self.copy_from(other)
    }
    fn copy_from_view(&mut self, other: &Self::View<'_>) {
        self.copy_from(other)
    }
    fn from_element(nstates: usize, value: Self::T) -> Self {
        faer::Col::from_vec(vec![value; nstates])
    }
    fn from_vec(vec: Vec<Self::T>) -> Self {
        faer::Col::from_vec(vec)
    }
    fn zeros(nstates: usize) -> Self {
        Self::from_element(nstates, 0.0)
    }
    fn add_scalar_mut(&mut self, scalar: Self::T) {
        self = self + scalar
    }
    fn axpy(&mut self, alpha: Self::T, x: &Self, beta: Self::T) {
        // faer::linalg::matmul::matmul(
        //     self.as_mut(),
        //     self.as_ref(),
        //     x.as_ref(),
        //     Some(beta),
        //     alpha,
        //     faer::Parallelism::None,
        // );
        self = self * faer::scale(beta) + x * faer::scale(alpha)
    }
    fn component_mul_assign(&mut self, other: &Self) {
        // faer::linalg::matmul::matmul(
        //     self.as_mut(),
        //     self.as_ref(),
        //     other.as_ref(),
        //     None,
        //     1.0,
        //     faer::Parallelism::None,
        // );
        // zipped!(&mut self, &other).for_each(|unzipped!(s, o)| *s = *s + *o);
        self = self * other
    }
}

impl VectorIndex for Vec<IndexType> {
    fn zeros(len: IndexType) -> Self {
        vec![0; len as usize]
    }
    fn len(&self) -> IndexType {
        self.len() as IndexType
    }
}

impl VectorCommon for faer::Col<f64> {
    type T = f64;
    type Op = faer::Scale<f64>;
}
impl<'a> VectorCommon for faer::ColRef<'a, f64> {
    type T = f64;
    type Op = faer::Scale<f64>;
}
impl<'a> VectorCommon for faer::ColMut<'a, f64> {
    type T = f64;
    type Op = faer::Scale<f64>;
}

impl<'a> VectorView<'a> for faer::ColRef<'a, f64> {
    type Owned = faer::Col<f64>;
    fn abs(&self) -> faer::Col<f64> {
        zipped!(self).map(|unzipped!(xi)| xi.abs())
    }
    fn into_owned(self) -> faer::Col<f64> {
        self.to_owned()
    }
}

impl<'a> VectorViewMut<'a> for faer::ColMut<'a, f64> {
    type Owned = faer::Col<f64>;
    type View = faer::ColRef<'a, f64>;
    fn abs(&self) -> faer::Col<f64> {
        zipped!(self).map(|unzipped!(xi)| xi.abs())
    }
    fn copy_from(&mut self, other: &Self::Owned) {
        self.copy_from(other);
    }
    fn copy_from_view(&mut self, other: &Self::View) {
        self.copy_from(other);
    }
}
