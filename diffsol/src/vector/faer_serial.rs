use std::ops::{Div, Index, IndexMut, Mul, MulAssign, SubAssign, AddAssign, Sub, Add};
use std::slice;

use faer::{get_global_parallelism, unzip, zip, Col, ColMut, ColRef, Par};

use crate::{scalar::Scale, IndexType, Scalar, Vector};

use crate::{VectorCommon, VectorHost, VectorIndex, VectorView, VectorViewMut};

use super::utils::*;
use super::DefaultDenseMatrix;

#[derive(Clone)]
struct FaerContext {
    par: Par,
}

impl Default for FaerContext {
    fn default() -> Self {
        Self { par: get_global_parallelism() }
    }
}

#[derive(Debug, Clone)]
struct FaerVec<T: Scalar> {
    data: Col<T>,
    context: FaerContext,
}

#[derive(Debug, Clone)]
struct FaerVecRef<'a, T: Scalar> {
    data: ColRef<'a, T>,
    context: FaerContext,
}

#[derive(Debug, Clone)]
struct FaerVecMut<'a, T: Scalar> {
    data: ColMut<'a, T>,
    context: FaerContext,
}

impl<T: Scalar> DefaultDenseMatrix for FaerVec<T> {
    type M = FaerMat<T>;
}

impl_vector_common!(FaerVec<T>, FaerContext);
impl_vector_common!(FaerVecRef<'_, T>, FaerContext);
impl_vector_common!(FaerVecMut<'_, T>, FaerContext);

impl_mul_scalar!(FaerVec<T>, FaerVec<T>);
impl_mul_scalar!(FaerVecRef<'_, T>, FaerVec<T>);
impl_mul_scalar!(FaerVecMut<'_, T>, FaerVec<T>);
impl_div_scalar!(FaerVec<T>, FaerVec<T>);
impl_mul_assign_scalar!(FaerVecMut<'a, T>);
impl_mul_assign_scalar!(FaerVec<T>);

impl_sub_assign!(FaerVec<T>, FaerVec<T>);
impl_sub_assign!(FaerVec<T>, &FaerVec<T>);
impl_sub_assign!(FaerVec<T>, FaerVecRef<'_, T>);
impl_sub_assign!(FaerVec<T>, &FaerVecRef<'_, T>);

impl_sub_assign!(FaerVecMut<'_, T>, FaerVec<T>);
impl_sub_assign!(FaerVecMut<'_, T>, &FaerVec<T>);
impl_sub_assign!(FaerVecMut<'_, T>, FaerVecRef<'_, T>);
impl_sub_assign!(FaerVecMut<'_, T>, &FaerVecRef<'_, T>);

impl_add_assign!(FaerVec<T>, FaerVec<T>);
impl_add_assign!(FaerVec<T>, &FaerVec<T>);
impl_add_assign!(FaerVec<T>, FaerVecRef<'_, T>);
impl_add_assign!(FaerVec<T>, &FaerVecRef<'_, T>);

impl_add_assign!(FaerVecMut<'_, T>, FaerVec<T>);
impl_add_assign!(FaerVecMut<'_, T>, &FaerVec<T>);
impl_add_assign!(FaerVecMut<'_, T>, FaerVecRef<'_, T>);
impl_add_assign!(FaerVecMut<'_, T>, &FaerVecRef<'_, T>);

impl_sub!(FaerVec<T>, FaerVec<T>, FaerVec<T>);
impl_sub!(FaerVec<T>, &FaerVec<T>, FaerVec<T>);
impl_sub!(FaerVec<T>, FaerVecRef<'_, T>, FaerVec<T>);
impl_sub!(FaerVec<T>, &FaerVecRef<'_, T>, FaerVec<T>);

impl_sub!(FaerVecRef<'_, T>, FaerVec<T>, FaerVec<T>);
impl_sub!(FaerVecRef<'_, T>, &FaerVec<T>, FaerVec<T>);
impl_sub!(FaerVecRef<'_, T>, FaerVecRef<'_, T>, FaerVec<T>);
impl_sub!(FaerVecRef<'_, T>, &FaerVecRef<'_, T>, FaerVec<T>);

impl_add!(FaerVec<T>, FaerVec<T>, FaerVec<T>);
impl_add!(FaerVec<T>, &FaerVec<T>, FaerVec<T>);
impl_add!(FaerVec<T>, FaerVecRef<'_, T>, FaerVec<T>);
impl_add!(FaerVec<T>, &FaerVecRef<'_, T>, FaerVec<T>);

impl_add!(FaerVecRef<'_, T>, FaerVec<T>, FaerVec<T>);
impl_add!(FaerVecRef<'_, T>, &FaerVec<T>, FaerVec<T>);
impl_add!(FaerVecRef<'_, T>, FaerVecRef<'_, T>, FaerVec<T>);
impl_add!(FaerVecRef<'_, T>, &FaerVecRef<'_, T>, FaerVec<T>);

impl_index!(FaerVec<T>);


impl<T: Scalar> VectorHost for FaerVec<T> {
    fn as_mut_slice(&mut self) -> &mut [Self::T] {
        unsafe { slice::from_raw_parts_mut(self.data.as_ptr_mut(), self.len()) }
    }
    fn as_slice(&self) -> &[Self::T] {
        unsafe { slice::from_raw_parts(self.data.as_ptr(), self.len()) }
    }
}


impl<T: Scalar> Vector for FaerVec<T> {
    type View<'a> = FaerVecRef<'a, T>;
    type ViewMut<'a> = FaerVecMut<'a, T>;
    type Index = Vec<IndexType>;
    fn context(&self) -> &Self::C {
        &self.context
    }
    fn len(&self) -> IndexType {
        self.data.nrows()
    }
    fn get_index(&self, index: IndexType) -> Self::T {
        self.data[index]
    }
    fn set_index(&mut self, index: IndexType, value: Self::T) {
        self.data[index] = value;
    }
    fn norm(&self, k: i32) -> T {
        match k {
            1 => self.data.norm_l1(),
            2 => self.data.norm_l2(),
            _ => self.data
                .iter()
                .fold(T::zero(), |acc, x| acc + x.pow(k))
                .pow(T::one() / T::from(k as f64)),
        }
    }

    fn squared_norm(&self, y: &Self, atol: &Self, rtol: Self::T) -> Self::T {
        let mut acc = T::zero();
        if y.len() != self.len() || y.len() != atol.len() {
            panic!("Vector lengths do not match");
        }
        for i in 0..self.len() {
            let yi = unsafe { y.get_unchecked(i) };
            let ai = unsafe { atol.get_unchecked(i) };
            let xi = unsafe { self.data.get_unchecked(i) };
            acc += (*xi / (yi.abs() * rtol + *ai)).powi(2);
        }
        acc / Self::T::from(self.len() as f64)
    }
    fn as_view(&self) -> Self::View<'_> {
        FaerVecRef { data: self.data.as_ref(), context: self.context.clone() }
    }
    fn as_view_mut(&mut self) -> Self::ViewMut<'_> {
        FaerVecMut { data: self.data.as_mut(), context: self.context.clone() }
    }
    fn copy_from(&mut self, other: &Self) {
        self.data.copy_from(&other.data)
    }
    fn copy_from_view(&mut self, other: &Self::View<'_>) {
        self.data.copy_from(&other.data)
    }
    fn fill(&mut self, value: Self::T) {
        self.data.iter_mut().for_each(|s| *s = value);
    }
    fn from_element(nstates: usize, value: Self::T, ctx: Self::C) -> Self {
        let data = Col::from_vec(vec![value; nstates], ctx);
        FaerVec { data, context: ctx }
    }
    fn from_vec(vec: Vec<Self::T>, ctx: Self::C) -> Self {
        let data = Col::from_fn(vec.len(), |i| vec[i]);
        FaerVec { data, context: ctx }
    }
    fn clone_as_vec(&self) -> Vec<Self::T> {
        self.data.iter().cloned().collect()
    }
    fn zeros(nstates: usize, ctx: Self::C) -> Self {
        Self::from_element(nstates, T::zero(), ctx)
    }
    fn axpy(&mut self, alpha: Self::T, x: &Self, beta: Self::T) {
        zip!(self.data.as_mut(), x.data.as_view()).for_each(|unzip!(si, xi)| *si = *si * beta + *xi * alpha);
    }
    fn axpy_v(&mut self, alpha: Self::T, x: &Self::View<'_>, beta: Self::T) {
        zip!(self.data.as_mut(), x.data).for_each(|unzip!(si, xi)| *si = *si * beta + *xi * alpha);
    }
    fn component_mul_assign(&mut self, other: &Self) {
        zip!(self.data.as_mut(), other.data.as_view()).for_each(|unzip!(s, o)| *s *= *o);
    }
    fn component_div_assign(&mut self, other: &Self) {
        zip!(self.data.as_mut(), other.data.as_view()).for_each(|unzip!(s, o)| *s /= *o);
    }

    fn root_finding(&self, g1: &Self) -> (bool, Self::T, i32) {
        let mut max_frac = T::zero();
        let mut max_frac_index = -1;
        let mut found_root = false;
        assert_eq!(self.len(), g1.len(), "Vector lengths do not match");
        for i in 0..self.len() {
            let g0 = unsafe { *self.data.get_unchecked(i) };
            let g1 = unsafe { *g1.data.get_unchecked(i) };
            if g1 == T::zero() {
                found_root = true;
            }
            if g0 * g1 < T::zero() {
                let frac = (g1 / (g1 - g0)).abs();
                if frac > max_frac {
                    max_frac = frac;
                    max_frac_index = i as i32;
                }
            }
        }
        (found_root, max_frac, max_frac_index)
    }

    fn assign_at_indices(&mut self, indices: &Self::Index, value: Self::T) {
        for i in indices {
            self[*i] = value;
        }
    }

    fn copy_from_indices(&mut self, other: &Self, indices: &Self::Index) {
        for i in indices {
            self[*i] = other[*i];
        }
    }

    fn gather(&mut self, other: &Self, indices: &Self::Index) {
        assert_eq!(self.len(), indices.len(), "Vector lengths do not match");
        for (s, o) in self.iter_mut().zip(indices.iter()) {
            *s = other[*o];
        }
    }

    fn scatter(&self, indices: &Self::Index, other: &mut Self) {
        assert_eq!(self.len(), indices.len(), "Vector lengths do not match");
        for (s, o) in self.iter().zip(indices.iter()) {
            other[*o] = *s;
        }
    }
}

impl VectorIndex for Vec<IndexType> {
    type C = FaerContext;
    fn zeros(len: IndexType) -> Self {
        vec![0; len]
    }
    fn len(&self) -> IndexType {
        self.len() as IndexType
    }
    fn from_vec(v: Vec<IndexType>) -> Self {
        v
    }
    fn clone_as_vec(&self) -> Vec<IndexType> {
        self.clone()
    }
}


impl<'a, T: Scalar> VectorView<'a> for FaerVecRef<'a, T> {
    type Owned = FaerVec<T>;
    fn into_owned(self) -> FaerVec<T> {
        FaerVec { data: self.data.to_owned(), context: self.context }
    }
    fn squared_norm(&self, y: &Self::Owned, atol: &Self::Owned, rtol: Self::T) -> Self::T {
        let mut acc = T::zero();
        if y.len() != self.nrows() || y.nrows() != atol.nrows() {
            panic!("Vector lengths do not match");
        }
        for i in 0..self.nrows() {
            let yi = unsafe { y.get_unchecked(i) };
            let ai = unsafe { atol.get_unchecked(i) };
            let xi = unsafe { self.get_unchecked(i) };
            acc += (*xi / (yi.abs() * rtol + *ai)).powi(2);
        }
        acc / Self::T::from(self.nrows() as f64)
    }
}

impl<'a, T: Scalar> VectorViewMut<'a> for FaerVecMut<'a, T> {
    type Owned = FaerVec<T>;
    type View = FaerVecRef<'a, T>;
    type Index = Vec<IndexType>;
    fn copy_from(&mut self, other: &Self::Owned) {
        self.copy_from(other);
    }
    fn copy_from_view(&mut self, other: &Self::View) {
        self.copy_from(other);
    }
    fn axpy(&mut self, alpha: Self::T, x: &Self::Owned, beta: Self::T) {
        zip!(self.as_mut(), x.as_view()).for_each(|unzip!(si, xi)| *si = *si * beta + *xi * alpha);
    }
}

// tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::scale;

    #[test]
    fn test_mult() {
        let v = FaerVec::from_vec(vec![1.0, -2.0, 3.0], Default::default());
        let s = scale(2.0);
        let r = FaerVec::from_vec(vec![2.0, -4.0, 6.0], Default::default());
        assert_eq!(v * s, r);
    }

    #[test]
    fn test_mul_assign() {
        let mut v = FaerVec::from_vec(vec![1.0, -2.0, 3.0], Default::default());
        let s = scale(2.0);
        let r = FaerVec::from_vec(vec![2.0, -4.0, 6.0], Default::default());
        v.mul_assign(s);
        assert_eq!(v, r);
    }

    #[test]
    fn test_error_norm() {
        let v = FaerVec::from_vec(vec![1.0, -2.0, 3.0], Default::default());
        let y = FaerVec::from_vec(vec![1.0, 2.0, 3.0], Default::default());
        let atol = FaerVec::from_vec(vec![0.1, 0.2, 0.3], Default::default());
        let rtol = 0.1;
        let mut tmp = y.clone() * scale(rtol);
        tmp += &atol;
        let mut r = v.clone();
        r.component_div_assign(&tmp);
        let errorn_check = r.squared_norm_l2() / 3.0;
        assert!(
            (v.squared_norm(&y, &atol, rtol) - errorn_check).abs() < 1e-10,
            "{} vs {}",
            v.squared_norm(&y, &atol, rtol),
            errorn_check
        );
        assert!(
            (v.as_ref().squared_norm(&y, &atol, rtol) - errorn_check).abs() < 1e-10,
            "{} vs {}",
            v.as_ref().squared_norm(&y, &atol, rtol),
            errorn_check
        );
    }

    #[test]
    fn test_root_finding() {
        super::super::tests::test_root_finding::<Col<f64>>();
    }
}
