use std::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

use nalgebra::{Const, DMatrix, DVector, Dyn, LpNorm, MatrixView, MatrixViewMut};

use crate::{Context, IndexType, NalgebraContext, NalgebraMat, NalgebraScalar, Scale, VectorHost};

use super::{DefaultDenseMatrix, Vector, VectorCommon, VectorIndex, VectorView, VectorViewMut};

#[derive(Debug, Clone, PartialEq)]
pub struct NalgebraIndex {
    pub(crate) data: DVector<IndexType>,
    pub(crate) context: NalgebraContext,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NalgebraVec<T: NalgebraScalar> {
    pub(crate) data: DMatrix<T>,
    pub(crate) context: NalgebraContext,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NalgebraVecRef<'a, T: NalgebraScalar> {
    pub(crate) data: MatrixView<'a, T, Dyn, Dyn, Const<1>, Dyn>,
    pub(crate) context: NalgebraContext,
}

#[derive(Debug, PartialEq)]
pub struct NalgebraVecMut<'a, T: NalgebraScalar> {
    pub(crate) data: MatrixViewMut<'a, T, Dyn, Dyn, Const<1>, Dyn>,
    pub(crate) context: NalgebraContext,
}

impl<T: NalgebraScalar> NalgebraVec<T> {
    fn batch(&self, batch: usize) -> usize {
        batch % self.data.ncols()
    }
}

impl<T: NalgebraScalar> From<DVector<T>> for NalgebraVec<T> {
    fn from(data: DVector<T>) -> Self {
        Self {
            data: DMatrix::from_column_slice(data.len(), 1, data.as_slice()),
            context: NalgebraContext::default(),
        }
    }
}

impl<T: NalgebraScalar> DefaultDenseMatrix for NalgebraVec<T> {
    type M = NalgebraMat<T>;
}

macro_rules! common {
    ($t:ty, $inner:ty) => {
        impl<T: NalgebraScalar> VectorCommon for $t {
            type T = T;
            type C = NalgebraContext;
            type Inner = $inner;
            fn inner(&self) -> &Self::Inner {
                &self.data
            }
        }
    };
}
common!(NalgebraVec<T>, DMatrix<T>);
macro_rules! common_ref {
    ($t:ty, $inner:ty) => {
        impl<'a, T: NalgebraScalar> VectorCommon for $t {
            type T = T;
            type C = NalgebraContext;
            type Inner = $inner;
            fn inner(&self) -> &Self::Inner {
                &self.data
            }
        }
    };
}
common_ref!(
    NalgebraVecRef<'a, T>,
    MatrixView<'a, T, Dyn, Dyn, Const<1>, Dyn>
);
common_ref!(
    NalgebraVecMut<'a, T>,
    MatrixViewMut<'a, T, Dyn, Dyn, Const<1>, Dyn>
);

macro_rules! vec_binary {
    ($trait:ident, $method:ident, $lhs:ty, $rhs:ty, $op:tt, $binary:tt) => {
        impl<T: NalgebraScalar> $trait<$rhs> for $lhs {
            type Output = NalgebraVec<T>;

            fn $method(self, rhs: $rhs) -> Self::Output {
                self.context
                    .assert_compatible_nbatch(rhs.context.nbatch(), stringify!($method));
                if self.data.ncols() == rhs.data.ncols() {
                    return NalgebraVec {
                        data: &self.data $binary &rhs.data,
                        context: self.context,
                    };
                }
                let nb = self.data.ncols().max(rhs.data.ncols());
                let mut data = DMatrix::zeros(self.data.nrows(), nb);
                for b in 0..nb {
                    let mut column = data.column_mut(b);
                    column.copy_from(&self.data.column(b % self.data.ncols()));
                    column $op &rhs.data.column(b % rhs.data.ncols());
                }
                NalgebraVec {
                    data,
                    context: if self.data.ncols() == nb {
                        self.context
                    } else {
                        rhs.context
                    },
                }
            }
        }
    };
}
macro_rules! vec_assign {
    ($trait:ident, $method:ident, $lhs:ty, $rhs:ty, $op:tt) => {
        impl<T: NalgebraScalar> $trait<$rhs> for $lhs {
            fn $method(&mut self, rhs: $rhs) {
                self.context
                    .assert_compatible_nbatch(rhs.context.nbatch(), stringify!($method));
                if self.data.ncols() == rhs.data.ncols() {
                    self.data $op &rhs.data;
                    return;
                }
                for b in 0..self.data.ncols() {
                    let mut column = self.data.column_mut(b);
                    column $op &rhs.data.column(b % rhs.data.ncols());
                }
            }
        }
    };
}
macro_rules! binary_set {
    ($trait:ident,$method:ident,$op:tt,$binary:tt) => {
        vec_binary!(
            $trait,
            $method,
            NalgebraVec<T>,
            NalgebraVec<T>,
            $op,
            $binary
        );
        vec_binary!(
            $trait,
            $method,
            NalgebraVec<T>,
            &NalgebraVec<T>,
            $op,
            $binary
        );
        vec_binary!(
            $trait,
            $method,
            NalgebraVec<T>,
            NalgebraVecRef<'_, T>,
            $op,
            $binary
        );
        vec_binary!(
            $trait,
            $method,
            NalgebraVec<T>,
            &NalgebraVecRef<'_, T>,
            $op,
            $binary
        );
        vec_binary!(
            $trait,
            $method,
            NalgebraVecRef<'_, T>,
            NalgebraVec<T>,
            $op,
            $binary
        );
        vec_binary!(
            $trait,
            $method,
            NalgebraVecRef<'_, T>,
            &NalgebraVec<T>,
            $op,
            $binary
        );
        vec_binary!(
            $trait,
            $method,
            NalgebraVecRef<'_, T>,
            NalgebraVecRef<'_, T>,
            $op,
            $binary
        );
        vec_binary!(
            $trait,
            $method,
            NalgebraVecRef<'_, T>,
            &NalgebraVecRef<'_, T>,
            $op,
            $binary
        );
        vec_binary!(
            $trait,
            $method,
            &NalgebraVec<T>,
            NalgebraVec<T>,
            $op,
            $binary
        );
        vec_binary!(
            $trait,
            $method,
            &NalgebraVec<T>,
            &NalgebraVec<T>,
            $op,
            $binary
        );
        vec_binary!(
            $trait,
            $method,
            &NalgebraVec<T>,
            NalgebraVecRef<'_, T>,
            $op,
            $binary
        );
        vec_binary!(
            $trait,
            $method,
            &NalgebraVec<T>,
            &NalgebraVecRef<'_, T>,
            $op,
            $binary
        );
    };
}
binary_set!(Add, add, +=, +);
binary_set!(Sub, sub, -=, -);
macro_rules! assign_set {
    ($trait:ident,$method:ident,$op:tt) => {
        vec_assign!($trait, $method, NalgebraVec<T>, NalgebraVec<T>, $op);
        vec_assign!($trait, $method, NalgebraVec<T>, &NalgebraVec<T>, $op);
        vec_assign!($trait, $method, NalgebraVec<T>, NalgebraVecRef<'_, T>, $op);
        vec_assign!($trait, $method, NalgebraVec<T>, &NalgebraVecRef<'_, T>, $op);
        vec_assign!($trait, $method, NalgebraVecMut<'_, T>, NalgebraVec<T>, $op);
        vec_assign!($trait, $method, NalgebraVecMut<'_, T>, &NalgebraVec<T>, $op);
        vec_assign!(
            $trait,
            $method,
            NalgebraVecMut<'_, T>,
            NalgebraVecRef<'_, T>,
            $op
        );
        vec_assign!(
            $trait,
            $method,
            NalgebraVecMut<'_, T>,
            &NalgebraVecRef<'_, T>,
            $op
        );
    };
}
assign_set!(AddAssign, add_assign, +=);
assign_set!(SubAssign, sub_assign, -=);

macro_rules! scale {
    ($t:ty) => {
        impl<T: NalgebraScalar> Mul<Scale<T>> for $t {
            type Output = NalgebraVec<T>;
            fn mul(self, rhs: Scale<T>) -> Self::Output {
                NalgebraVec {
                    data: self.data.clone().into_owned() * rhs.value(),
                    context: self.context,
                }
            }
        }
    };
}
scale!(NalgebraVec<T>);
scale!(&NalgebraVec<T>);
scale!(NalgebraVecRef<'_, T>);
impl<T: NalgebraScalar> Mul<Scale<T>> for NalgebraVecMut<'_, T> {
    type Output = NalgebraVec<T>;
    fn mul(self, rhs: Scale<T>) -> Self::Output {
        NalgebraVec {
            data: self.data.into_owned() * rhs.value(),
            context: self.context,
        }
    }
}
impl<T: NalgebraScalar> Div<Scale<T>> for NalgebraVec<T> {
    type Output = Self;
    fn div(self, rhs: Scale<T>) -> Self {
        Self {
            data: self.data * (T::one() / rhs.value()),
            context: self.context,
        }
    }
}
impl<T: NalgebraScalar> MulAssign<Scale<T>> for NalgebraVec<T> {
    fn mul_assign(&mut self, rhs: Scale<T>) {
        self.data *= rhs.value();
    }
}
impl<T: NalgebraScalar> MulAssign<Scale<T>> for NalgebraVecMut<'_, T> {
    fn mul_assign(&mut self, rhs: Scale<T>) {
        self.data *= rhs.value();
    }
}
macro_rules! index {
    ($t:ty) => {
        impl<T: NalgebraScalar> Index<IndexType> for $t {
            type Output = T;
            fn index(&self, i: IndexType) -> &T {
                &self.data[(i, 0)]
            }
        }
    };
}
index!(NalgebraVec<T>);
index!(NalgebraVecRef<'_, T>);
impl<T: NalgebraScalar> IndexMut<IndexType> for NalgebraVec<T> {
    fn index_mut(&mut self, i: IndexType) -> &mut T {
        &mut self.data[(i, 0)]
    }
}

impl VectorIndex for NalgebraIndex {
    type C = NalgebraContext;
    fn zeros(len: IndexType, ctx: Self::C) -> Self {
        Self {
            data: DVector::zeros(len),
            context: ctx,
        }
    }
    fn len(&self) -> IndexType {
        self.data.len()
    }
    fn from_vec(v: Vec<IndexType>, ctx: Self::C) -> Self {
        Self {
            data: DVector::from_vec(v),
            context: ctx,
        }
    }
    fn clone_as_vec(&self) -> Vec<IndexType> {
        self.data.iter().copied().collect()
    }
    fn context(&self) -> &Self::C {
        &self.context
    }
}

impl<'a, T: NalgebraScalar> VectorView<'a> for NalgebraVecRef<'a, T> {
    type Owned = NalgebraVec<T>;
    fn get_index(&self, i: IndexType) -> T {
        assert_eq!(self.context.nbatch(), 1, "get_index requires nbatch == 1");
        self.data[(i, 0)]
    }
    fn into_owned(self) -> Self::Owned {
        NalgebraVec {
            data: self.data.into_owned(),
            context: self.context,
        }
    }
    fn squared_norm(&self, y: &Self::Owned, atol: &Self::Owned, rtol: T) -> T {
        self.context
            .assert_compatible_nbatch(y.context.nbatch(), "squared_norm");
        self.context
            .assert_compatible_nbatch(atol.context.nbatch(), "squared_norm");
        let mut max_norm = T::zero();
        for b in 0..self.data.ncols() {
            let yb = y.batch(b);
            let atolb = atol.batch(b);
            let mut norm = T::zero();
            for i in 0..self.data.nrows() {
                // Bounds follow shared row count and validated batch broadcasting above.
                let x = unsafe { *self.data.get_unchecked((i, b)) };
                let y = unsafe { *y.data.get_unchecked((i, yb)) };
                let atol = unsafe { *atol.data.get_unchecked((i, atolb)) };
                let term = x / (y.abs() * rtol + atol);
                norm += term * term;
            }
            max_norm = max_norm.max(norm / T::from_f64(self.data.nrows() as f64).unwrap());
        }
        max_norm
    }
}
impl<'a, T: NalgebraScalar> VectorViewMut<'a> for NalgebraVecMut<'a, T> {
    type Owned = NalgebraVec<T>;
    type View = NalgebraVecRef<'a, T>;
    type Index = NalgebraIndex;
    fn copy_from(&mut self, o: &Self::Owned) {
        self.context
            .assert_compatible_nbatch(o.context.nbatch(), "copy_from");
        if self.context.nbatch() == 1 && self.data.nrows() <= 10 {
            self.data.column_mut(0).copy_from(&o.data.column(0));
            return;
        }
        if self.data.ncols() == o.data.ncols() {
            self.data.copy_from(&o.data);
            return;
        }
        for b in 0..self.data.ncols() {
            self.data
                .column_mut(b)
                .copy_from(&o.data.column(b % o.data.ncols()));
        }
    }
    fn copy_from_view(&mut self, o: &Self::View) {
        self.context
            .assert_compatible_nbatch(o.context.nbatch(), "copy_from_view");
        if self.context.nbatch() == 1 && self.data.nrows() <= 10 {
            self.data.column_mut(0).copy_from(&o.data.column(0));
            return;
        }
        if self.data.ncols() == o.data.ncols() {
            self.data.copy_from(&o.data);
            return;
        }
        for b in 0..self.data.ncols() {
            self.data
                .column_mut(b)
                .copy_from(&o.data.column(b % o.data.ncols()));
        }
    }
    fn set_index(&mut self, i: IndexType, v: T) {
        self.data.row_mut(i).fill(v);
    }
    fn axpy(&mut self, a: T, x: &Self::Owned, beta: T) {
        self.context
            .assert_compatible_nbatch(x.context.nbatch(), "axpy");
        for b in 0..self.data.ncols() {
            self.data
                .column_mut(b)
                .axpy(a, &x.data.column(x.batch(b)), beta);
        }
    }
}
impl<T: NalgebraScalar> VectorHost for NalgebraVec<T> {
    fn as_slice(&self) -> &[T] {
        assert_eq!(self.context.nbatch(), 1);
        self.data.as_slice()
    }
    fn as_mut_slice(&mut self) -> &mut [T] {
        assert_eq!(self.context.nbatch(), 1);
        self.data.as_mut_slice()
    }
}
impl<T: NalgebraScalar> Vector for NalgebraVec<T> {
    type View<'a> = NalgebraVecRef<'a, T>;
    type ViewMut<'a> = NalgebraVecMut<'a, T>;
    type Index = NalgebraIndex;
    fn len(&self) -> IndexType {
        self.data.nrows()
    }
    fn inner_mut(&mut self) -> &mut Self::Inner {
        &mut self.data
    }
    fn context(&self) -> &Self::C {
        &self.context
    }
    fn norm(&self, k: i32) -> T {
        (0..self.data.ncols())
            .map(|b| self.data.column(b).apply_norm(&LpNorm(k)))
            .fold(T::zero(), |a, b| a.max(b))
    }
    fn get_index(&self, i: IndexType) -> T {
        assert_eq!(
            self.context.nbatch(),
            1,
            "get_index not supported for batched vectors"
        );
        self.data[(i, 0)]
    }
    fn set_index(&mut self, i: IndexType, v: T) {
        self.data.row_mut(i).fill(v);
    }
    fn squared_norm(&self, y: &Self, atol: &Self, rtol: T) -> T {
        self.context
            .assert_compatible_nbatch(y.context.nbatch(), "squared_norm");
        self.context
            .assert_compatible_nbatch(atol.context.nbatch(), "squared_norm");
        let mut max_norm = T::zero();
        for b in 0..self.data.ncols() {
            let yb = y.batch(b);
            let atolb = atol.batch(b);
            let mut norm = T::zero();
            for i in 0..self.data.nrows() {
                let x = unsafe { *self.data.get_unchecked((i, b)) };
                let y = unsafe { *y.data.get_unchecked((i, yb)) };
                let atol = unsafe { *atol.data.get_unchecked((i, atolb)) };
                let term = x / (y.abs() * rtol + atol);
                norm += term * term;
            }
            max_norm = max_norm.max(norm / T::from_f64(self.data.nrows() as f64).unwrap());
        }
        max_norm
    }
    fn as_view(&self) -> Self::View<'_> {
        NalgebraVecRef {
            data: self.data.as_view(),
            context: self.context,
        }
    }
    fn as_view_mut(&mut self) -> Self::ViewMut<'_> {
        NalgebraVecMut {
            data: self.data.as_view_mut(),
            context: self.context,
        }
    }
    fn get_batch(&self, b: usize) -> Self::View<'_> {
        assert!(b < self.data.ncols());
        NalgebraVecRef {
            data: unsafe {
                MatrixView::from_slice_with_strides_generic_unchecked(
                    self.data.as_slice(),
                    b * self.data.nrows(),
                    Dyn(self.data.nrows()),
                    Dyn(1),
                    Const,
                    Dyn(self.data.nrows()),
                )
            },
            context: NalgebraContext::default(),
        }
    }
    fn get_batch_mut(&mut self, b: usize) -> Self::ViewMut<'_> {
        assert!(b < self.data.ncols());
        let nrows = self.data.nrows();
        NalgebraVecMut {
            data: unsafe {
                MatrixViewMut::from_slice_with_strides_generic_unchecked(
                    self.data.as_mut_slice(),
                    b * nrows,
                    Dyn(nrows),
                    Dyn(1),
                    Const,
                    Dyn(nrows),
                )
            },
            context: NalgebraContext::default(),
        }
    }
    fn copy_from(&mut self, o: &Self) {
        self.as_view_mut().copy_from(o)
    }
    fn fill(&mut self, v: T) {
        self.data.fill(v)
    }
    fn copy_from_view(&mut self, o: &Self::View<'_>) {
        self.as_view_mut().copy_from_view(o)
    }
    fn from_element(n: usize, v: T, ctx: Self::C) -> Self {
        Self {
            data: DMatrix::from_element(n, ctx.nbatch(), v),
            context: ctx,
        }
    }
    fn from_vec(v: Vec<T>, ctx: Self::C) -> Self {
        assert!(
            v.len().is_multiple_of(ctx.nbatch()),
            "vector length must be divisible by nbatch"
        );
        Self {
            data: DMatrix::from_vec(v.len() / ctx.nbatch(), ctx.nbatch(), v),
            context: ctx,
        }
    }
    fn from_slice(v: &[T], ctx: Self::C) -> Self {
        Self::from_vec(v.to_vec(), ctx)
    }
    fn clone_as_vec(&self) -> Vec<T> {
        self.data.iter().copied().collect()
    }
    fn zeros(n: usize, ctx: Self::C) -> Self {
        Self {
            data: DMatrix::zeros(n, ctx.nbatch()),
            context: ctx,
        }
    }
    fn axpy(&mut self, a: T, x: &Self, b: T) {
        self.as_view_mut().axpy(a, x, b)
    }
    fn axpy_v(&mut self, a: T, x: &Self::View<'_>, b: T) {
        self.context
            .assert_compatible_nbatch(x.context.nbatch(), "axpy_v");
        for c in 0..self.data.ncols() {
            self.data
                .column_mut(c)
                .axpy(a, &x.data.column(c % x.data.ncols()), b);
        }
    }
    fn batched_axpy(&mut self, a: &[T], x: &Self, b: T) {
        assert_eq!(
            a.len(),
            self.context.nbatch(),
            "alpha.len() must equal nbatch"
        );
        for (c, a) in a.iter().copied().enumerate() {
            self.data
                .column_mut(c)
                .axpy(a, &x.data.column(x.batch(c)), b);
        }
    }
    fn component_div_assign(&mut self, o: &Self) {
        self.context
            .assert_compatible_nbatch(o.context.nbatch(), "component_div_assign");
        if self.data.ncols() == o.data.ncols() {
            self.data.component_div_assign(&o.data);
            return;
        }
        for c in 0..self.data.ncols() {
            self.data
                .column_mut(c)
                .component_div_assign(&o.data.column(o.batch(c)));
        }
    }
    fn component_mul_assign(&mut self, o: &Self) {
        self.context
            .assert_compatible_nbatch(o.context.nbatch(), "component_mul_assign");
        if self.data.ncols() == o.data.ncols() {
            self.data.component_mul_assign(&o.data);
            return;
        }
        for c in 0..self.data.ncols() {
            self.data
                .column_mut(c)
                .component_mul_assign(&o.data.column(o.batch(c)));
        }
    }
    fn root_finding(&self, g1: &Self) -> (bool, T, i32) {
        self.context
            .assert_compatible_nbatch(g1.context.nbatch(), "root_finding");
        let mut out = None;
        for b in 0..self.data.ncols() {
            let mut found = false;
            let mut frac = T::zero();
            let mut idx = -1;
            for i in 0..self.len() {
                let g0 = self.data[(i, b)];
                let g = g1.data[(i, g1.batch(b))];
                if g == T::zero() {
                    found = true
                }
                if g0 * g < T::zero() {
                    let q = (g / (g - g0)).abs();
                    if q > frac {
                        frac = q;
                        idx = i as i32
                    }
                }
            }
            if let Some(x) = out {
                assert_eq!(
                    x,
                    (found, frac, idx),
                    "root finding results differ across batches"
                )
            }
            out = Some((found, frac, idx));
        }
        out.unwrap_or((false, T::zero(), -1))
    }
    fn assign_at_indices(&mut self, idx: &Self::Index, v: T) {
        for b in 0..self.data.ncols() {
            for i in idx.data.iter() {
                self.data[(*i, b)] = v
            }
        }
    }
    fn copy_from_indices(&mut self, o: &Self, idx: &Self::Index) {
        for b in 0..self.data.ncols() {
            for i in idx.data.iter() {
                self.data[(*i, b)] = o.data[(*i, o.batch(b))]
            }
        }
    }
    fn gather(&mut self, o: &Self, idx: &Self::Index) {
        assert_eq!(self.len(), idx.len());
        for b in 0..self.data.ncols() {
            for (i, j) in idx.data.iter().enumerate() {
                self.data[(i, b)] = o.data[(*j, o.batch(b))]
            }
        }
    }
    fn scatter(&self, idx: &Self::Index, o: &mut Self) {
        assert_eq!(self.len(), idx.len());
        for b in 0..self.data.ncols() {
            for (i, j) in idx.data.iter().enumerate() {
                let batch = o.batch(b);
                o.data[(*j, batch)] = self.data[(i, b)]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_norm() {
        let v = NalgebraVec::from_vec(vec![1.0, -2.0, 3.0], Default::default());
        let y = NalgebraVec::from_vec(vec![1.0, 2.0, 3.0], Default::default());
        let atol = NalgebraVec::from_vec(vec![0.1, 0.2, 0.3], Default::default());
        let rtol = 0.1;
        let mut tmp = y.clone() * Scale(rtol);
        tmp += &atol;
        let mut r = v.clone();
        r.component_div_assign(&tmp);
        let errorn_check = r.data.column(0).norm_squared() / 3.0;
        assert_eq!(v.squared_norm(&y, &atol, rtol), errorn_check);
        let vview = v.as_view();
        assert_eq!(
            VectorView::squared_norm(&vview, &y, &atol, rtol),
            errorn_check
        );
    }

    #[test]
    fn test_root_finding() {
        super::super::tests::test_root_finding::<NalgebraVec<f64>>();
    }

    #[test]
    fn test_from_slice() {
        let slice = [1.0, 2.0, 3.0];
        let v = NalgebraVec::from_slice(&slice, Default::default());
        assert_eq!(v.clone_as_vec(), slice);
    }

    #[test]
    fn test_into() {
        let vector = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let v: NalgebraVec<f64> = vector.into();
        assert_eq!(v.clone_as_vec(), vec![1.0, 2.0, 3.0]);
    }

    super::super::generate_vector_tests_nonbatched!(nalgebra, NalgebraVec<f64>);
    super::super::generate_vector_tests_batched!(
        nalgebra,
        NalgebraVec<f64>,
        NalgebraContext::with_nbatch(2),
        NalgebraContext::with_nbatch(3)
    );
}
