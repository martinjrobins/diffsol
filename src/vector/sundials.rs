use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

use sundials_sys::{
    realtype, N_VAbs, N_VAddConst, N_VConst, N_VDestroy, N_VDiv, N_VGetArrayPointer,
    N_VGetLength_Serial, N_VLinearSum, N_VNew_Serial, N_VProd, N_VScale, N_VWrmsNorm_Serial,
    N_Vector, SUNContext,
};

use crate::IndexType;

use super::{Vector, VectorCommon, VectorIndex, VectorView, VectorViewMut};

#[derive(Debug)]
pub struct SundialsVector<'ctx> {
    nv: N_Vector,
    ctx: &'ctx SUNContext,
    // hack to allow to take a N_Vector and pass to an Op call (which needs a &V).
    // TODO: find a better way to handle this, perhaps Op calls should take a View?
    owned: bool,
}

impl<'ctx> SundialsVector<'ctx> {
    pub fn new_serial(len: usize, ctx: &'ctx SUNContext) -> Self {
        let nv = unsafe { N_VNew_Serial(len as i64, *ctx) };
        SundialsVector {
            nv,
            ctx,
            owned: true,
        }
    }
    pub fn new_not_owned(v: N_Vector, ctx: &'ctx SUNContext) -> Self {
        if v.is_null() {
            panic!("N_Vector is null");
        }
        SundialsVector {
            nv: v,
            ctx,
            owned: false,
        }
    }
}

impl<'ctx> Drop for SundialsVector<'ctx> {
    fn drop(&mut self) {
        if self.owned {
            unsafe { N_VDestroy(self.0) };
        }
    }
}

#[derive(Debug)]
struct SundialsVectorViewMut<'a, 'ctx>(&'a mut SundialsVector<'ctx>);

#[derive(Debug)]
struct SundialsVectorView<'a, 'ctx>(&'a SundialsVector<'ctx>);

#[derive(Debug)]
struct SundialsIndexVector(Vec<IndexType>);

impl Display for SundialsIndexVector {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "SundialsIndexVector")
    }
}

impl Display for SundialsVector<'_> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "SundialsVector")
    }
}

impl Display for SundialsVectorViewMut<'_, '_> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "SundialsVectorViewMut")
    }
}

impl Display for SundialsVectorView<'_, '_> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "SundialsVectorView")
    }
}

impl VectorCommon for SundialsVector<'_> {
    type T = realtype;
}

impl<'a> VectorCommon for SundialsVectorView<'_, 'a> {
    type T = realtype;
}

impl<'a> VectorCommon for SundialsVectorViewMut<'_, 'a> {
    type T = realtype;
}

macro_rules! impl_helper {
    ($trait:path, SundialsVector, $inner:tt) => {
        impl<'ctx> $trait for SundialsVector<'ctx> $inner
    };
    ($trait:path, SundialsVectorView, $inner:tt) => {
        impl<'a, 'ctx> $trait for SundialsVectorView<'a, 'ctx> $inner
    };
    ($trait:path, SundialsVectorViewMut, $inner:tt) => {
        impl<'a, 'ctx> $trait for SundialsVectorViewMut<'a, 'ctx> $inner
    };
}

// Clone
impl Clone for SundialsVector<'_> {
    fn clone(&self) -> Self {
        let mut z = SundialsVector::new(self.len(), self.ctx);
        z.copy_from(self);
        z
    }
}

// Index
macro_rules! impl_index {
    ($type:tt) => {
        impl_helper!(Index<IndexType>, $type, {
            type Output = <Self as VectorCommon>::T;
            fn index(&self, index: usize) -> &Self::Output {
                unsafe { N_VGetArrayPointer(self)[index] }
            }
        });
    };
}

impl_index!(SundialsVector);
impl_index!(SundialsVectorView);
impl_index!(SundialsVectorViewMut);

impl Index<IndexType> for SundialsIndexVector {
    type Output = IndexType;
    fn index(&self, index: IndexType) -> &Self::Output {
        &self[index as usize]
    }
}

// IndexMut
macro_rules! impl_index_mut {
    ($type:tt) => {
        impl_helper!(IndexMut<IndexType>, $type, {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                unsafe { N_VGetArrayPointer(self)[index] }
            }
        });
    };
}

impl_index_mut!(SundialsVector);
impl_index_mut!(SundialsVectorViewMut);

// div by realtype -> owned
macro_rules! impl_div {
    ($type:tt) => {
        impl_helper!(Div<realtype>, $type, {
            type Output = SundialsVector<'ctx>;
            fn div(self, rhs: realtype) -> Self::Output {
                let mut z = SundialsVector::new(self.len(), self.ctx);
                unsafe { N_VScale(1. / rhs, self, z) }
                z
            }
        });
    };
}

impl_div!(SundialsVector);
impl_div!(SundialsVectorView);
impl_div!(SundialsVectorViewMut);

// mul by realtype -> owned
macro_rules! impl_mul {
    ($type:tt) => {
        impl_helper!(Mul<realtype>, $type, {
            type Output = SundialsVector<'ctx>;
            fn mul(self, rhs: realtype) -> Self::Output {
                let mut z = SundialsVector::new(self.len(), self.ctx);
                unsafe { N_VScale(rhs, self, z) }
                z
            }
        });
    };
}

impl_mul!(SundialsVector);
impl_mul!(SundialsVectorView);
impl_mul!(SundialsVectorViewMut);

// div assign with realtype
macro_rules! impl_div_assign {
    ($type:tt) => {
        impl_helper!(DivAssign<realtype>, $type, {
            fn div_assign(&mut self, rhs: realtype) {
                unsafe { N_VScale(1. / rhs, self, self) }
            }
        });
    };
}

impl_div_assign!(SundialsVectorViewMut);
impl_div_assign!(SundialsVector);
impl_div_assign!(SundialsVectorView);

// mul assign with realtype
macro_rules! impl_mul_assign {
    ($type:tt) => {
        impl_helper!(MulAssign<realtype>, $type, {
            fn mul_assign(&mut self, rhs: realtype) {
                unsafe { N_VScale(rhs, self, self) }
            }
        });
    };
}

impl_mul_assign!(SundialsVectorViewMut);
impl_mul_assign!(SundialsVector);
impl_mul_assign!(SundialsVectorView);

// sub assign with owned and view
macro_rules! impl_sub_assign {
    ($type:tt, $rhs:ty) => {
        impl_helper!(SubAssign<$rhs>, $type, {
            fn sub_assign(&mut self, rhs: $rhs) {
                for i in 0..self.len() {
                    self[i] -= rhs[i];
                }
            }
        });
    };
}

impl_sub_assign!(SundialsVector, SundialsVector<'ctx>);
impl_sub_assign!(SundialsVector, &SundialsVector<'ctx>);
impl_sub_assign!(SundialsVector, SundialsVectorView<'_, 'ctx>);
impl_sub_assign!(SundialsVector, &SundialsVectorView<'_, 'ctx>);
impl_sub_assign!(SundialsVectorViewMut, SundialsVector<'ctx>);
impl_sub_assign!(SundialsVectorViewMut, &SundialsVector<'ctx>);
impl_sub_assign!(SundialsVectorViewMut, SundialsVectorView<'_, 'ctx>);
impl_sub_assign!(SundialsVectorViewMut, &SundialsVectorView<'_, 'ctx>);

// add assign
macro_rules! impl_add_assign {
    ($type:tt, $rhs:ty) => {
        impl_helper!(AddAssign<$rhs>, $type, {
            fn add_assign(&mut self, rhs: $rhs) {
                for i in 0..self.len() {
                    self[i] += rhs[i];
                }
            }
        });
    };
}

impl_add_assign!(SundialsVector, SundialsVector<'ctx>);
impl_add_assign!(SundialsVector, &SundialsVector<'ctx>);
impl_add_assign!(SundialsVector, SundialsVectorView<'_, 'ctx>);
impl_add_assign!(SundialsVector, &SundialsVectorView<'_, 'ctx>);
impl_add_assign!(SundialsVectorViewMut, SundialsVector<'ctx>);
impl_add_assign!(SundialsVectorViewMut, &SundialsVector<'ctx>);
impl_add_assign!(SundialsVectorViewMut, SundialsVectorView<'_, 'ctx>);
impl_add_assign!(SundialsVectorViewMut, &SundialsVectorView<'_, 'ctx>);

// sub by vector -> owned
macro_rules! impl_sub {
    ($type:tt, $rhs:ty) => {
        impl_helper!(Sub<$rhs>, $type, {
            type Output = SundialsVector<'ctx>;
            fn sub(self, rhs: $rhs) -> Self::Output {
                let mut z = SundialsVector::new(self.len(), self.ctx);
                z.copy_from(&self);
                z -= rhs;
                z
            }
        });
    };
}

impl_sub!(SundialsVector, SundialsVector<'ctx>);
impl_sub!(SundialsVector, &SundialsVector<'ctx>);
impl_sub!(SundialsVector, &SundialsVectorView<'_, 'ctx>);
impl_sub!(SundialsVector, SundialsVectorView<'_, 'ctx>);
impl_sub!(SundialsVectorView, &SundialsVector<'ctx>);
impl_sub!(SundialsVectorView, &SundialsVectorView<'_, 'ctx>);
impl_sub!(SundialsVectorView, SundialsVectorView<'_, 'ctx>);
impl_sub!(SundialsVectorView, SundialsVector<'ctx>);

// add by vector -> owned
macro_rules! impl_add {
    ($type:tt, $rhs:ty) => {
        impl_helper!(Add<$rhs>, $type, {
            type Output = SundialsVector<'ctx>;
            fn add(self, rhs: $rhs) -> Self::Output {
                let mut z = SundialsVector::new(self.len(), self.ctx);
                z.copy_from(&self);
                z += rhs;
                z
            }
        });
    };
}

impl_add!(SundialsVector, SundialsVector<'ctx>);
impl_add!(SundialsVector, &SundialsVector<'ctx>);
impl_add!(SundialsVector, &SundialsVectorView<'_, 'ctx>);
impl_add!(SundialsVector, SundialsVectorView<'_, 'ctx>);
impl_add!(SundialsVectorView, &SundialsVector<'ctx>);
impl_add!(SundialsVectorView, &SundialsVectorView<'_, 'ctx>);
impl_add!(SundialsVectorView, SundialsVectorView<'_, 'ctx>);
impl_add!(SundialsVectorView, SundialsVector<'ctx>);

impl<'a, 'ctx> VectorViewMut<'a> for SundialsVectorViewMut<'a, 'ctx> {
    type Owned = SundialsVector<'ctx>;
    type View = SundialsVectorView<'a, 'ctx>;
    fn abs(&self) -> Self::Owned {
        let z = SundialsVector::new(self.len(), self.ctx);
        unsafe { N_VAbs(self.0, z) }
        z
    }
    fn copy_from(&mut self, other: &Self::Owned) {
        unsafe { N_VScale(1.0, other.0, self) }
    }
    fn copy_from_view(&mut self, other: &Self::View) {
        unsafe { N_VScale(1.0, other.0, self) }
    }
}

impl<'a, 'ctx> VectorView<'a> for SundialsVectorView<'a, 'ctx> {
    type Owned = SundialsVector<'ctx>;
    fn abs(&self) -> Self::Owned {
        let z = SundialsVector::new(self.len(), self.ctx);
        unsafe { N_VAbs(self.0, z) }
        z
    }
    fn into_owned(self) -> Self::Owned {
        let z = SundialsVector::new(self.len(), self.ctx);
        z.copy_from_view(&self);
        z
    }
}

impl VectorIndex for SundialsIndexVector {
    fn zeros(len: IndexType) -> Self {
        vec![0; len]
    }
    fn len(&self) -> IndexType {
        self.len() as IndexType
    }
}

//impl<'ctx> Vector for SundialsVector<'ctx>
//where
//    'ctx: 'static,
//{
//    type View<'a> = SundialsVectorView<'a, 'ctx> where Self: 'a;
//    type ViewMut<'a> = SundialsVectorViewMut<'a, 'ctx> where Self: 'a;
//    type Index = SundialsIndexVector;
//    fn len(&self) -> IndexType {
//        unsafe { N_VGetLength_Serial(self.0) as IndexType }
//    }
//    fn norm(&self) -> Self::T {
//        unsafe { N_VWrmsNorm_Serial(self.0, self.0) }
//    }
//    fn is_empty(&self) -> bool {
//        self.len() == 0
//    }
//    fn abs(&self) -> Self {
//        let z = SundialsVector::new(self.len(), self.ctx);
//        unsafe { N_VAbs(self.0, z) }
//        z
//    }
//    fn add_scalar_mut(&mut self, scalar: Self::T) {
//        unsafe { N_VAddConst(self.0, scalar, self.0) }
//    }
//    fn as_view(&self) -> Self::View<'_> {
//        self.as_view()
//    }
//    fn as_view_mut(&mut self) -> Self::ViewMut<'_> {
//        self.as_view_mut()
//    }
//    fn axpy(&mut self, alpha: Self::T, x: &Self, beta: Self::T) {
//        unsafe { N_VLinearSum(alpha, x, beta, self.0, self.0) };
//    }
//    fn component_div_assign(&mut self, other: &Self) {
//        unsafe { N_VDiv(self.0, other.0, self.0) };
//    }
//    fn component_mul_assign(&mut self, other: &Self) {
//        unsafe { N_VProd(self.0, other.0, self.0) };
//    }
//    fn copy_from(&mut self, other: &Self) {
//        unsafe { N_VScale(1.0, other.0, self.0) }
//    }
//    fn copy_from_view(&mut self, other: &Self::View<'_>) {
//        unsafe { N_VScale(1.0, other.0, self.0) }
//    }
//    fn exp(&self) -> Self {
//        let mut z = SundialsVector::new(self.len(), self.ctx);
//        for i in 0..self.len() {
//            z[i] = self[i].exp();
//        }
//        z
//    }
//    fn filter_indices<F: Fn(Self::T) -> bool>(&self, f: F) -> Self::Index {
//        let mut indices = vec![];
//        for (i, &x) in self.iter().enumerate() {
//            if f(x) {
//                indices.push(i as IndexType);
//            }
//        }
//        SundialsIndexVector(indices)
//    }
//    fn from_element(nstates: usize, value: Self::T) -> Self {
//        let mut v = SundialsVector::new(nstates, self.ctx);
//        unsafe { N_VConst(value, v.0) };
//        v
//    }
//    fn from_vec(vec: Vec<Self::T>) -> Self {
//        let mut v = SundialsVector::new(vec.len());
//        for (i, &x) in vec.iter().enumerate() {
//            v[i] = x;
//        }
//        v
//    }
//    fn gather_from(&mut self, other: &Self, indices: &Self::Index) {
//        for (i, &index) in indices.iter().enumerate() {
//            self[i] = other[index];
//        }
//    }
//    fn scatter_from(&mut self, other: &Self, indices: &Self::Index) {
//        for (i, &index) in indices.iter().enumerate() {
//            self[index] = other[i];
//        }
//    }
//}
