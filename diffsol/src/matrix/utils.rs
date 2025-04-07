macro_rules! impl_matrix_common {
    ($mat:ty, $vec:ty, $con:ty, $in:ty) => {
        impl<T: Scalar> MatrixCommon for $mat {
            type T = T;
            type V = $vec;
            type C = $con;
            type Inner = $in;

            fn nrows(&self) -> IndexType {
                self.data.nrows()
            }
            fn ncols(&self) -> IndexType {
                self.data.ncols()
            }
            fn inner(&self) -> &Self::Inner {
                &self.data
            }
        }
    };
}

pub(crate) use impl_matrix_common;

macro_rules! impl_matrix_common_ref {
    ($mat:ty, $vec:ty, $con:ty, $in:ty) => {
        impl<'a, T: Scalar> MatrixCommon for $mat {
            type T = T;
            type V = $vec;
            type C = $con;
            type Inner = $in;

            fn nrows(&self) -> IndexType {
                self.data.nrows()
            }
            fn ncols(&self) -> IndexType {
                self.data.ncols()
            }
            fn inner(&self) -> &Self::Inner {
                &self.data
            }
        }
    };
}

pub(crate) use impl_matrix_common_ref;

macro_rules! impl_add {
    ($lhs:ty, $rhs:ty, $out:ty) => {
        impl<T: Scalar> Add<$rhs> for $lhs {
            type Output = $out;

            fn add(self, rhs: $rhs) -> Self::Output {
                Self::Output {
                    data: self.data + &rhs.data,
                    context: self.context,
                }
            }
        }
    };
}
pub(crate) use impl_add;

macro_rules! impl_sub {
    ($lhs:ty, $rhs:ty, $out:ty) => {
        impl<T: Scalar> Sub<$rhs> for $lhs {
            type Output = $out;

            fn sub(self, rhs: $rhs) -> Self::Output {
                Self::Output {
                    data: self.data - &rhs.data,
                    context: self.context,
                }
            }
        }
    };
}
pub(crate) use impl_sub;

macro_rules! impl_add_assign {
    ($lhs:ty, $rhs:ty) => {
        impl<T: Scalar> AddAssign<$rhs> for $lhs {
            fn add_assign(&mut self, rhs: $rhs) {
                self.data += &rhs.data;
            }
        }
    };
}
pub(crate) use impl_add_assign;

macro_rules! impl_sub_assign {
    ($lhs:ty, $rhs:ty) => {
        impl<T: Scalar> SubAssign<$rhs> for $lhs {
            fn sub_assign(&mut self, rhs: $rhs) {
                self.data -= &rhs.data;
            }
        }
    };
}
pub(crate) use impl_sub_assign;

macro_rules! impl_index {
    ($lhs:ty) => {
        impl<T: Scalar> Index<(IndexType, IndexType)> for $lhs {
            type Output = T;
            fn index(&self, index: (IndexType, IndexType)) -> &T {
                &self.data[index]
            }
        }
    };
}
pub(crate) use impl_index;

macro_rules! impl_index_mut {
    ($lhs:ty) => {
        impl<T: Scalar> IndexMut<(IndexType, IndexType)> for $lhs {
            fn index_mut(&mut self, index: (IndexType, IndexType)) -> &mut T {
                &mut self.data[index]
            }
        }
    };
}
pub(crate) use impl_index_mut;
