///! A set of macros to implement common operations for vectors
/// 
macro_rules! impl_vector_common {
    ($vec:ty, $con:ty) => {
        impl<T: Scalar> VectorCommon for $vec {
            type T = T;
            type C = $con;
        }
    };
}
pub(crate) use impl_vector_common;



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

macro_rules! impl_sub {
    ($lhs:ty, $rhs:ty, $out:ty) => {
        impl<T: Scalar> Sub<$rhs> for $lhs {
            type Output = $out;
            fn sub(self, rhs: $rhs) -> Self::Output {
                Self::Output { data: self.data - &rhs.data, context: self.context }
            }
        }
    };
}
pub(crate) use impl_sub;

macro_rules! impl_add {
    ($lhs:ty, $rhs:ty, $out:ty) => {
        impl<T: Scalar> Add<$rhs> for $lhs {
            type Output = $out;
            fn add(self, rhs: $rhs) -> Self::Output {
                Self::Output { data: self.data + &rhs.data, context: self.context }
            }
        }
    };
}
pub(crate) use impl_add;

macro_rules! impl_index {
    ($lhs:ty) => {
        impl<T: Scalar> Index<IndexType> for $lhs {
            type Output = T;
            fn index(&self, index: IndexType) -> &Self::Output {
                &self.data[index]
            }
        }
        impl<T: Scalar> IndexMut<IndexType> for $lhs {
            fn index_mut(&mut self, index: IndexType) -> &mut Self::Output {
                &mut self.data[index]
            }
        }
    };
}
pub(crate) use impl_index;

