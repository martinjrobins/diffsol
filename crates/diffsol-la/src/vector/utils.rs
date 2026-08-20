macro_rules! impl_vector_common {
    ($vec:ty, $con:ty, $in:ty) => {
        impl<T: Scalar> VectorCommon for $vec {
            type T = T;
            type C = $con;
            type Inner = $in;
            fn inner(&self) -> &Self::Inner {
                &self.data
            }
        }
    };
    ($vec:ty, $con:ty, $in:ty, $bound:path) => {
        impl<T: Scalar + $bound> VectorCommon for $vec {
            type T = T;
            type C = $con;
            type Inner = $in;
            fn inner(&self) -> &Self::Inner {
                &self.data
            }
        }
    };
}
pub(crate) use impl_vector_common;

macro_rules! impl_vector_common_ref {
    ($vec:ty, $con:ty, $in:ty) => {
        impl<'a, T: Scalar> VectorCommon for $vec {
            type T = T;
            type C = $con;
            type Inner = $in;
            fn inner(&self) -> &Self::Inner {
                &self.data
            }
        }
    };
    ($vec:ty, $con:ty, $in:ty, $bound:path) => {
        impl<'a, T: Scalar + $bound> VectorCommon for $vec {
            type T = T;
            type C = $con;
            type Inner = $in;
            fn inner(&self) -> &Self::Inner {
                &self.data
            }
        }
    };
}
pub(crate) use impl_vector_common_ref;

macro_rules! impl_sub_assign {
    ($lhs:ty, $rhs:ty) => {
        impl<T: Scalar> SubAssign<$rhs> for $lhs {
            fn sub_assign(&mut self, rhs: $rhs) {
                self.context
                    .assert_compatible_nbatch(rhs.context.nbatch(), "sub_assign");
                for (batch, data) in self.data.iter_mut().enumerate() {
                    *data -= &rhs.data[batch % rhs.data.len()];
                }
            }
        }
    };
    ($lhs:ty, $rhs:ty, $bound:path) => {
        impl<T: Scalar + $bound> SubAssign<$rhs> for $lhs {
            fn sub_assign(&mut self, rhs: $rhs) {
                self.context
                    .assert_compatible_nbatch(rhs.context.nbatch(), "sub_assign");
                for (batch, data) in self.data.iter_mut().enumerate() {
                    *data -= &rhs.data[batch % rhs.data.len()];
                }
            }
        }
    };
}
pub(crate) use impl_sub_assign;

macro_rules! impl_add_assign {
    ($lhs:ty, $rhs:ty) => {
        impl<T: Scalar> AddAssign<$rhs> for $lhs {
            fn add_assign(&mut self, rhs: $rhs) {
                self.context
                    .assert_compatible_nbatch(rhs.context.nbatch(), "add_assign");
                for (batch, data) in self.data.iter_mut().enumerate() {
                    *data += &rhs.data[batch % rhs.data.len()];
                }
            }
        }
    };
    ($lhs:ty, $rhs:ty, $bound:path) => {
        impl<T: Scalar + $bound> AddAssign<$rhs> for $lhs {
            fn add_assign(&mut self, rhs: $rhs) {
                self.context
                    .assert_compatible_nbatch(rhs.context.nbatch(), "add_assign");
                for (batch, data) in self.data.iter_mut().enumerate() {
                    *data += &rhs.data[batch % rhs.data.len()];
                }
            }
        }
    };
}
pub(crate) use impl_add_assign;

macro_rules! impl_sub_lhs {
    ($lhs:ty, $rhs:ty, $out:ty) => {
        impl<T: Scalar> Sub<$rhs> for $lhs {
            type Output = $out;
            fn sub(self, rhs: $rhs) -> Self::Output {
                self.context
                    .assert_compatible_nbatch(rhs.context.nbatch(), "sub");
                Self::Output {
                    data: (0..self.data.len())
                        .map(|batch| &self.data[batch] - &rhs.data[batch % rhs.data.len()])
                        .collect(),
                    context: self.context,
                }
            }
        }
    };
    ($lhs:ty, $rhs:ty, $out:ty, $bound:path) => {
        impl<T: Scalar + $bound> Sub<$rhs> for $lhs {
            type Output = $out;
            fn sub(self, rhs: $rhs) -> Self::Output {
                self.context
                    .assert_compatible_nbatch(rhs.context.nbatch(), "sub");
                Self::Output {
                    data: (0..self.data.len())
                        .map(|batch| &self.data[batch] - &rhs.data[batch % rhs.data.len()])
                        .collect(),
                    context: self.context,
                }
            }
        }
    };
}
pub(crate) use impl_sub_lhs;

macro_rules! impl_sub_rhs {
    ($lhs:ty, $rhs:ty, $out:ty) => {
        impl<T: Scalar> Sub<$rhs> for $lhs {
            type Output = $out;
            fn sub(self, rhs: $rhs) -> Self::Output {
                rhs.context
                    .assert_compatible_nbatch(self.context.nbatch(), "sub");
                Self::Output {
                    data: (0..rhs.data.len())
                        .map(|batch| &self.data[batch % self.data.len()] - &rhs.data[batch])
                        .collect(),
                    context: rhs.context,
                }
            }
        }
    };
    ($lhs:ty, $rhs:ty, $out:ty, $bound:path) => {
        impl<T: Scalar + $bound> Sub<$rhs> for $lhs {
            type Output = $out;
            fn sub(self, rhs: $rhs) -> Self::Output {
                rhs.context
                    .assert_compatible_nbatch(self.context.nbatch(), "sub");
                Self::Output {
                    data: (0..rhs.data.len())
                        .map(|batch| &self.data[batch % self.data.len()] - &rhs.data[batch])
                        .collect(),
                    context: rhs.context,
                }
            }
        }
    };
}
pub(crate) use impl_sub_rhs;

macro_rules! impl_sub_both_ref {
    ($lhs:ty, $rhs:ty, $out:ty) => {
        impl<T: Scalar> Sub<$rhs> for $lhs {
            type Output = $out;
            fn sub(self, rhs: $rhs) -> Self::Output {
                self.context
                    .assert_compatible_nbatch(rhs.context.nbatch(), "sub");
                Self::Output {
                    data: (0..self.data.len())
                        .map(|batch| &self.data[batch] - &rhs.data[batch % rhs.data.len()])
                        .collect(),
                    context: self.context.clone(),
                }
            }
        }
    };
    ($lhs:ty, $rhs:ty, $out:ty, $bound:path) => {
        impl<T: Scalar + $bound> Sub<$rhs> for $lhs {
            type Output = $out;
            fn sub(self, rhs: $rhs) -> Self::Output {
                self.context
                    .assert_compatible_nbatch(rhs.context.nbatch(), "sub");
                Self::Output {
                    data: (0..self.data.len())
                        .map(|batch| &self.data[batch] - &rhs.data[batch % rhs.data.len()])
                        .collect(),
                    context: self.context.clone(),
                }
            }
        }
    };
}
pub(crate) use impl_sub_both_ref;

macro_rules! impl_add_lhs {
    ($lhs:ty, $rhs:ty, $out:ty) => {
        impl<T: Scalar> Add<$rhs> for $lhs {
            type Output = $out;
            fn add(self, rhs: $rhs) -> Self::Output {
                self.context
                    .assert_compatible_nbatch(rhs.context.nbatch(), "add");
                Self::Output {
                    data: (0..self.data.len())
                        .map(|batch| &self.data[batch] + &rhs.data[batch % rhs.data.len()])
                        .collect(),
                    context: self.context,
                }
            }
        }
    };
    ($lhs:ty, $rhs:ty, $out:ty, $bound:path) => {
        impl<T: Scalar + $bound> Add<$rhs> for $lhs {
            type Output = $out;
            fn add(self, rhs: $rhs) -> Self::Output {
                self.context
                    .assert_compatible_nbatch(rhs.context.nbatch(), "add");
                Self::Output {
                    data: (0..self.data.len())
                        .map(|batch| &self.data[batch] + &rhs.data[batch % rhs.data.len()])
                        .collect(),
                    context: self.context,
                }
            }
        }
    };
}
pub(crate) use impl_add_lhs;

macro_rules! impl_add_rhs {
    ($lhs:ty, $rhs:ty, $out:ty) => {
        impl<T: Scalar> Add<$rhs> for $lhs {
            type Output = $out;
            fn add(self, rhs: $rhs) -> Self::Output {
                rhs.context
                    .assert_compatible_nbatch(self.context.nbatch(), "add");
                Self::Output {
                    data: (0..rhs.data.len())
                        .map(|batch| &self.data[batch % self.data.len()] + &rhs.data[batch])
                        .collect(),
                    context: rhs.context,
                }
            }
        }
    };
    ($lhs:ty, $rhs:ty, $out:ty, $bound:path) => {
        impl<T: Scalar + $bound> Add<$rhs> for $lhs {
            type Output = $out;
            fn add(self, rhs: $rhs) -> Self::Output {
                rhs.context
                    .assert_compatible_nbatch(self.context.nbatch(), "add");
                Self::Output {
                    data: (0..rhs.data.len())
                        .map(|batch| &self.data[batch % self.data.len()] + &rhs.data[batch])
                        .collect(),
                    context: rhs.context,
                }
            }
        }
    };
}
pub(crate) use impl_add_rhs;

macro_rules! impl_add_both_ref {
    ($lhs:ty, $rhs:ty, $out:ty) => {
        impl<T: Scalar> Add<$rhs> for $lhs {
            type Output = $out;
            fn add(self, rhs: $rhs) -> Self::Output {
                self.context
                    .assert_compatible_nbatch(rhs.context.nbatch(), "add");
                Self::Output {
                    data: (0..self.data.len())
                        .map(|batch| &self.data[batch] + &rhs.data[batch % rhs.data.len()])
                        .collect(),
                    context: self.context.clone(),
                }
            }
        }
    };
    ($lhs:ty, $rhs:ty, $out:ty, $bound:path) => {
        impl<T: Scalar + $bound> Add<$rhs> for $lhs {
            type Output = $out;
            fn add(self, rhs: $rhs) -> Self::Output {
                self.context
                    .assert_compatible_nbatch(rhs.context.nbatch(), "add");
                Self::Output {
                    data: (0..self.data.len())
                        .map(|batch| &self.data[batch] + &rhs.data[batch % rhs.data.len()])
                        .collect(),
                    context: self.context.clone(),
                }
            }
        }
    };
}
pub(crate) use impl_add_both_ref;

macro_rules! impl_index {
    ($lhs:ty) => {
        impl<T: Scalar> Index<IndexType> for $lhs {
            type Output = T;
            fn index(&self, index: IndexType) -> &Self::Output {
                &self.data[0][index]
            }
        }
    };
    ($lhs:ty, $bound:path) => {
        impl<T: Scalar + $bound> Index<IndexType> for $lhs {
            type Output = T;
            fn index(&self, index: IndexType) -> &Self::Output {
                &self.data[0][index]
            }
        }
    };
}
pub(crate) use impl_index;

macro_rules! impl_index_mut {
    ($lhs:ty) => {
        impl<T: Scalar> IndexMut<IndexType> for $lhs {
            fn index_mut(&mut self, index: IndexType) -> &mut Self::Output {
                &mut self.data[0][index]
            }
        }
    };
    ($lhs:ty, $bound:path) => {
        impl<T: Scalar + $bound> IndexMut<IndexType> for $lhs {
            fn index_mut(&mut self, index: IndexType) -> &mut Self::Output {
                &mut self.data[0][index]
            }
        }
    };
}
pub(crate) use impl_index_mut;
