use burn::tensor::ops::FloatTensor;
use diffsol::{matrix::MatrixRef, DefaultDenseMatrix, VectorRef};

use crate::backend::Solve;

pub trait Vector: diffsol::Vector + diffsol::DefaultDenseMatrix {}

impl<T> Vector for T where T: diffsol::Vector + diffsol::DefaultDenseMatrix {}
