use crate::{C, V};
use diffsol::Vector;
pub struct MyEquations {
    pub p: V,
}

impl MyEquations {
    pub fn new() -> Self {
        MyEquations {
            p: V::from_vec(vec![1.0, 10.0], C::default()),
        }
    }
}
