use crate::{C, V};
use diffsol::Vector;
pub struct MyEquations {
    pub p: V,
}

impl MyEquations {
    pub fn new() -> Self {
        MyEquations {
            p: V::zeros(2, C::default()),
        }
    }
}
