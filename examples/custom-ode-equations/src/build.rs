use diffsol::OdeBuilder;
use crate::{MyEquations, M};

pub fn build() {
    OdeBuilder::<M>::new()
            .p(vec![1.0, 10.0])
            .build_from_eqn(MyEquations::new())
            .unwrap();
}