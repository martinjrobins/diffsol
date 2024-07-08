use crate::Vector;

pub struct OdeSolution<V: Vector> {
    pub t: Vec<V::T>,
    pub y: Vec<V>,
}
