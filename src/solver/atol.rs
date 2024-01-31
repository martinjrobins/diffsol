use crate::{callable::Callable, vector::Vector, Scalar};

use super::{Options, Problem};


pub struct Atol<V: Vector>{
    value: Option<V>,
}


impl<V: Vector> Default for Atol<V> {
    fn default() -> Self {
        Self {
            value: None,
        }
    }
}


impl <V: Vector> Atol<V> {
    pub fn new(value: V) -> Self {
        Self {
            value: Some(value),
        }
    }
    pub fn value<'a, P: Problem<V>, O: Options<V::T>>(&mut self, problem: &P, options: &O) -> &V {
        let nstates = problem.nstates();
        if problem.atol().is_some() {
            problem.atol().unwrap()
        } else {
            if self.value.is_none() {
                self.value = Some(V::from_element(nstates, options.atol()));
            } else if self.value.unwrap().len() != nstates {
                self.value = Some(V::from_element(nstates, options.atol()));
            }
            &self.value.unwrap()
        }
    }
}