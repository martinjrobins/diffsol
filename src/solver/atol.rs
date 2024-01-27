use crate::{callable::Callable, vector::Vector, Scalar};

use super::{Options, Problem, SolverOptions, SolverProblem};


pub struct Atol<T: Scalar, V: Vector<T>>{
    value: Option<V>,
    _phantom: std::marker::PhantomData<T>,
}


impl<T: Scalar, V: Vector<T>> Default for Atol<T, V> {
    fn default() -> Self {
        Self {
            value: None,
            _phantom: std::marker::PhantomData,
        }
    }
}


impl <V: Vector<T>, T: Scalar> Atol<T, V> {
    pub fn new(value: V) -> Self {
        Self {
            value: Some(value),
            _phantom: std::marker::PhantomData,
        }
    }
    pub fn value<'a, P: Problem<T, V>, O: Options<T>>(&mut self, problem: &P, options: &O) -> &V {
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