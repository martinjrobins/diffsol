use burn::{module::Module, prelude::Backend, Tensor};

pub trait Rhs<B: Backend>: Module<B> {
    fn forward(&self, params: Tensor<B, 1>, state: Tensor<B, 1>) -> Tensor<B, 1>;
}
