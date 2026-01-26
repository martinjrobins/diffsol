use burn::{
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
};

#[derive(Debug, Module)]
pub struct RhsNN<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    linear3: Linear<B>,
    activation: Relu,
}

impl<B: Backend> RhsNN<B> {
    pub fn forward(&self, input: Tensor<B, 1>) -> Tensor<B, 1> {
        let x = self.linear1.forward(input);
        let x = self.activation.forward(x);
        let x = self.linear2.forward(x);
        let x = self.activation.forward(x);
        let x = self.linear3.forward(x);
        x
    }
}

#[derive(Config, Debug)]
pub struct RhsNNConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
}

impl RhsNNConfig {
    pub fn init<B: Backend>(&self) -> RhsNN<B> {
        RhsNN {
            linear1: LinearConfig::new(self.input_dim, self.hidden_dim).init(&Default::default()),
            linear2: LinearConfig::new(self.hidden_dim, self.hidden_dim).init(&Default::default()),
            linear3: LinearConfig::new(self.hidden_dim, self.output_dim).init(&Default::default()),
            activation: Relu,
        }
    }
}
