use rustorch_core::Tensor;
use crate::Module;
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let size = in_features * out_features;
        // Simple uniform init (mock)
        let w_data = vec![0.01; size];
        
        let weight = Tensor::new(&w_data, &[out_features, in_features]).set_requires_grad(true);
        
        let b_data = vec![0.01; out_features];
        let bias = Tensor::new(&b_data, &[1, out_features]).set_requires_grad(true); 
        // Note: Bias shape set to [1, out_features] to match output [1, out_features] for batch=1
        
        Self {
            weight,
            bias: Some(bias),
        }
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        // y = x @ W.t() + b
        let w_t = self.weight.t();
        let output = input.matmul(&w_t);
        
        if let Some(bias) = &self.bias {
            // Check broadcasting
            // output: (N, Out)
            // bias: (1, Out)
            // Should be fine if we support broadcasting
            return output + bias.clone();
        }
        
        output
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(bias) = &self.bias {
            params.push(bias.clone());
        }
        params
    }
}
