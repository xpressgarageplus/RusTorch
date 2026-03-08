use crate::Module;
use rustorch_core::Tensor;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ReLU;

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl ReLU {
    pub fn new() -> Self {
        Self
    }
}

impl Module for ReLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.relu()
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}
