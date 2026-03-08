//! RusTorch: A comprehensive deep learning framework in Rust.
//!
//! This crate re-exports the core components:
//! - `rustorch_core` as `core`
//! - `rustorch_nn` as `nn`
//! - `rustorch_vision` as `vision` (feature "vision")
//! - `rustorch_text` as `text` (feature "text")
//! - `rustorch_wasm` as `wasm` (feature "wasm")

pub use rustorch_core as core;
pub use rustorch_nn as nn;

#[cfg(feature = "vision")]
pub use rustorch_vision as vision;

#[cfg(feature = "text")]
pub use rustorch_text as text;

#[cfg(feature = "wasm")]
pub use rustorch_wasm as wasm;

// Convenience re-exports
pub use core::Tensor;
pub use nn::Module;
