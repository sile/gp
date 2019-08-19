//! Gaussian Process.
//!
//! # References
//!
//! - [A Tutorial on Bayesian Optimization]: https://arxiv.org/abs/1807.02811
#[macro_use]
extern crate trackable;

use self::vector::Vector;

pub use self::error::{Error, ErrorKind};

pub mod autograd;
pub mod distributions;
pub mod kernels;
pub mod matrix;
pub mod means;
pub mod vector;

mod error;

pub type Result<T> = std::result::Result<T, Error>;

pub trait Mean<X> {
    fn mean(&self, x: &X) -> f64;
}

pub trait Kernel<X = Vector> {
    type HyperParams;

    fn kernel(&self, x0: &X, x1: &X) -> f64;
}

pub trait DifferentiableKernel: Kernel {
    fn partial<'a>(
        &self,
        x0: &'a Vector,
        x1: &'a Vector,
    ) -> Box<dyn 'a + Fn(Self::HyperParams) -> f64>;
}
