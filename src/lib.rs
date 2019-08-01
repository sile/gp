/// Gaussian Process.
///
/// # References
///
/// - [A Tutorial on Bayesian Optimization]: https://arxiv.org/abs/1807.02811
pub mod distributions;
pub mod kernels;
pub mod matrix;
pub mod means;
pub mod vector;

pub trait Mean<X> {
    fn mean(&self, x: &X) -> f64;
}

pub trait Kernel<X> {
    fn kernel(&self, x0: &X, x1: &X) -> f64;
}
