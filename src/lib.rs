use nalgebra;
use nalgebra::base::dimension::{Dynamic, U1};
use nalgebra::base::VecStorage;
use nalgebra::linalg::Cholesky;

pub mod distributions;
pub mod vector;

pub type Matrix = nalgebra::base::Matrix<f64, Dynamic, Dynamic, VecStorage<f64, Dynamic, Dynamic>>;
pub type Vector = nalgebra::base::Vector<f64, Dynamic, VecStorage<f64, Dynamic, U1>>;

pub trait Mean<X> {
    fn mean(&self, x: &X) -> f64;
}

pub trait Kernel<X> {
    fn kernel(&self, x0: &X, x1: &X) -> f64;
}

// NOTE: Must be a positive semi-definite matrix
#[derive(Debug, Clone)]
pub struct Covariance {
    matrix: Matrix,
}
impl Covariance {
    pub fn cholesky(self) -> Option<Cholesky<f64, Dynamic>> {
        Cholesky::new(self.matrix)
    }
}
