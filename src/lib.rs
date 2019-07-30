use nalgebra;
use nalgebra::base::dimension::Dynamic;
use nalgebra::base::VecStorage;
use nalgebra::linalg::Cholesky;

pub mod distributions;

pub type Matrix = nalgebra::base::Matrix<f64, Dynamic, Dynamic, VecStorage<f64, Dynamic, Dynamic>>;

pub trait Mean {
    fn mean(&self, x: &[f64]) -> f64;
}

pub trait Kernel {
    fn kernel(&self, x0: &[f64], x1: &[f64]) -> f64;
}

#[derive(Debug)]
pub struct Means {
    means: Vec<f64>,
}

// NOTE: Must be a positive semi-definite matrix
#[derive(Debug)]
pub struct Covariance {
    matrix: Matrix,
}
impl Covariance {
    pub fn cholesky(&self) -> Option<Cholesky<f64, Dynamic>> {
        Cholesky::new(self.matrix.clone())
    }
}
