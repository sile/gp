use nalgebra;
use nalgebra::base::dimension::{Dynamic, U1};
use nalgebra::base::VecStorage;
use nalgebra::linalg::Cholesky;

pub mod distributions;

pub type Matrix = nalgebra::base::Matrix<f64, Dynamic, Dynamic, VecStorage<f64, Dynamic, Dynamic>>;
pub type Vector = nalgebra::base::Vector<f64, Dynamic, VecStorage<f64, Dynamic, U1>>;

pub trait Mean {
    fn mean(&self, x: &[f64]) -> f64;
}

pub trait Kernel {
    fn kernel(&self, x0: &[f64], x1: &[f64]) -> f64;
}

#[derive(Debug, Clone)]
pub struct Means {
    means: Vector,
}
impl Means {
    pub fn as_slice(&self) -> &[f64] {
        self.means.as_slice()
    }

    pub fn as_inner(&self) -> &Vector {
        &self.means
    }
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
