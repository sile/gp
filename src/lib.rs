pub mod distributions;
pub mod matrix;
pub mod vector;

pub trait Mean<X> {
    fn mean(&self, x: &X) -> f64;
}

pub trait Kernel<X> {
    fn kernel(&self, x0: &X, x1: &X) -> f64;
}
