use crate::vector::Vector;

pub trait Kernel<X> {
    fn call(&self, x0: &X, x1: &X) -> f64;
}

#[derive(Debug, Clone)]
pub struct GaussianKernel {
    length_scale: f64,
}
impl GaussianKernel {
    pub fn new(length_scale: f64) -> Self {
        assert!(length_scale > 0.0);
        Self { length_scale }
    }
}
impl Kernel<f64> for GaussianKernel {
    fn call(&self, &x0: &f64, &x1: &f64) -> f64 {
        let a = (x0 - x1).powi(2);
        let b = self.length_scale.powi(2);
        (-a / b).exp()
    }
}
impl Kernel<Vector> for GaussianKernel {
    fn call(&self, x0: &Vector, x1: &Vector) -> f64 {
        let a = x0
            .iter()
            .zip(x1.iter())
            .map(|(x0, x1)| (x0 - x1).powi(2))
            .sum::<f64>();
        let b = self.length_scale.powi(2);
        (-a / b).exp()
    }
}
