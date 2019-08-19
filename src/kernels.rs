use crate::vector::Vector;
use crate::{DifferentiableKernel, Kernel};

/// Isotropic RBF kernel.
#[derive(Debug)]
pub struct RbfKernel {
    length_scale: f64,
}
impl RbfKernel {
    pub const fn new(length_scale: f64) -> Self {
        Self { length_scale }
    }
}
impl Kernel<Vector> for RbfKernel {
    type HyperParams = f64;

    fn kernel(&self, x0: &Vector, x1: &Vector) -> f64 {
        let a = x0
            .as_slice()
            .iter()
            .zip(x1.as_slice().iter())
            .map(|(x0, x1)| (x0 - x1).powi(2))
            .sum::<f64>();
        let b = 2.0 * self.length_scale.powi(2);
        (-a / b).exp()
    }
}
impl DifferentiableKernel for RbfKernel {
    fn partial<'a>(
        &self,
        _x0: &'a Vector,
        _x1: &'a Vector,
    ) -> Box<dyn 'a + Fn(Self::HyperParams) -> f64> {
        panic!()
    }
}
