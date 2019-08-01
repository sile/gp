use crate::vector::Vector;
use crate::Kernel;

/// a.k.a., power exponential kernel
#[derive(Debug)]
pub struct GaussianKernel {
    alphas: Vector,
}
impl GaussianKernel {
    pub fn new(alphas: Vector) -> Self {
        Self { alphas }
    }

    fn distance(&self, x0: &Vector, x1: &Vector) -> f64 {
        self.alphas
            .as_slice()
            .iter()
            .skip(1)
            .zip(x0.as_slice().iter())
            .zip(x1.as_slice().iter())
            .map(|((a, x0), x1)| a * (x0 - x1).powi(2))
            .sum()
    }
}
impl Kernel<Vector> for GaussianKernel {
    fn kernel(&self, x0: &Vector, x1: &Vector) -> f64 {
        assert_eq!(self.alphas.len(), x0.len() + 1);
        assert_eq!(x0.len(), x1.len());

        self.alphas.as_slice()[0] * (-self.distance(x0, x1)).exp()
    }
}

// TODO
//
// #[derive(Debug)]
// pub struct MaternKernel {}
