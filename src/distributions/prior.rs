use crate::distributions::MultivariateNormal;
use crate::kernels::Kernel;
use crate::matrix::Matrix;
use crate::means::Mean;
use crate::vector::ColVec;
use crate::Result;
use rand::distributions::Distribution;
use rand::Rng;

#[derive(Debug)]
pub struct GaussianProcessPrior {
    inner: MultivariateNormal,
}
impl GaussianProcessPrior {
    pub fn new<X, M, K>(xs: &[X], mean: M, kernel: K) -> Result<Self>
    where
        M: Mean<X>,
        K: Kernel<X>,
    {
        let means = xs.iter().map(|x| mean.call(x)).collect::<ColVec>();
        let covariance = Matrix::cov(xs, kernel);
        let inner = track!(MultivariateNormal::new(means, covariance))?;
        Ok(Self { inner })
    }
}
impl Distribution<ColVec> for GaussianProcessPrior {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> ColVec {
        self.inner.sample(rng)
    }
}
