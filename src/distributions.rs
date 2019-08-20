use crate::matrix::Matrix;
use crate::vector::Vector;
use crate::{Kernel, Mean};
use rand::distributions::Distribution;
use rand::Rng;
use rand_distr;

pub use self::mvn::MultivariateNormal;
pub use self::posterior::GaussianProcessPosterior;
pub use self::prior::GaussianProcessPrior;

mod mvn;
mod posterior;
mod prior;

#[derive(Debug)]
pub struct Posterior {
    inner: rand_distr::Normal<f64>,
}
impl Posterior {
    pub fn new<X, F, M, K>(x: &X, xs: &[X], f: F, mean: M, kernel: K) -> Option<Self>
    where
        X: Clone,
        F: Fn(&X) -> f64,
        M: Mean<X>,
        K: Kernel<X>,
    {
        let cov0 = Matrix::covariance(&[x.clone()], xs, &kernel);
        let cov1_i = Matrix::covariance(xs, xs, &kernel).inverse()?;

        let m0 = mean.mean(x);
        let means0 = xs
            .iter()
            .map(|x| f(x) - mean.mean(x) + m0)
            .collect::<Vector>();
        let means =
            cov0.clone().into_inner() * (cov1_i.clone().into_inner() * means0.clone().into_inner());
        let mean = means.get((0, 0))?;

        let cov2 = kernel.kernel(x, x);
        let cov3 = Matrix::covariance(xs, &[x.clone()], &kernel);
        let cov4 = cov0.into_inner() * (cov1_i.into_inner() * cov3.into_inner());
        let variance = cov2 - cov4.get((0, 0))?;

        Some(Self {
            inner: rand_distr::Normal::new(*mean, variance.sqrt()).ok()?,
        })
    }
}
impl Distribution<f64> for Posterior {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.inner.sample(rng)
    }
}
