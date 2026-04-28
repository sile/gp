use crate::matrix::Matrix;
use crate::vector::ColVec;
use crate::{Error, Result};
use rand::Rng;
use rand::distr::Distribution;
use rand_distr::StandardNormal;

/// https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Drawing_values_from_the_distribution
#[derive(Debug)]
pub struct MultivariateNormal {
    means: ColVec,
    covariance_l: Matrix,
}
impl MultivariateNormal {
    pub fn new(means: ColVec, covariance: Matrix) -> Result<Self> {
        let covariance_l = covariance.l().ok_or_else(|| {
            Error::InvalidInput("covariance matrix is not positive definite".into())
        })?;
        Ok(Self {
            means,
            covariance_l,
        })
    }
}
impl Distribution<ColVec> for MultivariateNormal {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> ColVec {
        let n = self.means.len();
        let z = StandardNormal.sample_iter(rng).take(n).collect::<ColVec>();
        self.means.clone() + self.covariance_l.clone() * z
    }
}
