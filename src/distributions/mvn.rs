use crate::matrix::Matrix;
use crate::vector::ColVec;
use crate::{ErrorKind, Result};
use rand::distributions::Distribution;
use rand::Rng;
use rand_distr::StandardNormal;

/// https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Drawing_values_from_the_distribution
#[derive(Debug)]
pub struct MultivariateNormal {
    means: ColVec,
    covariance_l: Matrix,
}
impl MultivariateNormal {
    pub fn new(means: ColVec, covariance: Matrix) -> Result<Self> {
        let covariance_l = track_assert_some!(covariance.l(), ErrorKind::InvalidInput);
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
