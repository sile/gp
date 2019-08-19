use crate::matrix::Matrix;
use crate::vector::Vector;
use crate::{ErrorKind, Result};
use rand::distributions::Distribution;
use rand::Rng;
use rand_distr::StandardNormal;

/// https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Drawing_values_from_the_distribution
#[derive(Debug)]
pub struct MultivariateNormal {
    means: Vector,
    covariance_l: Matrix,
}
impl MultivariateNormal {
    pub fn new(means: Vector, covariance: Matrix) -> Result<Self> {
        let covariance_l = track_assert_some!(covariance.l(), ErrorKind::InvalidInput);
        Ok(Self {
            means,
            covariance_l,
        })
    }
}
impl Distribution<Vector> for MultivariateNormal {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vector {
        let n = self.means.len();
        let z = StandardNormal.sample_iter(rng).take(n).collect::<Vector>();
        let x = self.means.clone().into_inner()
            + self.covariance_l.clone().into_inner() * z.into_inner();
        Vector::new(x)
    }
}
