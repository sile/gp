use super::{Covariance, Matrix};
use crate::vector::Vector;
use rand::distributions::Distribution;
use rand::Rng;
use rand_distr::StandardNormal;

/// https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Drawing_values_from_the_distribution
#[derive(Debug)]
pub struct Normal {
    means: Vector,
    covariance_l: Matrix,
}
impl Normal {
    pub fn new(means: Vector, covariance: Covariance) -> Option<Self> {
        let covariance_l = covariance.cholesky()?.l();
        Some(Self {
            means,
            covariance_l,
        })
    }
}
impl Distribution<Vector> for Normal {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vector {
        let n = self.means.as_slice().len();
        let z = Vector::from(StandardNormal.sample_iter(rng).take(n).collect::<Vec<_>>());
        let x = self.means.clone().into_inner() + self.covariance_l.clone() * z.into_inner();
        Vector::new(x)
    }
}

// #[derive(Debug)]
// pub struct Prior{

// }
