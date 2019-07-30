use super::{Covariance, Matrix, Means, Vector};
use nalgebra::base::dimension::{Dim, Dynamic, U1};
use nalgebra::base::VecStorage;
use rand::distributions::Distribution;
use rand::Rng;
use rand_distr::StandardNormal;

/// https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Drawing_values_from_the_distribution
#[derive(Debug)]
pub struct Normal {
    means: Means,
    covariance_l: Matrix,
}
impl Normal {
    pub fn new(means: Means, covariance: Covariance) -> Option<Self> {
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
        let z = Vector::from_data(VecStorage::new(
            Dynamic::from_usize(n),
            U1,
            StandardNormal.sample_iter(rng).take(n).collect::<Vec<_>>(),
        ));
        self.means.as_inner().clone() + self.covariance_l.clone() * z
    }
}

// #[derive(Debug)]
// pub struct Prior{

// }
