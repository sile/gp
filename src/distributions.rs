use super::{Covariance, Matrix};
use crate::vector::Vector;
use crate::{Kernel, Mean};
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
    pub fn new(means: Vector, covariance: Covariance) -> Option<Self> {
        let covariance_l = covariance.cholesky()?.l();
        Some(Self {
            means,
            covariance_l,
        })
    }
}
impl Distribution<Vector> for MultivariateNormal {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vector {
        let n = self.means.as_slice().len();
        let z = StandardNormal.sample_iter(rng).take(n).collect::<Vector>();
        let x = self.means.clone().into_inner() + self.covariance_l.clone() * z.into_inner();
        Vector::new(x)
    }
}

#[derive(Debug)]
pub struct Prior {}
impl Prior {
    pub fn new<X, M, K>(xs: &[X], mean: &M, kernel: &K) -> Self
    where
        M: Mean<X>,
        K: Kernel<X>,
    {
        let means = xs.iter().map(|x| mean.mean(x)).collect::<Vector>();
        for x0 in xs {
            for x1 in xs {
                kernel.kernel(x0, x1);
            }
        }
        panic!()
    }
}
