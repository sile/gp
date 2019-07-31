use crate::matrix::Matrix;
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
    pub fn new(means: Vector, covariance: Matrix) -> Option<Self> {
        let covariance_l = covariance.l()?;
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
        let x = self.means.clone().into_inner()
            + self.covariance_l.clone().into_inner() * z.into_inner();
        Vector::new(x)
    }
}

#[derive(Debug)]
pub struct Prior {
    inner: MultivariateNormal,
}
impl Prior {
    pub fn new<X, M, K>(xs: &[X], mean: &M, kernel: &K) -> Option<Self>
    where
        M: Mean<X>,
        K: Kernel<X>,
    {
        let means = xs.iter().map(|x| mean.mean(x)).collect::<Vector>();
        let covariance = Matrix::new_covariance(xs, kernel);
        let inner = MultivariateNormal::new(means, covariance)?;
        Some(Self { inner })
    }
}
impl Distribution<Vector> for Prior {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vector {
        self.inner.sample(rng)
    }
}
