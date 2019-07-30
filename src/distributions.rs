use super::{Covariance, Means};
use rand::distributions::Distribution;

/// https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Drawing_values_from_the_distribution
#[derive(Debug)]
pub struct Normal {}
impl Normal {
    pub fn new(means: Means, covariance: Covariance) -> Self {
        panic!()
    }
}
