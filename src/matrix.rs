use crate::Kernel;
use nalgebra::base::dimension::{Dim as _, Dynamic};
use nalgebra::base::{Matrix as InnerMatrix, VecStorage};
use nalgebra::linalg::Cholesky;

#[derive(Debug, Clone)]
pub struct Matrix {
    inner: InnerMatrix<f64, Dynamic, Dynamic, VecStorage<f64, Dynamic, Dynamic>>,
}
impl Matrix {
    pub fn new(
        inner: InnerMatrix<f64, Dynamic, Dynamic, VecStorage<f64, Dynamic, Dynamic>>,
    ) -> Self {
        Self { inner }
    }

    pub fn new_covariance<X, K>(xs: &[X], kernel: &K) -> Self
    where
        K: Kernel<X>,
    {
        let mut covariance = Vec::new();
        for x0 in xs {
            for x1 in xs {
                // FIXME: calculate only lower triangle.
                covariance.push(kernel.kernel(x0, x1));
            }
        }
        Self::from_vec(xs.len(), xs.len(), covariance)
    }

    pub fn from_vec(rows: usize, cols: usize, vec: Vec<f64>) -> Self {
        let data = VecStorage::new(Dynamic::from_usize(rows), Dynamic::from_usize(cols), vec);
        Self::new(InnerMatrix::from_data(data))
    }

    pub fn cholesky(self) -> Option<Cholesky<f64, Dynamic>> {
        Cholesky::new(self.inner)
    }

    pub fn l(self) -> Option<Self> {
        self.cholesky()
            .map(|decomposed| decomposed.l())
            .map(Self::new)
    }

    pub fn into_inner(
        self,
    ) -> InnerMatrix<f64, Dynamic, Dynamic, VecStorage<f64, Dynamic, Dynamic>> {
        self.inner
    }
}
