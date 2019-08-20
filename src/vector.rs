use nalgebra::base::dimension::{Dim as _, Dynamic, U1};
use nalgebra::base::{VecStorage, Vector as InnerVector};
use std::iter::FromIterator;
use std::ops;

/// Column vector.
pub type ColVec = Vector;

#[derive(Debug, Clone)]
pub struct Vector {
    inner: InnerVector<f64, Dynamic, VecStorage<f64, Dynamic, U1>>,
}
impl Vector {
    pub fn new(inner: InnerVector<f64, Dynamic, VecStorage<f64, Dynamic, U1>>) -> Self {
        Self { inner }
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn iter<'a>(&'a self) -> impl 'a + Iterator<Item = f64> {
        self.inner.iter().copied()
    }

    pub fn as_slice(&self) -> &[f64] {
        self.inner.as_slice()
    }

    pub fn into_inner(self) -> InnerVector<f64, Dynamic, VecStorage<f64, Dynamic, U1>> {
        self.inner
    }
}
impl From<Vec<f64>> for Vector {
    fn from(f: Vec<f64>) -> Self {
        let rows = Dynamic::from_usize(f.len());
        let inner = InnerVector::from_data(VecStorage::new(rows, U1, f));
        Self { inner }
    }
}
impl FromIterator<f64> for Vector {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = f64>,
    {
        Self::from(iter.into_iter().collect::<Vec<_>>())
    }
}
impl ops::Add<ColVec> for ColVec {
    type Output = ColVec;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.into_inner() + rhs.into_inner())
    }
}
