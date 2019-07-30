use nalgebra::base::dimension::{Dim as _, Dynamic, U1};
use nalgebra::base::{VecStorage, Vector as InnerVector};

#[derive(Debug, Clone)]
pub struct Vector {
    inner: InnerVector<f64, Dynamic, VecStorage<f64, Dynamic, U1>>,
}
impl Vector {
    pub fn new(inner: InnerVector<f64, Dynamic, VecStorage<f64, Dynamic, U1>>) -> Self {
        Self { inner }
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
