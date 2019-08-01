use crate::Mean;

#[derive(Debug)]
pub struct ConstantMean(f64);
impl ConstantMean {
    pub const fn new(v: f64) -> Self {
        ConstantMean(v)
    }
}
impl<X> Mean<X> for ConstantMean {
    fn mean(&self, _x: &X) -> f64 {
        self.0
    }
}
