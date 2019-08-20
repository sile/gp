pub trait Mean<X> {
    fn call(&self, x: &X) -> f64;
}

pub const ZERO_MEAN: ConstantMean = ConstantMean::new(0.0);

#[derive(Debug, Clone, Copy)]
pub struct ConstantMean(f64);
impl ConstantMean {
    pub const fn new(v: f64) -> Self {
        ConstantMean(v)
    }
}
impl<X> Mean<X> for ConstantMean {
    fn call(&self, _x: &X) -> f64 {
        self.0
    }
}
