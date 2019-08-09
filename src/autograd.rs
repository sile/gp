use std::ops;

pub trait Expr: Sized {
    fn powi(self, n: i32) -> Powi<Self> {
        Powi { v: self, n }
    }
}

pub trait Grad {
    type Derivation;
    fn grad(self) -> Self::Derivation;
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct F64(f64);
impl F64 {
    pub const fn new(v: f64) -> Self {
        Self(v)
    }
}
impl Grad for F64 {
    type Derivation = Self;

    fn grad(self) -> Self::Derivation {
        Self::new(0.0)
    }
}
impl<T: Grad> ops::Add<T> for F64 {
    type Output = Add<F64, T>;

    fn add(self, rhs: T) -> Self::Output {
        Add { lhs: self, rhs }
    }
}
impl<T: Grad> ops::Sub<T> for F64 {
    type Output = Sub<F64, T>;

    fn sub(self, rhs: T) -> Self::Output {
        Sub { lhs: self, rhs }
    }
}
impl<T: Grad> ops::Mul<T> for F64 {
    type Output = Mul<F64, T>;

    fn mul(self, rhs: T) -> Self::Output {
        Mul { lhs: self, rhs }
    }
}

#[derive(Debug, Clone)]
pub struct Add<X, Y> {
    lhs: X,
    rhs: Y,
}
impl<X, Y> Grad for Add<X, Y>
where
    X: Grad,
    Y: Grad,
    X::Derivation: ops::Add<Y::Derivation>,
{
    type Derivation = <X::Derivation as ops::Add<Y::Derivation>>::Output;

    fn grad(self) -> Self::Derivation {
        self.lhs.grad() + self.rhs.grad()
    }
}

#[derive(Debug, Clone)]
pub struct Sub<X, Y> {
    lhs: X,
    rhs: Y,
}
impl<X, Y> Grad for Sub<X, Y>
where
    X: Grad,
    Y: Grad,
    X::Derivation: ops::Sub<Y::Derivation>,
{
    type Derivation = <X::Derivation as ops::Sub<Y::Derivation>>::Output;

    fn grad(self) -> Self::Derivation {
        self.lhs.grad() - self.rhs.grad()
    }
}

#[derive(Debug, Clone)]
pub struct Mul<X, Y> {
    lhs: X,
    rhs: Y,
}
impl<X, Y> Grad for Mul<X, Y>
where
    X: Grad,
    Y: Grad,
    X::Derivation: ops::Mul<Y::Derivation>,
{
    type Derivation = <X::Derivation as ops::Mul<Y::Derivation>>::Output;

    fn grad(self) -> Self::Derivation {
        self.lhs.grad() * self.rhs.grad()
    }
}

#[derive(Debug, Clone)]
pub struct Powi<T> {
    v: T,
    n: i32,
}
impl<T: Expr + Grad> Grad for Powi<T> {
    type Derivation = Either<F64, Mul<F64, Powi<T>>>;

    fn grad(self) -> Self::Derivation {
        assert!(self.n >= 1);
        if self.n == 1 {
            Either::A(F64::new(1.0))
        } else {
            Either::B(F64::new(self.n as f64) * self.v.powi(self.n - 1))
        }
    }
}

#[derive(Debug, Clone)]
pub enum Either<A, B> {
    A(A),
    B(B),
}
