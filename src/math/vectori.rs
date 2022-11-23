pub use crate::math::interval::Interval;
pub use crate::math::scalar::{Float, GFloat, Int};
pub use crate::math::vector::{GenericVector, Point3, Vector3};
pub use crate::math::vector_traits::Norm;

pub type Vector3fi = Vector3<Interval<Float>>;
pub type Vector3ii = Vector3<Interval<Int>>;
pub type Point3fi = Point3<Interval<Float>>;
pub type Point3ii = Point3<Interval<Int>>;

impl<T: GFloat, const N: usize, U> std::ops::Mul<T> for GenericVector<Interval<T>, N, U> {
    type Output = GenericVector<Interval<T>, N, U>;

    fn mul(self, rhs: T) -> Self::Output {
        Self::from(self.vec.map(|x| x * rhs))
    }
}

impl<T: GFloat, const N: usize, U> std::ops::Div<T> for GenericVector<Interval<T>, N, U> {
    type Output = GenericVector<Interval<T>, N, U>;

    fn div(self, rhs: T) -> Self::Output {
        Self::from(self.vec.map(|x| x / rhs))
    }
}

impl<T: GFloat, const N: usize, U> From<GenericVector<T, N, U>>
    for GenericVector<Interval<T>, N, U>
{
    fn from(rhs: GenericVector<T, N, U>) -> GenericVector<Interval<T>, N, U> {
        Self::from(rhs.vec.map(|x| Interval::from(x)))
    }
}
impl<T: GFloat, const N: usize, U> From<GenericVector<Interval<T>, N, U>>
    for GenericVector<T, N, U>
{
    fn from(rhs: GenericVector<Interval<T>, N, U>) -> GenericVector<T, N, U> {
        Self::from(rhs.vec.map(|x| x.approx()))
    }
}

// this is more accurate than self.dot(self) because of interval arithmetic
impl Norm for Vector3fi {
    fn square_norm(self) -> Self::ScalarType {
        self.x().square() + self.y().square() + self.z().square()
    }
}
