use crate::scalar::{Float, Int, Scalar, SignedScalar, UInt};
pub use crate::vector_traits::{Dimension, InnerProduct, InnerScalar, Norm};

pub type Vector<T, const N: usize> = GenericVector<T, N, VectorMarker>;
pub type Vector4<T> = Vector<T, 4>;
pub type Vector3<T> = Vector<T, 3>;
pub type Vector2<T> = Vector<T, 2>;

pub type Point<T, const N: usize> = GenericVector<T, N, PointMarker>;
pub type Point3<T> = Point<T, 3>;
pub type Point2<T> = Point<T, 2>;

pub type Normal<T, const N: usize> = GenericVector<T, N, NormalMarker>;
pub type Normal3<T> = Normal<T, 3>;

pub type Vector4f = Vector4<Float>;
pub type Vector4i = Vector4<Int>;
pub type Vector4u = Vector4<UInt>;

pub type Vector3f = Vector3<Float>;
pub type Vector3i = Vector3<Int>;
pub type Vector3u = Vector3<UInt>;

pub type Vector2f = Vector2<Float>;
pub type Vector2i = Vector2<Int>;
pub type Vector2u = Vector2<UInt>;

pub type Point3f = Point3<Float>;
pub type Point3i = Point3<Int>;
pub type Point3u = Point3<UInt>;

pub type Point2f = Point2<Float>;
pub type Point2i = Point2<Int>;
pub type Point2u = Point2<UInt>;

pub type Normal3f = Normal3<Float>;
pub type Normal3i = Normal3<Int>;

#[derive(Debug, Copy, Clone)]
pub struct VectorMarker {}
#[derive(Debug, Copy, Clone)]
pub struct PointMarker {}
#[derive(Debug, Copy, Clone)]
pub struct NormalMarker {}

pub trait HasSub {}
impl HasSub for VectorMarker {}

#[derive(Debug)]
pub struct GenericVector<T: Scalar, const N: usize, U> {
    pub vec: [T; N],
    p: std::marker::PhantomData<U>,
}

impl<T: Scalar, U> GenericVector<T, 4, U> {
    pub fn new(x: T, y: T, z: T, w: T) -> Self {
        let p = std::marker::PhantomData;
        Self {
            vec: [x, y, z, w],
            p,
        }
    }
}
impl<T: Scalar, U> GenericVector<T, 3, U> {
    pub fn new(x: T, y: T, z: T) -> Self {
        let p = std::marker::PhantomData;
        Self { vec: [x, y, z], p }
    }

    pub fn cross(self, vec: Self) -> Self {
        Self::new(
            self.y() * vec.z() - self.z() * vec.y(),
            self.z() * vec.x() - self.x() * vec.z(),
            self.x() * vec.y() - self.y() * vec.x(),
        )
    }
}
impl<T: Scalar, U> GenericVector<T, 2, U> {
    pub fn new(x: T, y: T) -> Self {
        let p = std::marker::PhantomData;
        Self { vec: [x, y], p }
    }
}

impl<T: Scalar, const N: usize, U> GenericVector<T, N, U> {
    pub fn x(self) -> T {
        assert!(N > 0);
        self.vec[0]
    }
    pub fn y(self) -> T {
        assert!(N > 1);
        self.vec[1]
    }
    pub fn z(self) -> T {
        assert!(N > 2);
        self.vec[2]
    }
    pub fn w(self) -> T {
        assert!(N > 3);
        self.vec[3]
    }
    pub fn x_mut(&mut self) -> &mut T {
        assert!(N > 0);
        &mut self.vec[0]
    }
    pub fn y_mut(&mut self) -> &mut T {
        assert!(N > 1);
        &mut self.vec[1]
    }
    pub fn z_mut(&mut self) -> &mut T {
        assert!(N > 2);
        &mut self.vec[2]
    }
    pub fn w_mut(&mut self) -> &mut T {
        assert!(N > 3);
        &mut self.vec[3]
    }
}

impl<T: Scalar, const N: usize, U> InnerScalar for GenericVector<T, N, U> {
    type ScalarType = T;
}
impl<T: Scalar, const N: usize, U> Dimension for GenericVector<T, N, U> {
    const DIM: usize = N;
}

impl<T: Scalar, const N: usize, U> std::cmp::PartialEq<Self> for GenericVector<T, N, U> {
    fn eq(&self, other: &Self) -> bool {
        self.vec == other.vec
    }
}

impl<T: SignedScalar, const N: usize, U> std::ops::Neg for GenericVector<T, N, U> {
    type Output = GenericVector<T, N, U>;

    fn neg(self) -> Self::Output {
        let vec: [T; N] = self.vec.map(|v| -v);
        let p = std::marker::PhantomData;
        Self { vec, p }
    }
}

impl<T: Scalar, const N: usize, U> std::ops::Add<GenericVector<T, N, U>>
    for GenericVector<T, N, U>
{
    type Output = GenericVector<T, N, U>;

    fn add(self, rhs: GenericVector<T, N, U>) -> Self::Output {
        let mut vec: [T; N] = self.vec;
        for i in 0..N {
            vec[i] += rhs.vec[i];
        }
        Self {
            vec,
            p: std::marker::PhantomData,
        }
    }
}

impl<T: Scalar, const N: usize, U> std::ops::AddAssign<GenericVector<T, N, U>>
    for GenericVector<T, N, U>
{
    fn add_assign(&mut self, rhs: GenericVector<T, N, U>) {
        for i in 0..N {
            self.vec[i] += rhs.vec[i];
        }
    }
}

impl<T: Scalar, const N: usize, U: HasSub> std::ops::Sub<GenericVector<T, N, U>>
    for GenericVector<T, N, U>
{
    type Output = GenericVector<T, N, U>;

    fn sub(self, rhs: GenericVector<T, N, U>) -> Self::Output {
        let mut vec: [T; N] = self.vec;
        for i in 0..N {
            vec[i] -= rhs.vec[i];
        }
        Self {
            vec,
            p: std::marker::PhantomData,
        }
    }
}

impl<T: Scalar, const N: usize, U: HasSub> std::ops::SubAssign<GenericVector<T, N, U>>
    for GenericVector<T, N, U>
{
    fn sub_assign(&mut self, rhs: GenericVector<T, N, U>) {
        for i in 0..N {
            self.vec[i] -= rhs.vec[i];
        }
    }
}

impl<T: Scalar, const N: usize, U> Copy for GenericVector<T, N, U> {}
impl<T: Scalar, const N: usize, U> Clone for GenericVector<T, N, U> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Scalar, const N: usize, U> std::ops::Mul<T> for GenericVector<T, N, U> {
    type Output = GenericVector<T, N, U>;

    fn mul(self, rhs: T) -> Self::Output {
        Self::from(self.vec.map(|x| x * rhs))
    }
}

impl<T: Scalar, const N: usize, U> std::ops::Div<T> for GenericVector<T, N, U> {
    type Output = GenericVector<T, N, U>;

    fn div(self, rhs: T) -> Self::Output {
        Self::from(self.vec.map(|x| x / rhs))
    }
}

impl<T: SignedScalar, const N: usize, U: HasSub, I> InnerProduct<GenericVector<T, N, I>>
    for GenericVector<T, N, U>
{
    fn dot(&self, rhs: GenericVector<T, N, I>) -> Self::ScalarType {
        let mut res = T::zero();
        for i in 0..N {
            res += self.vec[i] * rhs.vec[i]
        }
        res
    }
}

impl<T: Scalar, const N: usize, U> From<[T; N]> for GenericVector<T, N, U> {
    fn from(vec: [T; N]) -> Self {
        let p = std::marker::PhantomData;
        Self { vec, p }
    }
}

// interactions between vector/point/normal
impl<T: Scalar, const N: usize> From<Vector<T, N>> for Normal<T, N> {
    fn from(v: Vector<T, N>) -> Normal<T, N> {
        let p = std::marker::PhantomData;
        Normal { vec: v.vec, p }
    }
}
impl<T: Scalar, const N: usize> From<Vector<T, N>> for Point<T, N> {
    fn from(v: Vector<T, N>) -> Point<T, N> {
        let p = std::marker::PhantomData;
        Point { vec: v.vec, p }
    }
}
impl<T: Scalar, const N: usize> From<Point<T, N>> for Vector<T, N> {
    fn from(v: Point<T, N>) -> Vector<T, N> {
        let p = std::marker::PhantomData;
        Vector { vec: v.vec, p }
    }
}
impl<T: Scalar, const N: usize> std::ops::Sub<Point<T, N>> for Point<T, N> {
    type Output = Vector<T, N>;

    fn sub(self, rhs: Point<T, N>) -> Self::Output {
        Vector::from(self) - Vector::from(rhs)
    }
}
impl<T: Scalar, const N: usize> std::ops::Add<Point<T, N>> for Vector<T, N> {
    type Output = Point<T, N>;

    fn add(self, rhs: Point<T, N>) -> Self::Output {
        Point::from(self) + rhs
    }
}
impl<T: Scalar, const N: usize> std::ops::Add<Vector<T, N>> for Point<T, N> {
    type Output = Point<T, N>;

    fn add(self, rhs: Vector<T, N>) -> Self::Output {
        self + Point::from(rhs)
    }
}
impl<T: Scalar, const N: usize> std::ops::Sub<Vector<T, N>> for Point<T, N> {
    type Output = Point<T, N>;

    fn sub(self, rhs: Vector<T, N>) -> Self::Output {
        Point::from(self - Point::from(rhs))
    }
}

impl Norm for Vector2<f32> {}
impl Norm for Vector2<f64> {}
impl Norm for Vector3<f32> {}
impl Norm for Vector3<f64> {}
impl Norm for Vector4<f32> {}
impl Norm for Vector4<f64> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_diff_is_vector() {
        let p1 = Point3::<f64>::new(1., 2., 3.);
        let p2 = Point3::<f64>::new(4., 5., 6.);

        let v: Vector3<f64> = p2 - p1;

        assert_eq!(v, Vector3::<f64>::new(3., 3., 3.));
    }

    #[test]
    fn cross_product() {
        let p1 = Vector3::<f64>::new(1., 0., 0.);
        let p2 = Vector3::<f64>::new(0., 1., 0.);
        let p3 = Vector3::<f64>::new(0., 0., 1.);

        assert_eq!(p1.cross(p2), p3);
        assert_eq!(p2.cross(p3), p1);
        assert_eq!(p3.cross(p1), p2);

        assert_eq!(p2.cross(p1), -p3);
        assert_eq!(p3.cross(p2), -p1);
        assert_eq!(p1.cross(p3), -p2);
    }

    #[test]
    fn dot() {
        let p1 = Vector3::<f64>::new(1., 0., 0.);
        let p2 = Vector3::<f64>::new(0., 1., 0.);
        let p3 = Vector3::<f64>::new(0., 0., 1.);

        assert_eq!(p1.dot(p2), 0.0);
        assert_eq!(p2.dot(p3), 0.0);
        assert_eq!(p3.dot(p1), 0.0);

        assert_eq!(p1.dot(p1), 1.0);
        assert_eq!(p2.dot(p2), 1.0);
        assert_eq!(p3.dot(p3), 1.0);

        let p4 = Vector3::<f64>::new(1., 2., 3.);
        let p5 = Vector3::<f64>::new(4., 5., 6.);
        let p6 = Vector3::<f64>::new(7., 8., 9.);

        assert_eq!(p4.dot(p5), 32.0);
        assert_eq!(p5.dot(p6), 122.0);
    }

    #[test]
    fn normalize() {
        use crate::scalar::fequals;

        let p1 = Vector3::<f64>::new(1.3, 2.4, 3.6);
        let p2 = Vector3::<f64>::new(4.6, 3.2, 9.98);
        let p3 = Vector3::<f64>::new(0.1, 0.8, 0.03);

        assert!(fequals(p1.normalize().norm(), 1.0));
        assert!(fequals(p2.normalize().norm(), 1.0));
        assert!(fequals(p3.normalize().norm(), 1.0));
    }
}
