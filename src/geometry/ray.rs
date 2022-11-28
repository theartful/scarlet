use crate::math::{
    interval::Intervalf,
    scalar::{Float, GFloat, Scalar},
    transform::{Transform, TransformMap},
    vector::{Point3, Vector3},
};

pub type Rayf = Ray<Float>;
pub type Rayfi = Ray<Intervalf>;

#[derive(Copy, Clone, Debug)]
pub struct Ray<T: Scalar> {
    pub origin: Point3<T>,
    pub direction: Vector3<T>,
    pub tmax: T,
}

impl<T: Scalar> Ray<T> {
    #[inline]
    pub fn new(origin: Point3<T>, direction: Vector3<T>, tmax: T) -> Self {
        Ray {
            origin,
            direction,
            tmax,
        }
    }

    #[inline]
    pub fn eval(&self, t: T) -> Point3<T> {
        self.origin + self.direction * t
    }
}

impl<T, U> TransformMap<Ray<U>> for Transform<T>
where
    T: GFloat,
    U: Scalar
        + std::ops::Mul<T, Output = U>
        + std::ops::Div<T, Output = U>
        + std::ops::Add<T, Output = U>,
{
    type Output = Ray<U>;

    #[inline]
    fn map(&self, ray: Ray<U>) -> Self::Output {
        Ray::new(self.map(ray.origin), self.map(ray.direction), ray.tmax)
    }

    #[inline]
    fn map_inverse(&self, ray: Ray<U>) -> Self::Output {
        Ray::new(
            self.map_inverse(ray.origin),
            self.map_inverse(ray.direction),
            ray.tmax,
        )
    }
}
