use crate::math::{
    interval::Intervalf,
    scalar::{Float, Scalar},
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
    pub fn new(origin: Point3<T>, direction: Vector3<T>, tmax: T) -> Self {
        Ray {
            origin,
            direction,
            tmax,
        }
    }

    pub fn eval(&self, t: T) -> Point3<T> {
        self.origin + self.direction * t
    }
}
