use crate::{
    geometry::{bbox::Bbox3 /* bbox.rs file and this file are tightly coupled */, ray::Ray},
    math::{
        scalar::{GFloat, Scalar},
        vector::{Normal3, Point2, Point3, Vector3},
    },
};

pub struct SurfaceInteraction<T: GFloat> {
    pub p: Point3<T>,
    pub n: Normal3<T>,
    pub uv: Point2<T>,
    pub dpdu: Vector3<T>,
    pub dpdv: Vector3<T>,
}

pub struct IntersectionResult<T: GFloat> {
    pub surface_interaction: SurfaceInteraction<T>,
    pub t_ray: T,
}

pub trait Bounded3<T: Scalar> {
    fn bbox(&self) -> Bbox3<T>;

    fn xmin(&self) -> T {
        self.bbox().pmin.x()
    }
    fn ymin(&self) -> T {
        self.bbox().pmin.y()
    }
    fn zmin(&self) -> T {
        self.bbox().pmin.z()
    }
    fn xmax(&self) -> T {
        self.bbox().pmax.x()
    }
    fn ymax(&self) -> T {
        self.bbox().pmax.y()
    }
    fn zmax(&self) -> T {
        self.bbox().pmax.z()
    }
}

impl<T: GFloat> SurfaceInteraction<T> {
    pub fn new(p: Point3<T>, uv: Point2<T>, dpdu: Vector3<T>, dpdv: Vector3<T>) -> Self {
        SurfaceInteraction {
            p,
            n: Normal3::from(dpdu.cross(dpdv).normalize()),
            uv,
            dpdu,
            dpdv,
        }
    }
}

pub trait IntersectsRay<T: GFloat> {
    fn do_intersect(&self, ray: Ray<T>) -> bool;
    fn intersect(&self, ray: Ray<T>) -> Option<IntersectionResult<T>>;
}

pub trait Shape<T: GFloat>: IntersectsRay<T> + Bounded3<T> {}
