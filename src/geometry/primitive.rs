use crate::{
    geometry::{bbox::Bbox3 /* bbox.rs file and this file are tightly coupled */, ray::Rayf},
    math::{
        scalar::Float,
        transform::{TransformMap, Transformf},
        vector::{Normal3f, Point2f, Point3f, Vector3f},
    },
};

use std::rc::Rc;

#[derive(Debug, Clone, Copy)]
pub struct SurfaceInteraction {
    pub p: Point3f,
    pub n: Normal3f,
    pub uv: Point2f,
    pub dpdu: Vector3f,
    pub dpdv: Vector3f,
}

#[derive(Debug, Clone, Copy)]
pub struct IntersectionResult {
    pub surface_interaction: SurfaceInteraction,
    pub t_ray: Float,
}

pub trait Bounded3 {
    fn bbox(&self) -> Bbox3;

    fn xmin(&self) -> Float {
        self.bbox().pmin.x()
    }
    fn ymin(&self) -> Float {
        self.bbox().pmin.y()
    }
    fn zmin(&self) -> Float {
        self.bbox().pmin.z()
    }
    fn xmax(&self) -> Float {
        self.bbox().pmax.x()
    }
    fn ymax(&self) -> Float {
        self.bbox().pmax.y()
    }
    fn zmax(&self) -> Float {
        self.bbox().pmax.z()
    }
}

impl SurfaceInteraction {
    pub fn new(p: Point3f, uv: Point2f, dpdu: Vector3f, dpdv: Vector3f) -> Self {
        SurfaceInteraction {
            p,
            n: Normal3f::from(dpdu.cross(dpdv).normalize()),
            uv,
            dpdu,
            dpdv,
        }
    }
}

pub trait IntersectsRay {
    fn do_intersect(&self, ray: Rayf) -> bool;
    fn intersect(&self, ray: Rayf) -> Option<IntersectionResult>;
}

pub trait Primitive: IntersectsRay + Bounded3 {}

pub type PrimitiveHandle = Rc<dyn Primitive>;
pub fn alloc_primitive<T: Primitive + 'static>(primitive: T) -> PrimitiveHandle {
    Rc::new(primitive) as PrimitiveHandle
}

impl Bounded3 for PrimitiveHandle {
    fn bbox(&self) -> Bbox3 {
        (**self).bbox()
    }
}

pub struct TransformedPrimitive<T: Primitive> {
    primitive: T,
    otow: Transformf,
}

impl<T: Primitive> TransformedPrimitive<T> {
    pub fn new(primitive: T, otow: Transformf) -> Self {
        TransformedPrimitive { primitive, otow }
    }
    fn object_to_world<I, U>(&self, obj: I) -> U
    where
        Transformf: TransformMap<I, Output = U>,
    {
        self.otow.map(obj)
    }
    fn world_to_object<I, U>(&self, obj: I) -> U
    where
        Transformf: TransformMap<I, Output = U>,
    {
        self.otow.map_inverse(obj)
    }
}

impl<T: Primitive> Bounded3 for TransformedPrimitive<T> {
    fn bbox(&self) -> Bbox3 {
        self.object_to_world(self.primitive.bbox())
    }
}

impl<T: Primitive> IntersectsRay for TransformedPrimitive<T> {
    fn intersect(&self, ray: Rayf) -> Option<IntersectionResult> {
        // note that "t" computed in object space is the same as in world space
        let transformed_ray = self.world_to_object(ray);
        self.primitive.intersect(transformed_ray)
    }

    fn do_intersect(&self, ray: Rayf) -> bool {
        let transformed_ray = self.world_to_object(ray);
        self.primitive.do_intersect(transformed_ray)
    }
}

#[derive(Default)]
pub struct PrimitiveList {
    primitives: Vec<PrimitiveHandle>,
}

impl PrimitiveList {
    pub fn new() -> PrimitiveList {
        PrimitiveList {
            primitives: Vec::new(),
        }
    }

    pub fn add_primitive<T: Primitive + 'static>(&mut self, primitive: T) {
        self.primitives.push(alloc_primitive(primitive));
    }
}

impl Bounded3 for PrimitiveList {
    fn bbox(&self) -> Bbox3 {
        let mut bbox = Bbox3::default();
        for primitive in self.primitives.iter() {
            bbox += (*primitive).bbox();
        }
        bbox
    }
}

impl IntersectsRay for PrimitiveList {
    fn intersect(&self, ray: Rayf) -> Option<IntersectionResult> {
        let mut result = None;
        let mut ray_mut = ray;
        for primitive_handle in self.primitives.iter() {
            if let Some(intersection_result) = (*primitive_handle).intersect(ray) {
                ray_mut.tmax = intersection_result.t_ray;
                result = Some(intersection_result);
            }
        }
        result
    }
    fn do_intersect(&self, ray: Rayf) -> bool {
        for primitive_handle in self.primitives.iter() {
            if (*primitive_handle).do_intersect(ray) {
                return true;
            }
        }
        false
    }
}

impl<T: Primitive> Primitive for TransformedPrimitive<T> {}
impl Primitive for PrimitiveList {}
