use crate::{
    geometry::{
        primitive::{Bounded3, IntersectionResult, IntersectsRay, Primitive, SurfaceInteraction},
        ray::Rayf,
    },
    math::{
        scalar::{Float, GFloat, Scalar},
        transform::{Matrix4x4f, TransformMap, Transformf},
        vector::{Normal3f, Point2f, Point3f, Vector3f},
    },
};

#[derive(Debug, Clone, Copy)]
pub struct Bbox3 {
    pub pmin: Point3f,
    pub pmax: Point3f,
}

impl Default for Bbox3 {
    fn default() -> Self {
        let highest = Float::highest();
        let lowest = Float::lowest();
        Bbox3 {
            pmin: Point3f::new(highest, highest, highest),
            pmax: Point3f::new(lowest, lowest, lowest),
        }
    }
}

impl Bbox3 {
    pub fn new(p1: Point3f, p2: Point3f) -> Self {
        Bbox3 {
            pmin: Point3f::new(p1.x().min(p2.x()), p1.y().min(p2.y()), p1.z().min(p2.z())),
            pmax: Point3f::new(p1.x().max(p2.x()), p1.y().max(p2.y()), p1.z().max(p2.z())),
        }
    }

    pub fn union(&self, rhs: &Self) -> Self {
        Bbox3 {
            pmin: Point3f::new(
                self.xmin().min(rhs.xmin()),
                self.ymin().min(rhs.ymin()),
                self.zmin().min(rhs.zmin()),
            ),
            pmax: Point3f::new(
                self.xmax().max(rhs.xmax()),
                self.ymax().max(rhs.ymax()),
                self.zmax().max(rhs.zmax()),
            ),
        }
    }

    pub fn intersect(&self, rhs: &Self) -> Self {
        Bbox3 {
            pmin: Point3f::new(
                self.xmin().max(rhs.xmin()),
                self.ymin().max(rhs.ymin()),
                self.zmin().max(rhs.zmin()),
            ),
            pmax: Point3f::new(
                self.xmax().min(rhs.xmax()),
                self.ymax().min(rhs.ymax()),
                self.zmax().min(rhs.zmax()),
            ),
        }
    }

    pub fn overlaps(&self, rhs: &Self) -> bool {
        self.xmax() > rhs.xmin()
            && rhs.xmax() > self.xmin()
            && self.ymax() > rhs.ymin()
            && rhs.ymax() > self.ymin()
            && self.zmax() > rhs.zmin()
            && rhs.zmax() > self.zmin()
    }

    pub fn inside(&self, rhs: Point3f) -> bool {
        rhs.x() >= self.xmin()
            && rhs.x() <= self.xmax()
            && rhs.y() >= self.ymin()
            && rhs.y() <= self.ymax()
            && rhs.z() >= self.zmin()
            && rhs.z() <= self.zmax()
    }

    pub fn diagonal(&self) -> Vector3f {
        self.pmax - self.pmin
    }

    pub fn center(&self) -> Point3f {
        (self.pmax + self.pmin) * Float::half()
    }

    pub fn max_extent(&self) -> ((Float, Float), u8) {
        let x_extent = self.pmax.x() - self.pmin.x();
        let y_extent = self.pmax.y() - self.pmin.y();
        let z_extent = self.pmax.z() - self.pmin.z();

        if x_extent > y_extent {
            if x_extent > z_extent {
                ((self.pmin.x(), self.pmax.x()), 0)
            } else {
                ((self.pmin.z(), self.pmax.z()), 2)
            }
        } else if y_extent > z_extent {
            ((self.pmin.y(), self.pmax.y()), 1)
        } else {
            ((self.pmin.z(), self.pmax.z()), 2)
        }
    }
}

impl Bounded3 for Bbox3 {
    fn bbox(&self) -> Bbox3 {
        *self
    }
}

impl std::ops::Add<Bbox3> for Bbox3 {
    type Output = Bbox3;

    fn add(self, rhs: Bbox3) -> Self::Output {
        self.union(&rhs)
    }
}

impl std::ops::AddAssign<Bbox3> for Bbox3 {
    fn add_assign(&mut self, rhs: Bbox3) {
        *self = self.union(&rhs);
    }
}

impl Bbox3 {
    /// Performs ray-box intersection and returns the intersection period of
    /// the ray
    pub fn intersect_ray(self, ray: Rayf) -> Option<(Float, Float)> {
        // should we store the reciprocal of ray.direction?
        let tmin = (self.pmin - ray.origin) / ray.direction;
        let tmax = (self.pmax - ray.origin) / ray.direction;

        let t1 = tmin.min(tmax);
        let t2 = tmin.max(tmax);

        let tnear = t1.x().max(t1.y()).max(t1.z());
        let tfar = t2.x().min(t2.y()).min(t2.z());

        if tnear > ray.tmax || tfar < Float::zero() || tnear > tfar {
            None
        } else {
            Some((tnear.max(Float::zero()), tfar.min(ray.tmax)))
        }
    }
}

impl IntersectsRay for Bbox3 {
    fn do_intersect(&self, ray: Rayf) -> bool {
        self.intersect_ray(ray).is_some()
    }

    fn intersect(&self, ray: Rayf) -> Option<IntersectionResult> {
        // FIXME: this does not work when we're inside the box!
        self.intersect_ray(ray).map(|(t_ray, _)| {
            let p = ray.eval(t_ray);

            // we need to find which face we hit
            let distances = [
                (p.x() - self.xmin()).abs(),
                (p.x() - self.xmax()).abs(),
                (p.y() - self.ymin()).abs(),
                (p.y() - self.ymax()).abs(),
                (p.z() - self.zmin()).abs(),
                (p.z() - self.zmax()).abs(),
            ];

            let mut min_distance_idx = 0;
            for i in 0..6 {
                if distances[i] < distances[min_distance_idx] {
                    min_distance_idx = i;
                }
            }

            // TODO: implement uv mapping for bbox
            // why am I even doing this? I should implement a general mesh
            // structure with custom uv mapping!

            //
            //     6----7      y
            //    /|   /|      ^
            //   2----3 |      |
            //   | 4--|-5      |
            //   |/   |/       |
            //   0----1        -----------> x
            //
            //   0        4
            //   1        5
            //   2        6
            //   3        7
            //   ----------->z
            //
            //  0.75  6-----7
            //        |     |                          ^ v
            //        |     |                          |
            //   0.5  2-----3-----7-----6-----2        |
            //        |     |     |     |     |        |
            //        |     |     |     |     |        |             u
            //  0.25  0-----1-----5-----4-----0        -------------->
            //        |     |
            //        |     |
            //     0  4-----5
            //
            //        0    0.25  0.5   0.75  1
            //
            let surface_interaction = match min_distance_idx {
                // 0 2 6 4
                0 => SurfaceInteraction {
                    p: Point3f::new(self.xmin(), p.y(), p.z()),
                    n: Normal3f::new(-Float::one(), Float::zero(), Float::zero()),
                    uv: Point2f::new(Float::zero(), Float::zero()),
                    dpdu: Vector3f::new(Float::zero(), Float::zero(), -Float::one()),
                    dpdv: Vector3f::new(Float::zero(), Float::one(), Float::zero()),
                },
                // 1 5 7 3
                1 => SurfaceInteraction {
                    p: Point3f::new(self.xmax(), p.y(), p.z()),
                    n: Normal3f::new(Float::one(), Float::zero(), Float::zero()),
                    uv: Point2f::new(Float::zero(), Float::zero()),
                    dpdu: Vector3f::new(Float::zero(), Float::zero(), Float::one()),
                    dpdv: Vector3f::new(Float::zero(), Float::one(), Float::zero()),
                },
                // 0 1 5 4
                2 => SurfaceInteraction {
                    p: Point3f::new(p.x(), self.ymin(), p.z()),
                    n: Normal3f::new(Float::zero(), -Float::one(), Float::zero()),
                    uv: Point2f::new(Float::zero(), Float::zero()),
                    dpdu: Vector3f::new(Float::one(), Float::zero(), Float::zero()),
                    dpdv: Vector3f::new(Float::zero(), Float::zero(), -Float::one()),
                },
                // 2 6 7 3
                3 => SurfaceInteraction {
                    p: Point3f::new(p.x(), self.ymax(), p.z()),
                    n: Normal3f::new(Float::zero(), Float::one(), Float::zero()),
                    uv: Point2f::new(Float::zero(), Float::zero()),
                    dpdu: Vector3f::new(Float::one(), Float::zero(), Float::zero()),
                    dpdv: Vector3f::new(Float::zero(), Float::zero(), Float::one()),
                },
                // 0 2 3 1
                4 => SurfaceInteraction {
                    p: Point3f::new(p.x(), p.y(), self.zmin()),
                    n: Normal3f::new(Float::zero(), Float::zero(), -Float::one()),
                    uv: Point2f::new(Float::zero(), Float::zero()),
                    dpdu: Vector3f::new(Float::one(), Float::zero(), Float::zero()),
                    dpdv: Vector3f::new(Float::zero(), Float::one(), Float::zero()),
                },
                // 4 6 7 5
                5 => SurfaceInteraction {
                    p: Point3f::new(p.x(), p.y(), self.zmax()),
                    n: Normal3f::new(Float::zero(), Float::zero(), Float::one()),
                    uv: Point2f::new(Float::zero(), Float::zero()),
                    dpdu: Vector3f::new(-Float::one(), Float::zero(), Float::zero()),
                    dpdv: Vector3f::new(Float::zero(), Float::one(), Float::zero()),
                },
                _ => std::panic!(),
            };

            IntersectionResult {
                surface_interaction,
                t_ray,
            }
        })
    }
}

impl TransformMap<Bbox3> for Transformf {
    type Output = Bbox3;

    fn map(&self, bbox: Bbox3) -> Self::Output {
        map_bbox(self.matrix(), &bbox)
    }

    fn map_inverse(&self, bbox: Bbox3) -> Self::Output {
        map_bbox(self.inverse_matrix(), &bbox)
    }
}

#[inline(always)]
fn map_bbox(m: Matrix4x4f, bbox: &Bbox3) -> Bbox3 {
    // Graphics Gems: Transforming Axis-aligned Bounding Boxes
    // by James Arvo 1990
    let mut pmin = m.col(3);
    let mut pmax = pmin;

    let bbox_pmin: [Float; 3] = [bbox.pmin.x(), bbox.pmin.y(), bbox.pmin.z()];
    let bbox_pmax: [Float; 3] = [bbox.pmax.x(), bbox.pmax.y(), bbox.pmax.z()];

    for i in 0..3 {
        for j in 0..3 {
            let a = m[i][j] * bbox_pmin[j];
            let b = m[i][j] * bbox_pmax[j];

            if a < b {
                pmin[j] += a;
                pmax[j] += b;
            } else {
                pmin[j] += b;
                pmax[j] += a;
            }
        }
    }

    Bbox3::new(
        Point3f::new(pmin[0], pmin[1], pmin[2]),
        Point3f::new(pmax[0], pmax[1], pmax[2]),
    )
}

impl Primitive for Bbox3 {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::scalar::AlmostEqual;
    use rand::{
        distributions::{Distribution, Uniform},
        rngs::SmallRng,
        seq::SliceRandom,
        SeedableRng,
    };

    #[test]
    fn test_ray_box_intersection() {
        let seed: [u8; 32] = [73; 32];
        let mut rng = SmallRng::from_seed(seed);

        let num_tests = 100;
        for _ in 0..num_tests {
            let dist = Uniform::new_inclusive(-100.0, 100.0);

            let bbox = Bbox3::new(
                Point3f::new(
                    dist.sample(&mut rng),
                    dist.sample(&mut rng),
                    dist.sample(&mut rng),
                ),
                Point3f::new(
                    dist.sample(&mut rng),
                    dist.sample(&mut rng),
                    dist.sample(&mut rng),
                ),
            );

            //     6----7      y
            //    /|   /|      ^
            //   2----3 |      |
            //   | 4--|-5      |
            //   |/   |/       |
            //   0----1        -----------> x
            //
            let ps = [
                Point3f::new(bbox.xmin(), bbox.ymin(), bbox.zmin()),
                Point3f::new(bbox.xmax(), bbox.ymin(), bbox.zmin()),
                Point3f::new(bbox.xmin(), bbox.ymax(), bbox.zmin()),
                Point3f::new(bbox.xmax(), bbox.ymax(), bbox.zmin()),
                Point3f::new(bbox.xmin(), bbox.ymin(), bbox.zmax()),
                Point3f::new(bbox.xmax(), bbox.ymin(), bbox.zmax()),
                Point3f::new(bbox.xmin(), bbox.ymax(), bbox.zmax()),
                Point3f::new(bbox.xmax(), bbox.ymax(), bbox.zmax()),
            ];

            // choose two faces by random
            let mut faces = [
                (0, 1, 2, 3),
                (4, 5, 6, 7),
                (0, 1, 4, 5),
                (2, 3, 6, 7),
                (0, 2, 4, 6),
                (1, 3, 5, 7),
            ];
            faces.shuffle(&mut rng);

            let face0 = (
                ps[faces[0].0],
                ps[faces[0].1],
                ps[faces[0].2],
                ps[faces[0].3],
            );
            let face1 = (
                ps[faces[1].0],
                ps[faces[1].1],
                ps[faces[1].2],
                ps[faces[1].3],
            );

            let dist_t = Uniform::new_inclusive(0.0, 1.0);

            // sample a random point from the first face
            let (t0, t1) = (dist_t.sample(&mut rng), dist_t.sample(&mut rng));
            let p0 = face0.0 + (face0.1 - face0.0) * t0 + (face0.2 - face0.0) * t1;

            // sample a random point from the second face
            let (t0, t1) = (dist_t.sample(&mut rng), dist_t.sample(&mut rng));
            let p1 = face1.0 + (face1.1 - face1.0) * t0 + (face1.2 - face1.0) * t1;

            // create ray connecting p0 and p1
            let dir = p1 - p0;

            // line  fully passes through the cube
            let t_offset = dist_t.sample(&mut rng);
            let origin = p0 - dir * t_offset;
            let ray = Rayf::new(origin, dir, 1.0 + 2.0 * t_offset);

            let intersection = bbox.intersect_ray(ray);

            assert!(intersection.is_some());
            assert!(intersection.unwrap().0.almost_eq(t_offset));
            assert!(intersection.unwrap().1.almost_eq(1.0 + t_offset));

            // line goes into the cube but doesn't come out
            let ray = Rayf::new(origin, dir, 1.0);

            let intersection = bbox.intersect_ray(ray);

            assert!(intersection.is_some());
            assert!(intersection.unwrap().0.almost_eq(t_offset));
            assert!(intersection.unwrap().1.almost_eq(1.0));

            // line comes out of the cube but doesn't go in
            let ray = Rayf::new(p0 + dir * t_offset, dir, 1.0);

            let intersection = bbox.intersect_ray(ray);

            assert!(intersection.is_some());
            assert!(intersection.unwrap().0.almost_eq(0.0));
            assert!(intersection.unwrap().1.almost_eq(1.0 - t_offset));

            // line fully inside the cube
            let ray = Rayf::new(p0 + dir * t_offset, p1 - (p0 + dir * t_offset), 1.0);

            let intersection = bbox.intersect_ray(ray);

            assert!(intersection.is_some());
            assert!(intersection.unwrap().0.almost_eq(0.0));
            assert!(intersection.unwrap().1.almost_eq(1.0));

            // line outside the cube (almost reaching p1)
            let ray = Rayf::new(p1 + dir * 0.01, dir, 1.0);

            let intersection = bbox.intersect_ray(ray);

            assert!(intersection.is_none());

            // line outside the cube (almost reaching p0)
            let ray = Rayf::new(p0 - dir * 1.01, dir, 1.0);

            let intersection = bbox.intersect_ray(ray);

            assert!(intersection.is_none());
        }
    }
}
