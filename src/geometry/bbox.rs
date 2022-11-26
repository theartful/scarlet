use crate::{
    geometry::{
        ray::Ray,
        shape::{Bounded3, IntersectionResult, IntersectsRay, Shape, SurfaceInteraction},
    },
    math::{
        scalar::{Float, GFloat, Int, Scalar},
        vector::{Normal3, Point2, Point3, Vector3},
    },
};

pub type Bbox3f = Bbox3<Float>;
pub type Bbox3i = Bbox3<Int>;

#[derive(Debug, Clone, Copy)]
pub struct Bbox3<T: Scalar> {
    pub pmin: Point3<T>,
    pub pmax: Point3<T>,
}

impl<T: Scalar> Default for Bbox3<T> {
    fn default() -> Self {
        let highest = T::highest();
        let lowest = T::lowest();
        Bbox3 {
            pmin: Point3::<T>::new(highest, highest, highest),
            pmax: Point3::<T>::new(lowest, lowest, lowest),
        }
    }
}

impl<T: Scalar> Bbox3<T> {
    pub fn new(p1: Point3<T>, p2: Point3<T>) -> Self {
        Bbox3 {
            pmin: Point3::<T>::new(p1.x().min(p2.x()), p1.y().min(p2.y()), p1.z().min(p2.z())),
            pmax: Point3::<T>::new(p1.x().max(p2.x()), p1.y().max(p2.y()), p1.z().max(p2.z())),
        }
    }

    pub fn union(&self, rhs: Bbox3<T>) -> Self {
        Bbox3 {
            pmin: Point3::<T>::new(
                self.xmin().min(rhs.xmin()),
                self.ymin().min(rhs.ymin()),
                self.zmin().min(rhs.zmin()),
            ),
            pmax: Point3::<T>::new(
                self.xmax().max(rhs.xmax()),
                self.ymax().max(rhs.ymax()),
                self.zmax().max(rhs.zmax()),
            ),
        }
    }

    pub fn intersect(&self, rhs: Bbox3<T>) -> Self {
        Bbox3 {
            pmin: Point3::<T>::new(
                self.xmin().max(rhs.xmin()),
                self.ymin().max(rhs.ymin()),
                self.zmin().max(rhs.zmin()),
            ),
            pmax: Point3::<T>::new(
                self.xmax().min(rhs.xmax()),
                self.ymax().min(rhs.ymax()),
                self.zmax().min(rhs.zmax()),
            ),
        }
    }

    pub fn overlaps(&self, rhs: &Bbox3<T>) -> bool {
        self.xmax() > rhs.xmin()
            && rhs.xmax() > self.xmin()
            && self.ymax() > rhs.ymin()
            && rhs.ymax() > self.ymin()
            && self.zmax() > rhs.zmin()
            && rhs.zmax() > self.zmin()
    }

    pub fn inside(&self, rhs: Point3<T>) -> bool {
        rhs.x() >= self.xmin()
            && rhs.x() <= self.xmax()
            && rhs.y() >= self.ymin()
            && rhs.y() <= self.ymax()
            && rhs.z() >= self.zmin()
            && rhs.z() <= self.zmax()
    }

    pub fn diagonal(&self) -> Vector3<T> {
        self.pmax - self.pmin
    }

    pub fn center(&self) -> Point3<T>
    where
        T: GFloat,
    {
        (self.pmax + self.pmin) * T::half()
    }

    pub fn max_extent(&self) -> ((T, T), u8) {
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

impl<T: Scalar> Bounded3<T> for Bbox3<T> {
    fn bbox(&self) -> Bbox3<T> {
        *self
    }
}

impl<T: Scalar> std::ops::Add<Bbox3<T>> for Bbox3<T> {
    type Output = Bbox3<T>;

    fn add(self, rhs: Bbox3<T>) -> Self::Output {
        self.union(rhs)
    }
}

impl<T: Scalar> std::ops::AddAssign<Bbox3<T>> for Bbox3<T> {
    fn add_assign(&mut self, rhs: Bbox3<T>) {
        *self = self.union(rhs);
    }
}

impl<T: GFloat> Bbox3<T> {
    /// Performs ray-box intersection and returns the intersection period of
    /// the ray
    pub fn intersect_ray(self, ray: Ray<T>) -> Option<(T, T)> {
        // should we store the reciprocal of ray.direction?
        let tmin = (self.pmin - ray.origin) / ray.direction;
        let tmax = (self.pmax - ray.origin) / ray.direction;

        let t1 = tmin.min(tmax);
        let t2 = tmin.max(tmax);

        let tnear = t1.x().max(t1.y()).max(t1.z());
        let tfar = t2.x().min(t2.y()).min(t2.z());

        if tnear > ray.tmax || tfar < T::zero() {
            None
        } else {
            Some((tnear.max(T::zero()), tfar.min(ray.tmax)))
        }
    }
}

impl<T: GFloat> IntersectsRay<T> for Bbox3<T> {
    fn do_intersect(&self, ray: Ray<T>) -> bool {
        self.intersect_ray(ray).is_some()
    }

    fn intersect(&self, ray: Ray<T>) -> Option<IntersectionResult<T>> {
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
                    p: Point3::new(self.xmin(), p.y(), p.z()),
                    n: Normal3::new(-T::one(), T::zero(), T::zero()),
                    uv: Point2::new(T::zero(), T::zero()),
                    dpdu: Vector3::new(T::zero(), T::zero(), -T::one()),
                    dpdv: Vector3::new(T::zero(), T::one(), T::zero()),
                },
                // 1 5 7 3
                1 => SurfaceInteraction {
                    p: Point3::new(self.xmax(), p.y(), p.z()),
                    n: Normal3::new(T::one(), T::zero(), T::zero()),
                    uv: Point2::new(T::zero(), T::zero()),
                    dpdu: Vector3::new(T::zero(), T::zero(), T::one()),
                    dpdv: Vector3::new(T::zero(), T::one(), T::zero()),
                },
                // 0 1 5 4
                2 => SurfaceInteraction {
                    p: Point3::new(p.x(), self.ymin(), p.z()),
                    n: Normal3::new(T::zero(), -T::one(), T::zero()),
                    uv: Point2::new(T::zero(), T::zero()),
                    dpdu: Vector3::new(T::one(), T::zero(), T::zero()),
                    dpdv: Vector3::new(T::zero(), T::zero(), -T::one()),
                },
                // 2 6 7 3
                3 => SurfaceInteraction {
                    p: Point3::new(p.x(), self.ymax(), p.z()),
                    n: Normal3::new(T::zero(), T::one(), T::zero()),
                    uv: Point2::new(T::zero(), T::zero()),
                    dpdu: Vector3::new(T::one(), T::zero(), T::zero()),
                    dpdv: Vector3::new(T::zero(), T::zero(), T::one()),
                },
                // 0 2 3 1
                4 => SurfaceInteraction {
                    p: Point3::new(p.x(), p.y(), self.zmin()),
                    n: Normal3::new(T::zero(), T::zero(), -T::one()),
                    uv: Point2::new(T::zero(), T::zero()),
                    dpdu: Vector3::new(T::one(), T::zero(), T::zero()),
                    dpdv: Vector3::new(T::zero(), T::one(), T::zero()),
                },
                // 4 6 7 5
                5 => SurfaceInteraction {
                    p: Point3::new(p.x(), p.y(), self.zmax()),
                    n: Normal3::new(T::zero(), T::zero(), T::one()),
                    uv: Point2::new(T::zero(), T::zero()),
                    dpdu: Vector3::new(-T::one(), T::zero(), T::zero()),
                    dpdv: Vector3::new(T::zero(), T::one(), T::zero()),
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

impl<T: GFloat> Shape<T> for Bbox3<T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{interval::Interval, scalar::AlmostEqual};
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

            let bbox = Bbox3::<f64>::new(
                Point3::new(
                    dist.sample(&mut rng),
                    dist.sample(&mut rng),
                    dist.sample(&mut rng),
                ),
                Point3::new(
                    dist.sample(&mut rng),
                    dist.sample(&mut rng),
                    dist.sample(&mut rng),
                ),
            );
            let bboxi =
                Bbox3::<Interval<f64>>::new(Point3::from(bbox.pmin), Point3::from(bbox.pmax));

            //     6----7      y
            //    /|   /|      ^
            //   2----3 |      |
            //   | 4--|-5      |
            //   |/   |/       |
            //   0----1        -----------> x
            //
            let ps = [
                Point3::new(bbox.xmin(), bbox.ymin(), bbox.zmin()),
                Point3::new(bbox.xmax(), bbox.ymin(), bbox.zmin()),
                Point3::new(bbox.xmin(), bbox.ymax(), bbox.zmin()),
                Point3::new(bbox.xmax(), bbox.ymax(), bbox.zmin()),
                Point3::new(bbox.xmin(), bbox.ymin(), bbox.zmax()),
                Point3::new(bbox.xmax(), bbox.ymin(), bbox.zmax()),
                Point3::new(bbox.xmin(), bbox.ymax(), bbox.zmax()),
                Point3::new(bbox.xmax(), bbox.ymax(), bbox.zmax()),
            ];
            let psi: [Point3<Interval<f64>>; 8] = [
                Point3::from(ps[0]),
                Point3::from(ps[1]),
                Point3::from(ps[2]),
                Point3::from(ps[3]),
                Point3::from(ps[4]),
                Point3::from(ps[5]),
                Point3::from(ps[6]),
                Point3::from(ps[7]),
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
            let face0i = (
                psi[faces[0].0],
                psi[faces[0].1],
                psi[faces[0].2],
                psi[faces[0].3],
            );
            let face1i = (
                psi[faces[1].0],
                psi[faces[1].1],
                psi[faces[1].2],
                psi[faces[1].3],
            );

            let dist_t = Uniform::new_inclusive(0.0, 1.0);

            // sample a random point from the first face
            let (t0, t1) = (dist_t.sample(&mut rng), dist_t.sample(&mut rng));
            let p0 = face0.0 + (face0.1 - face0.0) * t0 + (face0.2 - face0.0) * t1;
            let p0i = face0i.0 + (face0i.1 - face0i.0) * t0 + (face0i.2 - face0i.0) * t1;

            // sample a random point from the second face
            let (t0, t1) = (dist_t.sample(&mut rng), dist_t.sample(&mut rng));
            let p1 = face1.0 + (face1.1 - face1.0) * t0 + (face1.2 - face1.0) * t1;
            let p1i = face1i.0 + (face1i.1 - face1i.0) * t0 + (face1i.2 - face1i.0) * t1;

            // create ray connecting p0 and p1
            let dir = p1 - p0;
            let diri = p1i - p0i;

            // line  fully passes through the cube
            let t_offset = dist_t.sample(&mut rng);
            let origin = p0 - dir * t_offset;
            let origini = p0i - diri * t_offset;
            let ray = Ray::<f64>::new(origin, dir, 1.0 + 2.0 * t_offset);
            let rayi = Ray::<Interval<f64>>::new(
                origini,
                diri,
                Interval::from(1.0) + Interval::from(2.0) * t_offset,
            );

            let intersection = bbox.intersect_ray(ray);
            let intersectioni = bboxi.intersect_ray(rayi);

            assert!(intersection.is_some());
            assert!(intersection.unwrap().0.almost_eq(t_offset));
            assert!(intersection.unwrap().1.almost_eq(1.0 + t_offset));

            assert!(intersectioni.is_some());
            assert!(intersectioni.unwrap().0.almost_eq(t_offset));
            assert!(intersectioni
                .unwrap()
                .1
                .almost_eq(Interval::one() + t_offset));

            // line goes into the cube but doesn't come out
            let ray = Ray::<f64>::new(origin, dir, 1.0);
            let rayi = Ray::<Interval<f64>>::new(origini, diri, Interval::one());

            let intersection = bbox.intersect_ray(ray);
            let intersectioni = bboxi.intersect_ray(rayi);

            assert!(intersection.is_some());
            assert!(intersection.unwrap().0.almost_eq(t_offset));
            assert!(intersection.unwrap().1.almost_eq(1.0));

            assert!(intersectioni.is_some());
            assert!(intersectioni.unwrap().0.almost_eq(t_offset));
            assert!(intersectioni.unwrap().1.almost_eq(1.0));

            // line comes out of the cube but doesn't go in
            let ray = Ray::<f64>::new(p0 + dir * t_offset, dir, 1.0);
            let rayi = Ray::<Interval<f64>>::new(p0i + diri * t_offset, diri, Interval::one());

            let intersection = bbox.intersect_ray(ray);
            let intersectioni = bboxi.intersect_ray(rayi);

            assert!(intersection.is_some());
            assert!(intersection.unwrap().0.almost_eq(0.0));
            assert!(intersection.unwrap().1.almost_eq(1.0 - t_offset));

            assert!(intersectioni.is_some());
            assert!(intersectioni.unwrap().0.almost_eq(0.0));
            assert!(intersectioni.unwrap().1.almost_eq(1.0 - t_offset));

            // line fully inside the cube
            let ray = Ray::<f64>::new(p0 + dir * t_offset, p1 - (p0 + dir * t_offset), 1.0);
            let rayi = Ray::<Interval<f64>>::new(
                p0i + diri * t_offset,
                p1i - (p0i + diri * t_offset),
                Interval::one(),
            );

            let intersection = bbox.intersect_ray(ray);
            let intersectioni = bboxi.intersect_ray(rayi);

            assert!(intersection.is_some());
            assert!(intersection.unwrap().0.almost_eq(0.0));
            assert!(intersection.unwrap().1.almost_eq(1.0));

            assert!(intersectioni.is_some());
            assert!(intersectioni.unwrap().0.almost_eq(0.0));
            assert!(intersectioni.unwrap().1.almost_eq(1.0));

            // line outside the cube (almost reaching p1)
            let ray = Ray::<f64>::new(p1 + dir * 0.01, dir, 1.0);
            let rayi = Ray::<Interval<f64>>::new(p1i + diri * 0.01, diri, Interval::one());

            let intersection = bbox.intersect_ray(ray);
            let intersectioni = bboxi.intersect_ray(rayi);

            assert!(intersection.is_none());
            assert!(intersectioni.is_none());

            // line outside the cube (almost reaching p0)
            let ray = Ray::<f64>::new(p0 - dir * 1.01, dir, 1.0);
            let rayi = Ray::<Interval<f64>>::new(p0i - diri * 1.01, diri, Interval::one());

            let intersection = bbox.intersect_ray(ray);
            let intersectioni = bboxi.intersect_ray(rayi);

            assert!(intersection.is_none());
            assert!(intersectioni.is_none());
        }
    }
}
