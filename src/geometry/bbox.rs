use crate::{
    geometry::ray::Ray,
    math::{
        scalar::{Float, GFloat, Int, Scalar},
        vector::{Point3, Vector3},
    },
};

pub type Bbox3f = Bbox3<Float>;
pub type Bbox3i = Bbox3<Int>;

#[derive(Debug, Clone, Copy)]
pub struct Bbox3<T: Scalar> {
    pub pmin: Point3<T>,
    pub pmax: Point3<T>,
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

impl<T: Scalar> Bounded3<T> for Bbox3<T> {
    fn bbox(&self) -> Bbox3<T> {
        *self
    }
}

impl<T: Scalar> Bounded3<T> for Point3<T> {
    fn bbox(&self) -> Bbox3<T> {
        Bbox3::<T>::new(
            Point3::<T>::new(self.x(), self.y(), self.z()),
            Point3::<T>::new(self.x(), self.y(), self.z()),
        )
    }
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

            // choose two faces by random
            let mut faces = [
                (ps[0], ps[1], ps[2], ps[3]),
                (ps[4], ps[5], ps[6], ps[7]),
                (ps[0], ps[1], ps[4], ps[5]),
                (ps[2], ps[3], ps[6], ps[7]),
                (ps[0], ps[2], ps[4], ps[6]),
                (ps[1], ps[3], ps[5], ps[7]),
            ];

            faces.shuffle(&mut rng);

            let face0 = faces[0];
            let face1 = faces[1];

            let dist_t = Uniform::new_inclusive(0.0, 1.0);

            // sample a random point from the first face
            let p0 = face0.0
                + (face0.1 - face0.0) * dist_t.sample(&mut rng)
                + (face0.2 - face0.0) * dist_t.sample(&mut rng);

            // sample a random point from the second face
            let p1 = face1.0
                + (face1.1 - face1.0) * dist_t.sample(&mut rng)
                + (face1.2 - face1.0) * dist_t.sample(&mut rng);

            // create ray connecting p0 and p1
            let dir = p1 - p0;

            // line  fully passes through the cube
            let t_offset = dist_t.sample(&mut rng);
            let origin = p0 - dir * t_offset;
            let ray = Ray::<f64>::new(origin, dir, 1.0 + 2.0 * t_offset);
            let rayi = Ray::<Interval<f64>>::new(
                Point3::from(origin),
                Vector3::from(dir),
                Interval::from(1.0 + 2.0 * t_offset),
            );

            let intersection = bbox.intersect_ray(ray);
            let intersectioni = bboxi.intersect_ray(rayi);

            assert!(intersection.is_some());
            assert!(intersection.unwrap().0.almost_eq(t_offset));
            assert!(intersection.unwrap().1.almost_eq(1.0 + t_offset));

            assert!(intersectioni.is_some());
            assert!(intersectioni.unwrap().0.almost_eq(t_offset));
            assert!(intersectioni.unwrap().1.almost_eq(1.0 + t_offset));

            // line goes into the cube but doesn't come out
            let ray = Ray::<f64>::new(origin, dir, 1.0);
            let rayi = Ray::<Interval<f64>>::new(
                Point3::from(origin),
                Vector3::from(dir),
                Interval::from(1.0),
            );

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
            let rayi = Ray::<Interval<f64>>::new(
                Point3::from(p0 + dir * t_offset),
                Vector3::from(dir),
                Interval::from(1.0),
            );

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
                Point3::from(p0 + dir * t_offset),
                Vector3::from(p1 - (p0 + dir * t_offset)),
                Interval::from(1.0),
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
            let rayi = Ray::<Interval<f64>>::new(
                Point3::from(p1 + dir * 0.01),
                Vector3::from(dir),
                Interval::from(1.0),
            );

            let intersection = bbox.intersect_ray(ray);
            let intersectioni = bboxi.intersect_ray(rayi);

            assert!(intersection.is_none());
            assert!(intersectioni.is_none());

            // line outside the cube (almost reaching p0)
            let ray = Ray::<f64>::new(p0 - dir * 1.01, dir, 1.0);

            let intersection = bbox.intersect_ray(ray);
            assert!(intersection.is_none());
        }
    }
}
