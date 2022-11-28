use crate::{
    geometry::{
        bbox::Bbox3,
        primitive::{Bounded3, IntersectionResult, IntersectsRay, Primitive, SurfaceInteraction},
        ray::Rayf,
    },
    math::{
        scalar::{Float, GFloat, Scalar, SignedScalar},
        vector::{Point2f, Point3f, Vector3f},
    },
};

/// Spheres are parameterized by the equations
/// x = r * sin(theta) * cos(phi)
/// y = r * sin(theta) * sin(phi)
/// z = r * cos(theta)
/// where phi in [-pi, pi] and theta in [0, pi]
///
/// this means that
/// phi = atan2(y, x)
/// theta = acos(z)
#[derive(Debug, Clone, Copy)]
pub struct Sphere {
    pub radius: Float,
    pub z_min: Float,
    pub z_max: Float,
    pub theta_min: Float,
    pub theta_max: Float,
    pub phi_max: Float,
}

pub struct SphereIntersectionParams {
    pub t_ray: Float,
    // intersection point in object space
    pub p: Point3f,
    pub phi: Float,
}

impl Bounded3 for Sphere {
    fn bbox(&self) -> Bbox3 {
        Bbox3::new(
            Point3f::new(-self.radius, -self.radius, self.z_min),
            Point3f::new(self.radius, self.radius, self.z_max),
        )
    }
}

impl Sphere {
    pub fn new(radius: Float, z_min: Float, z_max: Float, phi_max: Float) -> Self {
        assert!(z_min < z_max);

        let theta_min = Float::clamp(z_min / radius, Float::zero(), Float::one()).acos();
        let theta_max = Float::clamp(z_max / radius, Float::zero(), Float::one()).acos();

        Self {
            radius,
            z_min,
            z_max,
            theta_min,
            theta_max,
            phi_max,
        }
    }
}

// intersection related
impl Sphere {
    fn intersect_ray(&self, ray: Rayf) -> Option<SphereIntersectionParams> {
        // solve quadratic equation a t^2 + b t + c = 0 which is the result of
        // substituting parameteric line equation in the sphere equation
        // with a = d_x^2 + d_y^2 + d_z^2 ~= 1
        //      b = 2(o_x d_x + o_y d_y + o_z d_z)
        //      c = o_x^2 + o_y^2 + o_z^2 - r^2

        let a = ray.direction.square_norm();
        let b = Vector3f::from(ray.origin).dot(ray.direction) * Float::from_i64(2);
        let c = Vector3f::from(ray.origin).square_norm() - self.radius.square();

        let (t0, t1) = solve_quadratic(a, b, c)?;

        let get_intersection = |t: Float| -> Option<SphereIntersectionParams> {
            if !t.in_range(Float::zero(), ray.tmax) {
                return None;
            }

            // calculate hit point
            // should we refine hitpoint by further projecting it on the sphere?
            let hit_point = ray.eval(t);

            if !hit_point.z().in_range(self.z_min, self.z_max) {
                return None;
            }

            let phi = hit_point.y().atan2(hit_point.x());

            if phi > self.phi_max {
                return None;
            }

            Some(SphereIntersectionParams {
                t_ray: t,
                p: hit_point,
                phi,
            })
        };

        match get_intersection(t0) {
            None => get_intersection(t1),
            Some(intersection) => Some(intersection),
        }
    }
    pub fn area(&self) -> Float {
        // FloatODO: derive this formula
        self.phi_max * self.radius * (self.z_max - self.z_min)
    }
}

impl IntersectsRay for Sphere {
    fn do_intersect(&self, ray: Rayf) -> bool {
        self.intersect_ray(ray).is_some()
    }
    fn intersect(&self, ray: Rayf) -> Option<IntersectionResult> {
        let intersection_params = self.intersect_ray(ray)?;

        let t_ray = intersection_params.t_ray;
        let p = intersection_params.p;
        let phi = intersection_params.phi;
        let theta = Float::clamp(p.z() / self.radius, Float::zero(), Float::one()).acos();

        let sin_theta = theta.sin();
        let (sin_phi, cos_phi) = phi.sin_cos();

        // (u, v) in [0, 1]^2 while (phi, theta) in [-pi, pi] x [0, pi]
        // so we transform them
        let phi_max_plus_pi = self.phi_max + Float::pi();
        let theta_max_minus_theta_min = self.theta_max - self.theta_min;

        let u = (phi + Float::pi()) / phi_max_plus_pi;
        let v = (theta - self.theta_min) / theta_max_minus_theta_min;

        // now we calculate dp/du and dp/dv
        // phi   = u * (phi_max + pi) - pi
        // theta = v * theta_max + theta_min
        //
        // taking the derivatives we get

        // dxdu = -(phi_max + pi) * radius * sin(theta) * sin(phi);
        // dydu = (phi_max + pi) * radius * sin(theta) * cos(phi);
        // dzdu = 0

        // dxdv = (theta_max - theta_min) * radius * cos(theta) * cos(phi);
        // dydv = (theta_max - theta_min) * radius * cos(theta) * sin(phi);
        // dzdv = -(theta_max - theta_min) * radius * sin(theta);

        // which is
        let dpdu = Vector3f::new(-p.y(), p.x(), Float::zero()) * phi_max_plus_pi;
        let dpdv = Vector3f::new(p.z() * cos_phi, p.z() * sin_phi, -self.radius * sin_theta)
            * (theta_max_minus_theta_min);

        let surface_interaction =
            SurfaceInteraction::new(intersection_params.p, Point2f::new(u, v), dpdu, dpdv);

        Some(IntersectionResult {
            surface_interaction,
            t_ray,
        })
    }
}

// Should this be here?
fn solve_quadratic(a: Float, b: Float, c: Float) -> Option<(Float, Float)> {
    // discriminant
    let d = b.square() - a * c * Float::from_i64(4);
    if d.is_nonnegative() {
        let sqrt_d = d.sqrt();
        // https://people.csail.mit.edu/bkph/articles/Quadratics.pdf
        if b.is_nonnegative() {
            Some(Float::min_max(
                -(b + sqrt_d) * Float::half() / a,
                -(c + c) / (b + sqrt_d),
            ))
        } else {
            Some(Float::min_max(
                (-b + sqrt_d) * Float::half() / a,
                (c + c) / (-b + sqrt_d),
            ))
        }
    } else {
        None
    }
}

impl Primitive for Sphere {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::scalar::{AlmostEqual, Scalar};
    use rand::{
        distributions::{Distribution, Uniform},
        rngs::SmallRng,
        SeedableRng,
    };

    #[test]
    fn test_ray_full_sphere_intersection() {
        let seed: [u8; 32] = [251; 32];
        let mut rng = SmallRng::from_seed(seed);

        let dist_radius = Uniform::new_inclusive(0.0, 100.0);
        let dist_phi = Uniform::new_inclusive(-Float::pi(), Float::pi());
        let dist_theta = Uniform::new_inclusive(0.0, Float::pi());
        let dist_offset = Uniform::new_inclusive(0.0, 1.0);

        let num_tests = 1000;
        for _ in 0..num_tests {
            // generate random sphere
            let radius = dist_radius.sample(&mut rng);

            let dist_z = Uniform::new_inclusive(0.0, radius);
            let (z_min, z_max) = Float::min_max(dist_z.sample(&mut rng), dist_z.sample(&mut rng));
            let phi_max = dist_phi.sample(&mut rng);

            let sphere = Sphere::new(radius, z_min, z_max, phi_max);

            // generate two random points on the full sphere
            let theta = dist_theta.sample(&mut rng);
            let phi = dist_phi.sample(&mut rng);

            let p0 = Point3f::new(
                radius * theta.sin() * phi.cos(),
                radius * theta.sin() * phi.sin(),
                radius * theta.cos(),
            );
            let p0_on_sphere = p0.z() >= z_min && p0.z() <= z_max && phi <= phi_max;

            let theta = dist_theta.sample(&mut rng);
            let phi = dist_phi.sample(&mut rng);

            let p1 = Point3f::new(
                radius * theta.sin() * phi.cos(),
                radius * theta.sin() * phi.sin(),
                radius * theta.cos(),
            );

            let p1_on_sphere = p1.z() >= z_min && p1.z() <= z_max && phi <= phi_max;

            // outside the sphere, from p0 to p1, only intersects p0
            let dir = p1 - p0;
            let t_offset = dist_offset.sample(&mut rng);
            let origin = p0 - dir * t_offset;

            let ray = Rayf::new(origin, dir, 1.0);

            let intersection = sphere.intersect(ray);

            if p0_on_sphere {
                assert!(intersection.is_some());
                assert!(intersection.unwrap().t_ray.almost_eq(t_offset));
            } else {
                assert!(intersection.is_none());
            }

            // outside the sphere, from p0 to p1, intersects both p0 and p1
            let t_offset = dist_offset.sample(&mut rng);
            let origin = p0 - dir * t_offset;
            let ray = Rayf::new(origin, dir, 1.0 + 2.0 * t_offset);

            let intersection = sphere.intersect(ray);

            if p0_on_sphere || p1_on_sphere {
                assert!(intersection.is_some());

                if p0_on_sphere {
                    assert!(intersection.unwrap().t_ray.almost_eq(t_offset));
                } else {
                    assert!(intersection.unwrap().t_ray.almost_eq(1.0 + t_offset));
                }
            } else {
                assert!(intersection.is_none());
            }

            // inside the sphere, from p0 to p1, only intersects p1
            let t_offset = dist_offset.sample(&mut rng);
            let origin = p0 + dir * t_offset;
            let ray = Rayf::new(origin, dir, 1.0);

            let intersection = sphere.intersect(ray);

            if p1_on_sphere {
                assert!(intersection.is_some());
                assert!(intersection.unwrap().t_ray.almost_eq(1.0 - t_offset));
            } else {
                assert!(intersection.is_none());
            }

            // inside the sphere, from p0 to p1, intersects nothing
            let t_offset = dist_offset.sample(&mut rng);
            let origin = p0 + dir * t_offset;
            let ray = Rayf::new(origin, dir, 1.0 - t_offset - 0.1);

            let intersection = sphere.intersect(ray);

            assert!(intersection.is_none());
        }
    }
}
