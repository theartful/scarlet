use crate::{
    cameras::camera::{concentric_sample_disk, Camera, CameraSample},
    geometry::ray::Rayf,
    math::{
        scalar::{Float, GFloat, Scalar},
        transform::Transformf,
        vector::Point3f,
    },
};

/// A camera based on the thin-lens model. A thin lens is a lens which has
/// no thickness, and which forms an image based on the following rules:
/// 1. A ray passing  through the center of the lens will not be refracted.
/// 2. A ray passing parallel to the axis of the lens on one side will be
///    refracted to pass through the focal point on the other side.
/// 3. A ray passing through the focal point on one side will be refracted
///    parallel to the axis of the lens on the other side.
/// 4. All rays originating from a single point on one side, will meet at one
///    point on the other side.
///
/// Based on these rules, it is easy to prove the gaussian lens equation:
/// ```text
/// 1/f = 1/o + 1/i
/// ```
/// where `f` is the focal length, `o` is the distance from the object to the
/// plane on which the lens lies on (object distance), and `i` is the distance
/// from the image of the object to the plane on which the lens lies on (image
/// distance).
///
/// So if we want to make the camera focus on an object `o` distance away, we
/// would need to put the film `i` distance away on the other side of the lens.
pub struct ThinLensCamera {
    view: Transformf,
    // the distance between the lens and plane of focus
    focal_distance: Float,
    // radius of the lens
    lens_radius: Float,
    // the distance between the lens and the film such that objects at
    // z=-focal_distance are in focus
    film_distance: Float,
    // the width of the film
    film_width: Float,
    // the height of the film
    film_height: Float,
}

impl ThinLensCamera {
    pub fn new(
        view: Transformf,
        focal_distance: Float,
        fov: Float,
        focal_length: Float,
        lens_radius: Float,
        aspect_ratio: Float,
    ) -> Self {
        let film_distance = focal_distance * focal_length / (focal_distance - focal_length);

        // field of view h
        let film_width = Float::half() * (fov * Float::half()).tan() * film_distance;
        let film_height = film_width / aspect_ratio;

        Self {
            view,
            focal_distance,
            lens_radius,
            film_distance,
            film_width,
            film_height,
        }
    }
}

impl Camera for ThinLensCamera {
    fn generate_ray(&self, sample: CameraSample) -> Rayf {
        // let's first compute p_lens
        let p_lens_xy = concentric_sample_disk(sample.p_lens) * self.lens_radius;
        let p_lens = Point3f::new(p_lens_xy.x(), p_lens_xy.y(), Float::zero());

        // now let's compute p_film
        let p_film = Point3f::new(
            // the image on the other side of the lens is flipped, so we have to
            // flip it again
            Float::half() * self.film_width - sample.p_film.x() * self.film_width,
            Float::half() * self.film_height - sample.p_film.y() * self.film_height,
            self.film_distance,
        );

        // the ray passing from p_film to p_lens will refract and intersect with
        // the ray passing from p_film to the center of the lens at the focal
        // plane
        let film_to_lens = Rayf::new(
            p_film,
            Point3f::new(Float::zero(), Float::zero(), Float::zero()) - p_film,
            Float::zero(),
        );

        // find t at z = -focal_distance
        let t = -self.focal_distance / film_to_lens.direction.z();
        let p_at_focal_plane = film_to_lens.eval(t);

        // now the result is a ray passing from p_lens to p_at_focal_plane
        self.view.map_inverse(Rayf::new(
            p_lens,
            (p_at_focal_plane - p_lens).normalize(),
            Float::highest(),
        ))
    }
}
