use crate::{
    geometry::ray::Rayf,
    math::{
        scalar::{Float, GFloat, Scalar},
        vector::Point2f,
    },
};

pub struct CameraSample {
    // p_film in [0, 1] x [0, 1]
    pub p_film: Point2f,
    // p_lens in [0, 1]^2
    pub p_lens: Point2f,
}

pub trait Camera {
    fn generate_ray(&self, sample: CameraSample) -> Rayf;
}

impl CameraSample {
    pub fn new(p_film: Point2f, p_lens: Point2f) -> Self {
        Self { p_film, p_lens }
    }
}

/// Maps the unit square to the unit disk. Expects `p` to be in `[0, 1] x [0, 1]`.
///
/// The naive way is to directly map to polar coordinates with `r = x` and
/// `theta = y * 2 * pi`, but this will result in clumping near the center, and
/// more sparse points as r increases.
///
/// Ideally we would want a uniform distribution: `f(x, y) = 1/area = 1/pi`.
/// Now, the change of variables technique tells us that if we have `y = g(x)`,
/// where g is monotonically increasing, then
/// ``` text
/// fy(y) = fx(g^-1(x)) / |det J|
/// ```
/// where J is the jacobian. We have `r = sqrt(x^2 + y^2)` and
/// `theta = atan(y/x)`, which implies that
/// ```text
///  dr/dx      = x  / sqrt(x^2 + y^2)
///  dr/dy      = y  / sqrt(x^2 + y^2)
///  dtheta/dx  = -y / (x^2+y^2)
///  dtheta/dy  = x  / (x^2+y^2)
///
///  |det J|    = 1 / (x^2+y^2)
///             = 1 / r
/// ```
/// and therefore
/// ```text
///  f(r, theta) = r / pi
/// ```
/// Let's assume that `r` and `theta` are independent variables, then
/// ```text
///  f(r, theta) = f(r) * f(theta)
/// ```
/// which would imply that `f(r) = 2*r` and `f(theta) = 1/(2*pi)`. Now, let
/// `y` be a uniform random variable in the range `[0, 1]`, so that `f(y) = 1`
/// and such that `r = sqrt(y)`. Then using the change of variables technique
/// again we get:
/// ```text
///  f(r) = f(y) / (dr/dy)
///       = 1 / (1/(2*sqrt(y)))
///       = 2 * sqrt(y)
///       = 2 * r
/// ```
/// From this we can conclude that we can uniformly sample `(x,y) in [0,1]^2`,
/// and then putting `r = sqrt(y)` and `theta = 2 * pi * x` would uniformly
/// sample the unit disk.
pub fn uniform_sample_disk(p: Point2f) -> Point2f {
    let r = p.y().sqrt();
    let theta = Float::from_f64(2.0) * Float::pi() * p.x();
    let (s, c) = theta.sin_cos();
    Point2f::new(r * c, r * s)
}

/// Maps the unit square to the unit disk by mapping concentric squares to
/// concentric disks. Expects `p` to be in `[0, 1] x [0, 1]`.
///
/// This is done by dividing the the unit square into four quadrants resulting
/// from connecting the diagonals.
/// ```text
///  -------------
///  |\    2    /|
///  |  \     /  |
///  | 3  \ /  1 |
///  |    / \    |
///  |  /     \  |
///  |/    4    \|
///  -------------
/// ```
/// For the first and third quadrants, the radius is chosen to be the `x`
/// coordinate of the point, which would correspond to the length of the
/// concentric square that this point lies on, and theta is chosen to lie
/// in the same quadrant and proprtional to `y/x`, which translates to
/// `theta = (y/x) * pi/4` for the first quadrant and `theta = pi + (y/x) * pi/4`
/// for the third quadrant. The second and fourth quadrants are treated similarly.
///
/// This method is usually preferred to `uniform_sample_disk` because it causes
/// lower distortion.
pub fn concentric_sample_disk(p: Point2f) -> Point2f {
    let x = Float::one() - Float::from_f32(2.0) * p.x();
    let y = Float::one() - Float::from_f32(2.0) * p.y();

    if x.is_zero() && y.is_zero() {
        return Point2f::new(Float::zero(), Float::zero());
    }

    let (r, theta) = if x.abs() > y.abs() {
        (x, Float::frac_pi_4() * y / x)
    } else {
        (y, Float::frac_pi_2() - (x / y) * Float::frac_pi_4())
    };

    Point2f::new(r * theta.cos(), r * theta.sin())
}
