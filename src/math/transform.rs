use crate::math::scalar::{AlmostEqual, Float, GFloat, Int, Scalar};
use crate::math::transform_inverse_impl::inverse_impl;
use crate::math::vector::{InnerScalar, Norm, Normal3, Point3, Vector3, Vector4};
use std::ops::{Add, Div, Index, IndexMut, Mul};

pub type Matrix4x4f = Matrix4x4<Float>;
pub type Matrix4x4i = Matrix4x4<Int>;
pub type Transformf = Transform<Float>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Matrix4x4<T: Scalar> {
    pub data: [[T; 4]; 4],
}

impl<T: Scalar> Default for Matrix4x4<T> {
    fn default() -> Self {
        Self::identity()
    }
}

impl<T: Scalar> Matrix4x4<T> {
    pub fn new() -> Self {
        Self::identity()
    }

    pub fn identity() -> Self {
        Matrix4x4 {
            data: [
                [T::one(), T::zero(), T::zero(), T::zero()],
                [T::zero(), T::one(), T::zero(), T::zero()],
                [T::zero(), T::zero(), T::one(), T::zero()],
                [T::zero(), T::zero(), T::zero(), T::one()],
            ],
        }
    }

    pub fn zero() -> Self {
        Matrix4x4 {
            data: [
                [T::zero(), T::zero(), T::zero(), T::zero()],
                [T::zero(), T::zero(), T::zero(), T::zero()],
                [T::zero(), T::zero(), T::zero(), T::zero()],
                [T::zero(), T::zero(), T::zero(), T::zero()],
            ],
        }
    }

    pub fn transpose(&self) -> Matrix4x4<T> {
        let d = &self.data;
        Matrix4x4 {
            data: [
                [d[0][0], d[1][0], d[2][0], d[3][0]],
                [d[0][1], d[1][1], d[2][1], d[3][1]],
                [d[0][2], d[1][2], d[2][2], d[3][2]],
                [d[0][3], d[1][3], d[2][3], d[3][3]],
            ],
        }
    }

    pub fn row(&self, r: usize) -> [T; 4] {
        self.data[r]
    }

    pub fn col(&self, c: usize) -> [T; 4] {
        [
            self.data[0][c],
            self.data[1][c],
            self.data[2][c],
            self.data[3][c],
        ]
    }
    pub fn row_vec(&self, r: usize) -> Vector4<T> {
        Vector4::new(
            self.data[r][0],
            self.data[r][1],
            self.data[r][2],
            self.data[r][3],
        )
    }

    pub fn col_vec(&self, c: usize) -> Vector4<T> {
        Vector4::new(
            self.data[0][c],
            self.data[1][c],
            self.data[2][c],
            self.data[3][c],
        )
    }
}

impl<T: Scalar + AlmostEqual> AlmostEqual for Matrix4x4<T> {
    fn almost_eq(self, other: Self) -> bool {
        for i in 0..4 {
            for j in 0..4 {
                if !self.data[i][j].almost_eq(other.data[i][j]) {
                    return false;
                }
            }
        }
        true
    }
}

impl<T: Scalar> std::convert::From<[[T; 4]; 4]> for Matrix4x4<T> {
    fn from(data: [[T; 4]; 4]) -> Self {
        Matrix4x4 { data }
    }
}

impl<T: Scalar> Index<(usize, usize)> for Matrix4x4<T> {
    type Output = T;

    #[inline]
    fn index(&self, idx: (usize, usize)) -> &Self::Output {
        &self.data[idx.0][idx.1]
    }
}

impl<T: Scalar> Index<usize> for Matrix4x4<T> {
    type Output = [T; 4];

    fn index(&self, idx: usize) -> &Self::Output {
        &self.data[idx]
    }
}

impl<T: Scalar> IndexMut<(usize, usize)> for Matrix4x4<T> {
    fn index_mut(&mut self, idx: (usize, usize)) -> &mut Self::Output {
        &mut self.data[idx.0][idx.1]
    }
}

impl<T: Scalar> IndexMut<usize> for Matrix4x4<T> {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.data[idx]
    }
}

impl<T: Scalar> Mul<T> for Matrix4x4<T> {
    type Output = Matrix4x4<T>;
    fn mul(self, rhs: T) -> Self::Output {
        let d = self.data;
        Matrix4x4 {
            data: [
                [rhs * d[0][0], rhs * d[0][1], rhs * d[0][2], rhs * d[0][3]],
                [rhs * d[1][0], rhs * d[1][1], rhs * d[1][2], rhs * d[1][3]],
                [rhs * d[2][0], rhs * d[2][1], rhs * d[2][2], rhs * d[2][3]],
                [rhs * d[3][0], rhs * d[3][1], rhs * d[3][2], rhs * d[3][3]],
            ],
        }
    }
}

impl<T: Scalar> Mul<Matrix4x4<T>> for Matrix4x4<T> {
    type Output = Matrix4x4<T>;
    fn mul(self, rhs: Matrix4x4<T>) -> Self::Output {
        let mut res = Self::zero();
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    res.data[i][j] += self.data[i][k] * rhs.data[k][j];
                }
            }
        }
        res
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform<T: GFloat> {
    m: Matrix4x4<T>,
    minv: Matrix4x4<T>,
}

impl<T: GFloat> Transform<T> {
    pub fn new(m: Matrix4x4<T>, minv: Matrix4x4<T>) -> Self {
        Transform { m, minv }
    }

    pub fn identity() -> Self {
        Transform {
            m: Matrix4x4::<T>::identity(),
            minv: Matrix4x4::<T>::identity(),
        }
    }

    fn init_from_orthogonal_basis(
        camera_pos: Point3<T>,
        x: Vector3<T>,
        y: Vector3<T>,
        z: Vector3<T>,
    ) -> Self {
        Transform {
            m: Matrix4x4 {
                data: [
                    [x.x(), x.y(), x.z(), -x.dot(camera_pos)],
                    [y.x(), y.y(), y.z(), -y.dot(camera_pos)],
                    [z.x(), z.y(), z.z(), -z.dot(camera_pos)],
                    [T::zero(), T::zero(), T::zero(), T::one()],
                ],
            },
            minv: Matrix4x4 {
                data: [
                    [x.x(), y.x(), z.x(), camera_pos.x()],
                    [x.y(), y.y(), z.y(), camera_pos.y()],
                    [x.z(), y.z(), z.z(), camera_pos.z()],
                    [T::zero(), T::zero(), T::zero(), T::one()],
                ],
            },
        }
    }

    pub fn translate(vec: Vector3<T>) -> Self {
        Self::new(
            Matrix4x4 {
                data: [
                    [T::one(), T::zero(), T::zero(), vec.x()],
                    [T::zero(), T::one(), T::zero(), vec.y()],
                    [T::zero(), T::zero(), T::one(), vec.z()],
                    [T::zero(), T::zero(), T::zero(), T::one()],
                ],
            },
            Matrix4x4 {
                data: [
                    [T::one(), T::zero(), T::zero(), -vec.x()],
                    [T::zero(), T::one(), T::zero(), -vec.y()],
                    [T::zero(), T::zero(), T::one(), -vec.z()],
                    [T::zero(), T::zero(), T::zero(), T::one()],
                ],
            },
        )
    }

    pub fn scale(vec: Vector3<T>) -> Self {
        Self::new(
            Matrix4x4 {
                data: [
                    [vec.x(), T::zero(), T::zero(), T::zero()],
                    [T::zero(), vec.y(), T::zero(), T::zero()],
                    [T::zero(), T::zero(), vec.z(), T::zero()],
                    [T::zero(), T::zero(), T::zero(), T::one()],
                ],
            },
            Matrix4x4 {
                data: [
                    [T::one() / vec.x(), T::zero(), T::zero(), T::zero()],
                    [T::zero(), T::one() / vec.y(), T::zero(), T::zero()],
                    [T::zero(), T::zero(), T::one() / vec.z(), T::zero()],
                    [T::zero(), T::zero(), T::zero(), T::one()],
                ],
            },
        )
    }

    /// Constructs a transformation matrix that rotates a vector around the x
    /// axis by a certain angle
    pub fn rotate_x(angle: T) -> Self {
        let (s, c) = angle.sin_cos();
        Self::new(
            Matrix4x4 {
                data: [
                    [T::one(), T::zero(), T::zero(), T::zero()],
                    [T::zero(), c, -s, T::zero()],
                    [T::zero(), s, c, T::zero()],
                    [T::zero(), T::zero(), T::zero(), T::one()],
                ],
            },
            Matrix4x4 {
                data: [
                    [T::one(), T::zero(), T::zero(), T::zero()],
                    [T::zero(), c, s, T::zero()],
                    [T::zero(), -s, c, T::zero()],
                    [T::zero(), T::zero(), T::zero(), T::one()],
                ],
            },
        )
    }

    /// Constructs a transformation matrix that rotates a vector around the y
    /// axis by a certain angle
    pub fn rotate_y(angle: T) -> Self {
        let (s, c) = angle.sin_cos();
        Self::new(
            Matrix4x4 {
                data: [
                    [c, T::zero(), s, T::zero()],
                    [T::zero(), T::one(), T::zero(), T::zero()],
                    [-s, T::zero(), c, T::zero()],
                    [T::zero(), T::zero(), T::zero(), T::one()],
                ],
            },
            Matrix4x4 {
                data: [
                    [c, T::zero(), -s, T::zero()],
                    [T::zero(), T::one(), T::zero(), T::zero()],
                    [s, T::zero(), c, T::zero()],
                    [T::zero(), T::zero(), T::zero(), T::one()],
                ],
            },
        )
    }

    /// Constructs a transformation matrix that rotates a vector around the z
    /// axis by a certain angle
    pub fn rotate_z(angle: T) -> Self {
        let (s, c) = angle.sin_cos();
        Self::new(
            Matrix4x4 {
                data: [
                    [c, -s, T::zero(), T::zero()],
                    [s, c, T::zero(), T::zero()],
                    [T::zero(), T::zero(), T::one(), T::zero()],
                    [T::zero(), T::zero(), T::zero(), T::one()],
                ],
            },
            Matrix4x4 {
                data: [
                    [c, s, T::zero(), T::zero()],
                    [-s, c, T::zero(), T::zero()],
                    [T::zero(), T::zero(), T::one(), T::zero()],
                    [T::zero(), T::zero(), T::zero(), T::one()],
                ],
            },
        )
    }

    /// Constructs a transformation matrix that rotates a vector around an axis
    /// of rotation by a certain angle
    pub fn rotate(vec: Vector3<T>, angle: T) -> Self
    where
        Vector3<T>: Norm + InnerScalar<ScalarType = T>,
    {
        // Rodrigues' rotation formula
        let u = vec.normalize();

        let x = Vector3::new(T::one(), T::zero(), T::zero());
        let y = Vector3::new(T::zero(), T::one(), T::zero());
        let z = Vector3::new(T::zero(), T::zero(), T::one());

        let (s, c) = angle.sin_cos();

        let x_rotated = x * c + u * (T::one() - c) * (u.dot(x)) + u.cross(x) * s;
        let y_rotated = y * c + u * (T::one() - c) * (u.dot(y)) + u.cross(y) * s;
        let z_rotated = z * c + u * (T::one() - c) * (u.dot(z)) + u.cross(z) * s;

        let mat = Matrix4x4 {
            data: [
                [x_rotated.x(), y_rotated.x(), z_rotated.x(), T::zero()],
                [x_rotated.y(), y_rotated.y(), z_rotated.y(), T::zero()],
                [x_rotated.z(), y_rotated.z(), z_rotated.z(), T::zero()],
                [T::zero(), T::zero(), T::zero(), T::one()],
            ],
        };

        Self::new(mat, mat.transpose())
    }

    /// Constructs a transformation matrix such that:
    ///
    /// 1. The origin is mapped to `pos`
    /// 2. `target` lies on the negative direction of the z axis
    /// 3. The y axis is the result of the projection of `up` onto the plane
    ///    orthogonal to the construted z axis.
    ///
    /// Note that `up` should not be parallel to `target - pos`
    pub fn look_at(pos: Point3<T>, target: Point3<T>, up: Vector3<T>) -> Self
    where
        Vector3<T>: Norm + InnerScalar<ScalarType = T>,
    {
        let up_u = up.normalize();
        let z_u = (pos - target).normalize();
        let x_u = up_u.cross(z_u).normalize();
        let y_u = z_u.cross(x_u);
        Self::init_from_orthogonal_basis(pos, x_u, y_u, z_u)
    }

    /// ```text
    ///     original space                     mapped space
    /// z=0   z=-near     z=-far        z=0                   z=1
    ///                                   _____________________
    ///                 /  |             |                     |
    ///              /     |             |                     |
    ///        *  |        |             |                     |
    ///     *     |        |             |                     |
    ///  *   )fov |        |       =>    |                     |
    ///     *     |        |             |                     |
    ///        *  |        |             |                     |
    ///              \     |             |                     |
    ///                 \  |             |_____________________|
    ///
    /// ```
    ///
    /// Constructs a perspective projection matrix that maps a symmetrical view
    /// frustrum between the two clipping planes defined by `z = -near` and
    /// `z = -far` and whose field of view is `fov` into the cube
    /// `(x, y, z) \in [-1, 1] * [-1, 1] * [0, 1]`
    pub fn perspective(fov: T, near: T, far: T) -> Self {
        let e = (fov * T::half()).tan().recip();
        let mat = Matrix4x4 {
            data: [
                [-e, T::zero(), T::zero(), T::zero()],
                [T::zero(), -e, T::zero(), T::zero()],
                [
                    T::zero(),
                    T::zero(),
                    far / (far - near),
                    far * near / (far - near),
                ],
                [T::zero(), T::zero(), T::one(), T::zero()],
            ],
        };
        Self::new(mat, mat.inverse())
    }

    pub fn matrix(&self) -> Matrix4x4<T> {
        self.m
    }

    pub fn inverse_matrix(&self) -> Matrix4x4<T> {
        self.minv
    }

    #[inline]
    pub fn map<I, U>(&self, obj: I) -> U
    where
        Self: TransformMap<I, Output = U>,
    {
        <Self as TransformMap<I>>::map(self, obj)
    }

    #[inline]
    pub fn map_inverse<I, U>(&self, obj: I) -> U
    where
        Self: TransformMap<I, Output = U>,
    {
        <Self as TransformMap<I>>::map_inverse(self, obj)
    }
}

impl<T: GFloat + AlmostEqual> AlmostEqual for Transform<T> {
    fn almost_eq(self, other: Self) -> bool {
        self.m.almost_eq(other.m) && self.minv.almost_eq(other.minv)
    }
}

impl<T: GFloat> Mul<Transform<T>> for Transform<T> {
    type Output = Transform<T>;

    fn mul(self, rhs: Transform<T>) -> Self::Output {
        Self::new(self.m * rhs.m, rhs.minv * self.minv)
    }
}

pub trait Invertible {
    fn inverse(&self) -> Self;
}

impl<T: GFloat> Invertible for Matrix4x4<T> {
    fn inverse(&self) -> Self {
        Self::from(inverse_impl(&self.data))
    }
}

impl<T: GFloat> Invertible for Transform<T> {
    fn inverse(&self) -> Self {
        Self::new(self.minv, self.m)
    }
}

pub trait TransformMap<T> {
    type Output;

    fn map(&self, obj: T) -> Self::Output;

    fn map_inverse(&self, obj: T) -> Self::Output
    where
        Self: Invertible + Sized,
    {
        self.inverse().map(obj)
    }
}

impl<T, U> TransformMap<Vector3<U>> for Transform<T>
where
    T: GFloat,
    U: Scalar + Mul<T, Output = U>,
{
    type Output = Vector3<U>;

    fn map(&self, vec: Vector3<U>) -> Self::Output {
        map_vector(&self.m, &vec)
    }

    fn map_inverse(&self, vec: Vector3<U>) -> Self::Output {
        map_vector(&self.minv, &vec)
    }
}

fn map_vector<T, U>(m: &Matrix4x4<T>, vec: &Vector3<U>) -> Vector3<U>
where
    T: Scalar,
    U: Scalar + Mul<T, Output = U>,
{
    assert_eq!(
        vec.x() * m[(3, 0)] + vec.y() * m[(3, 1)] + vec.z() * m[(3, 2)],
        U::zero()
    );
    Vector3::new(
        vec.x() * m[(0, 0)] + vec.y() * m[(0, 1)] + vec.z() * m[(0, 2)],
        vec.x() * m[(1, 0)] + vec.y() * m[(1, 1)] + vec.z() * m[(1, 2)],
        vec.x() * m[(2, 0)] + vec.y() * m[(2, 1)] + vec.z() * m[(2, 2)],
    )
}

impl<T, U> TransformMap<Point3<U>> for Transform<T>
where
    T: GFloat,
    U: Scalar + Mul<T, Output = U> + Div<T, Output = U> + Add<T, Output = U>,
{
    type Output = Point3<U>;

    fn map(&self, p: Point3<U>) -> Self::Output {
        map_point(&self.m, &p)
    }

    fn map_inverse(&self, p: Point3<U>) -> Self::Output {
        map_point(&self.minv, &p)
    }
}

fn map_point<T, U>(m: &Matrix4x4<T>, p: &Point3<U>) -> Point3<U>
where
    T: GFloat,
    U: Scalar + Mul<T, Output = U> + Div<T, Output = U> + Add<T, Output = U>,
{
    let w = p.x() * m[(3, 0)] + p.y() * m[(3, 1)] + p.z() * m[(3, 2)] + m[(3, 3)];
    assert!(w != U::zero());
    Point3::new(
        (p.x() * m[(0, 0)] + p.y() * m[(0, 1)] + p.z() * m[(0, 2)] + m[(0, 3)]) / w,
        (p.x() * m[(1, 0)] + p.y() * m[(1, 1)] + p.z() * m[(1, 2)] + m[(1, 3)]) / w,
        (p.x() * m[(2, 0)] + p.y() * m[(2, 1)] + p.z() * m[(2, 2)] + m[(2, 3)]) / w,
    )
}

impl<T, U> TransformMap<Normal3<U>> for Transform<T>
where
    T: GFloat,
    U: Scalar + Mul<T, Output = U>,
{
    type Output = Normal3<U>;

    fn map(&self, n: Normal3<U>) -> Self::Output {
        map_normal(&self.minv, &n)
    }

    fn map_inverse(&self, n: Normal3<U>) -> Self::Output {
        map_normal(&self.m, &n)
    }
}

fn map_normal<T, U>(minv: &Matrix4x4<T>, n: &Normal3<U>) -> Normal3<U>
where
    T: GFloat,
    U: Scalar + Mul<T, Output = U>,
{
    // normals are mapped by the inverse transpose of the transformation matrix
    Normal3::new(
        n.x() * minv[(0, 0)] + n.y() * minv[(1, 0)] + n.z() * minv[(2, 0)] * n.z(),
        n.x() * minv[(0, 1)] + n.y() * minv[(1, 1)] + n.z() * minv[(2, 1)] * n.z(),
        n.x() * minv[(0, 2)] + n.y() * minv[(1, 2)] + n.z() * minv[(2, 2)] * n.z(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::scalar::AlmostEqual;
    use rand::distributions::{Distribution, Uniform};
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_matrix_transpose() {
        let m = Matrix4x4::<f64>::from([
            [1.1, 1.2, 1.3, 1.4],
            [2.1, 2.2, 2.3, 2.4],
            [3.1, 3.2, 3.3, 3.4],
            [4.1, 4.2, 4.3, 4.4],
        ]);

        assert_eq!(
            m.transpose(),
            Matrix4x4::<f64>::from([
                [1.1, 2.1, 3.1, 4.1],
                [1.2, 2.2, 3.2, 4.2],
                [1.3, 2.3, 3.3, 4.3],
                [1.4, 2.4, 3.4, 4.4]
            ])
        );
        assert_eq!(m.transpose().transpose(), m);
    }

    #[test]
    fn test_matrix_mul_scalar() {
        let m = Matrix4x4::<f64>::from([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]);

        assert_eq!(
            m * 2.0,
            Matrix4x4::<f64>::from([
                [2.0, 4.0, 6.0, 8.0],
                [10.0, 12.0, 14.0, 16.0],
                [18.0, 20.0, 22.0, 24.0],
                [26.0, 28.0, 30.0, 32.0],
            ])
        );
    }

    #[test]
    fn test_matrix_mul() {
        let a =
            Matrix4x4::<i32>::from([[7, 14, 15, 6], [4, 8, 12, 3], [14, 21, 6, 9], [13, 7, 6, 4]]);
        let b = Matrix4x4::<i32>::from([[5, 7, 14, 2], [8, 16, 4, 9], [13, 6, 8, 4], [6, 3, 2, 4]]);

        let ab = Matrix4x4::<i32>::from([
            [378, 381, 286, 224],
            [258, 237, 190, 140],
            [370, 497, 346, 277],
            [223, 251, 266, 129],
        ]);

        assert_eq!(a * b, ab);
    }

    #[test]
    fn test_rotate_on_axis() {
        let angle = std::f64::consts::FRAC_PI_2;

        let x = Vector3::new(1.0, 0.0, 0.0);
        let y = Vector3::new(0.0, 1.0, 0.0);
        let z = Vector3::new(0.0, 0.0, 1.0);

        let transform_x = Transform::<f64>::rotate_x(angle);
        let transform_y = Transform::<f64>::rotate_y(angle);
        let transform_z = Transform::<f64>::rotate_z(angle);

        assert_eq!(
            transform_x,
            Transform::<f64>::rotate(Vector3::<f64>::new(1.0, 0.0, 0.0), angle)
        );
        assert_eq!(
            transform_y,
            Transform::<f64>::rotate(Vector3::<f64>::new(0.0, 1.0, 0.0), angle)
        );
        assert_eq!(
            transform_z,
            Transform::<f64>::rotate(Vector3::<f64>::new(0.0, 0.0, 1.0), angle)
        );
        assert!(transform_x.map(y).almost_eq(z));
        assert!(transform_x.map(z).almost_eq(-y));
        assert!(transform_y.map(x).almost_eq(-z));
        assert!(transform_y.map(z).almost_eq(x));
        assert!(transform_z.map(x).almost_eq(y));
        assert!(transform_z.map(y).almost_eq(-x));
    }

    #[test]
    fn test_rotate_xyz_general() {
        let seed: [u8; 32] = [124; 32];
        let mut rng = SmallRng::from_seed(seed);

        let num_tests = 20;
        for _ in 0..num_tests {
            let dist_angle = Uniform::new_inclusive(-std::f64::consts::PI, std::f64::consts::PI);
            let dist_vec = Uniform::new_inclusive(-100.0, 100.0);
            let angle: f64 = dist_angle.sample(&mut rng);

            let vec = Vector3::<f64>::new(
                dist_vec.sample(&mut rng),
                dist_vec.sample(&mut rng),
                dist_vec.sample(&mut rng),
            );

            let transform_x = Transform::<f64>::rotate_x(angle);
            let transform_y = Transform::<f64>::rotate_y(angle);
            let transform_z = Transform::<f64>::rotate_z(angle);

            assert!(transform_x.almost_eq(Transform::<f64>::rotate(
                Vector3::<f64>::new(1.0, 0.0, 0.0),
                angle
            )));
            assert!(transform_y.almost_eq(Transform::<f64>::rotate(
                Vector3::<f64>::new(0.0, 1.0, 0.0),
                angle
            )));
            assert!(transform_z.almost_eq(Transform::<f64>::rotate(
                Vector3::<f64>::new(0.0, 0.0, 1.0),
                angle
            )));

            let rotated_x = transform_x.map(vec);
            let rotated_y = transform_y.map(vec);
            let rotated_z = transform_z.map(vec);

            assert_eq!(
                rotated_x[0], vec[0],
                "Expected rotating a vector around the x axis not to change \
                the x component!"
            );
            assert_eq!(
                rotated_y[1], vec[1],
                "Expected rotating a vector around the y axis not to change \
                the y component!"
            );
            assert_eq!(
                rotated_z[2], vec[2],
                "Expected rotating a vector around the z axis not to change \
                the z component!"
            );

            let cos_angle_x = Vector3::new(0.0, vec[1], vec[2])
                .normalize()
                .dot(Vector3::new(0.0, rotated_x[1], rotated_x[2]).normalize());

            let sin_angle_x = Vector3::new(0.0, vec[1], vec[2])
                .normalize()
                .cross(Vector3::new(0.0, rotated_x[1], rotated_x[2]).normalize())
                .dot(Vector3::new(1.0, 0.0, 0.0));

            let angle_x = sin_angle_x.atan2(cos_angle_x);
            assert!(
                angle_x.almost_eq(angle),
                "Expected that the angle between the original vector and rotated \
            vector after being projected on the plane perpendicular to the axis \
            of rotation to be equal to the rotation angle! Rotation angle = {} \
            while estimated angle = {}",
                angle,
                angle_x
            );

            let cos_angle_y = Vector3::new(vec[0], 0.0, vec[2])
                .normalize()
                .dot(Vector3::new(rotated_y[0], 0.0, rotated_y[2]).normalize());

            let sin_angle_y = Vector3::new(vec[0], 0.0, vec[2])
                .normalize()
                .cross(Vector3::new(rotated_y[0], 0.0, rotated_y[2]).normalize())
                .dot(Vector3::new(0.0, 1.0, 0.0));
            let angle_y = sin_angle_y.atan2(cos_angle_y);
            assert!(
                angle_y.almost_eq(angle),
                "Expected that the angle between the original vector and rotated \
            vector after being projected on the plane perpendicular to the axis \
            of rotation to be equal to the rotation angle! Rotation angle = {} \
            while estimated angle = {}",
                angle,
                angle_y
            );

            assert!(sin_angle_y.atan2(cos_angle_y).almost_eq(angle));

            let cos_angle_z = Vector3::new(vec[0], vec[1], 0.0)
                .normalize()
                .dot(Vector3::new(rotated_z[0], rotated_z[1], 0.0).normalize());

            let sin_angle_z = Vector3::new(vec[0], vec[1], 0.0)
                .normalize()
                .cross(Vector3::new(rotated_z[0], rotated_z[1], 0.0).normalize())
                .dot(Vector3::new(0.0, 0.0, 1.0));

            let angle_z = sin_angle_z.atan2(cos_angle_z);
            assert!(
                angle_z.almost_eq(angle),
                "Expected that the angle between the original vector and rotated \
            vector after being projected on the plane perpendicular to the axis \
            of rotation to be equal to the rotation angle! Rotation angle = {} \
            while estimated angle = {}",
                angle,
                angle_z
            );
        }
    }

    #[test]
    fn test_rotate_general() {
        let seed: [u8; 32] = [87; 32];
        let mut rng = SmallRng::from_seed(seed);

        let num_tests = 20;
        for _ in 0..num_tests {
            let dist_angle = Uniform::new_inclusive(-std::f64::consts::PI, std::f64::consts::PI);
            let dist_vec = Uniform::new_inclusive(-100.0, 100.0);
            let angle: f64 = dist_angle.sample(&mut rng);
            let axis_of_rotation = Vector3::<f64>::new(
                dist_vec.sample(&mut rng),
                dist_vec.sample(&mut rng),
                dist_vec.sample(&mut rng),
            )
            .normalize();

            let vec = Vector3::<f64>::new(
                dist_vec.sample(&mut rng),
                dist_vec.sample(&mut rng),
                dist_vec.sample(&mut rng),
            );

            let transform = Transform::<f64>::rotate(axis_of_rotation, angle);

            let rotated_vec = transform.map(vec);

            assert!(vec
                .dot(axis_of_rotation)
                .almost_eq(rotated_vec.dot(axis_of_rotation)));

            let remove_rotation_axis_component =
                |v: Vector3<f64>| v - axis_of_rotation * v.dot(axis_of_rotation);

            let cos_angle = remove_rotation_axis_component(vec)
                .normalize()
                .dot(remove_rotation_axis_component(rotated_vec).normalize());

            let sin_angle = remove_rotation_axis_component(vec)
                .normalize()
                .cross(remove_rotation_axis_component(rotated_vec).normalize())
                .dot(axis_of_rotation);

            let estimated_angle = sin_angle.atan2(cos_angle);
            assert!(
                estimated_angle.almost_eq(angle),
                "Expected that the angle between the original vector and rotated \
                        vector after being projected on the plane perpendicular to the axis \
                        of rotation to be equal to the rotation angle! Rotation angle = {} \
                        while estimated angle = {}",
                angle,
                estimated_angle
            );
        }
    }

    #[test]
    fn test_rotate_invertible() {
        let seed: [u8; 32] = [24; 32];
        let mut rng = SmallRng::from_seed(seed);
        let angle: f64 = rng.gen();

        let a = Transform::<f64>::rotate_x(angle);
        let b = Transform::<f64>::rotate_x(-angle);

        assert_eq!(a * b, Transform::<f64>::identity());
        assert_eq!(b * a, Transform::<f64>::identity());

        let c = Transform::<f64>::rotate_y(angle);
        let d = Transform::<f64>::rotate_y(-angle);

        assert_eq!(c * d, Transform::<f64>::identity());
        assert_eq!(d * c, Transform::<f64>::identity());

        let e = Transform::<f64>::rotate_z(angle);
        let f = Transform::<f64>::rotate_z(-angle);

        assert_eq!(e * f, Transform::<f64>::identity());
        assert_eq!(f * e, Transform::<f64>::identity());
    }

    #[test]
    fn test_look_at() {
        let seed: [u8; 32] = [123; 32];
        let mut rng = SmallRng::from_seed(seed);

        let num_tests = 20;
        for _ in 0..num_tests {
            let eye = Point3::<f64>::new(rng.gen(), rng.gen(), rng.gen());
            let target = Point3::<f64>::new(rng.gen(), rng.gen(), rng.gen());
            let up = Vector3::<f64>::new(rng.gen(), rng.gen(), rng.gen());

            let transform = Transform::<f64>::look_at(eye, target, up);

            let mapped_origin = transform.map(eye);
            assert!(
                mapped_origin.almost_eq(Point3::new(0.0, 0.0, 0.0)),
                "Expected the eye position to be mapped to zero, but instead it is mapped to {:?}",
                eye
            );

            let mapped_target_eye = transform.map((target - eye).normalize());
            assert!(
                mapped_target_eye.almost_eq(Vector3::new(0.0, 0.0, -1.0)),
                "Expected the mapped vector (target-eye) = {:?} to lie on the z-axis",
                mapped_target_eye
            );

            let actual_up = transform.map_inverse(Vector3::new(0.0, 1.0, 0.0));
            let up_without_z =
                (up - (eye - target).normalize() * up.dot((eye - target).normalize())).normalize();

            assert!(
                actual_up.almost_eq(up_without_z),
                "Expected the actual up vector to be the same as the suggested \
                up vector minus the component lying on the viewing direction \
                (to maintain orthogonality), but instead actual_up = {:?}, \
                while up_without_z = {:?}",
                actual_up,
                up_without_z
            );
        }
    }

    #[test]
    fn test_perspective() {
        let seed: [u8; 32] = [69; 32];
        let mut rng = SmallRng::from_seed(seed);

        let num_tests = 20;
        for _ in 0..num_tests {
            let dist_angle = Uniform::new_inclusive(0.0, std::f64::consts::FRAC_PI_2);
            let dist_plane = Uniform::new_inclusive(0.0, 100.0);
            let fov: f64 = dist_angle.sample(&mut rng);
            let (near, far) =
                f64::min_max(dist_plane.sample(&mut rng), dist_plane.sample(&mut rng));

            let transform = Transform::<f64>::perspective(fov, near, far);

            let tan_fov_2 = (fov / 2.0).tan();

            // tan(fov/2) = x_near / z_near
            // tan(fov/2) = y_near / z_near
            let p0 = Point3::<f64>::new(near * tan_fov_2, near * tan_fov_2, -near);
            let p1 = Point3::<f64>::new(-near * tan_fov_2, near * tan_fov_2, -near);
            let p2 = Point3::<f64>::new(near * tan_fov_2, -near * tan_fov_2, -near);
            let p3 = Point3::<f64>::new(-near * tan_fov_2, -near * tan_fov_2, -near);

            assert!(transform.map(p0).almost_eq(Point3::new(1.0, 1.0, 0.0)));
            assert!(transform.map(p1).almost_eq(Point3::new(-1.0, 1.0, 0.0)));
            assert!(transform.map(p2).almost_eq(Point3::new(1.0, -1.0, 0.0)));
            assert!(transform.map(p3).almost_eq(Point3::new(-1.0, -1.0, 0.0)));

            // tan(fov/2) = x_far / z_far
            // tan(fov/2) = y_far / z_far
            let p4 = Point3::<f64>::new(far * tan_fov_2, far * tan_fov_2, -far);
            let p5 = Point3::<f64>::new(-far * tan_fov_2, far * tan_fov_2, -far);
            let p6 = Point3::<f64>::new(far * tan_fov_2, -far * tan_fov_2, -far);
            let p7 = Point3::<f64>::new(-far * tan_fov_2, -far * tan_fov_2, -far);

            assert!(transform.map(p4).almost_eq(Point3::new(1.0, 1.0, 1.0)));
            assert!(transform.map(p5).almost_eq(Point3::new(-1.0, 1.0, 1.0)));
            assert!(transform.map(p6).almost_eq(Point3::new(1.0, -1.0, 1.0)));
            assert!(transform.map(p7).almost_eq(Point3::new(-1.0, -1.0, 1.0)));
        }
    }
}
