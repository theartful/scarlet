use crate::math::scalar::{AlmostEqual, Float, GFloat, Scalar, SignedScalar};
use std::cmp::{Ordering, PartialEq, PartialOrd};

pub type Intervalf = Interval<Float>;

/// A struct representing an inclusive interval, for the purpose of being used
/// in interval arithmetic
/// This implementation does not currently have the tighest error bounds, but
/// it aims to at least have correct bounds (i.e. the inclusion property holds)
/// ideally we should change floating point rounding mode to positive or negative
/// infinity before doing operations but we always over extend the bounds by one
/// ulp instead
#[derive(Debug, Clone, Copy)]
pub struct Interval<T: GFloat> {
    pub inf: T,
    pub sup: T,
}

impl<T: GFloat> Interval<T> {
    #[inline]
    pub fn new(inf: T, sup: T) -> Self {
        debug_assert!(
            sup >= inf,
            "Invalid interval! sup = {:?}, inf = {:?}",
            sup,
            inf
        );
        Self { inf, sup }
    }
    #[inline]
    pub fn is_positive(self) -> bool {
        self.inf > T::zero()
    }
    #[inline]
    pub fn is_negative(self) -> bool {
        self.sup < T::zero()
    }
    #[inline]
    pub fn is_nonpositive(self) -> bool {
        self.sup <= T::zero()
    }
    #[inline]
    pub fn is_nonnegative(self) -> bool {
        self.inf >= T::zero()
    }
    #[inline]
    pub fn square(self) -> Self {
        // adapted from CGAL
        if self.is_nonnegative() {
            Self::new(
                (self.inf * self.inf).next_down(),
                (self.sup * self.sup).next_up(),
            )
        } else if self.is_nonpositive() {
            Self::new(
                (self.sup * self.sup).next_down(),
                (self.inf * self.inf).next_up(),
            )
        } else {
            let abs_max = (-self.inf).max(self.sup);
            Self::new(T::zero(), (abs_max * abs_max).next_up())
        }
    }
    #[inline]
    pub fn sqrt(self) -> Option<Self> {
        if self.is_nonnegative() {
            Some(Self::new(
                self.inf.sqrt().next_down(),
                self.sup.sqrt().next_up(),
            ))
        } else {
            None
        }
    }
    #[inline]
    pub fn approx(self) -> T {
        (self.inf + self.sup) * T::half()
    }
    #[inline]
    pub fn in_range(self, t0: T, t1: T) -> bool {
        self.inf >= t0 && self.sup <= t1
    }
    #[inline]
    pub fn is_exact(self) -> bool {
        self.inf == self.sup
    }
    #[inline]
    pub fn contains(self, t: T) -> bool {
        self.sup >= t && self.inf <= t
    }
    #[inline]
    pub fn is_same(self, other: Self) -> bool {
        self.inf == other.inf && self.sup == other.sup
    }
    #[inline]
    pub fn extend_by_ulp(self) -> Self {
        Self::new(self.inf.next_down(), self.sup.next_up())
    }
    #[inline]
    pub fn map_monotonic_inc<F>(self, mut f: F) -> Self
    where
        F: FnMut(T) -> T,
    {
        Self::new(f(self.inf), f(self.sup)).extend_by_ulp()
    }
    #[inline]
    pub fn map_monotonic_dec<F>(self, mut f: F) -> Self
    where
        F: FnMut(T) -> T,
    {
        Self::new(f(self.sup), f(self.inf)).extend_by_ulp()
    }
    #[inline]
    pub fn intersect(self, rhs: Self) -> Self {
        Self::new(self.inf.max(rhs.inf), self.sup.min(rhs.sup))
    }
    #[inline]
    pub fn union(self, rhs: Self) -> Self {
        Self::new(self.inf.min(rhs.inf), self.sup.max(rhs.sup))
    }
}

impl<T: GFloat> AlmostEqual<Self> for Interval<T> {
    /// checks that the two intervals almost have the same bounds
    fn almost_eq(self, other: Self) -> bool {
        self.inf.fequals(other.inf) && self.sup.fequals(other.sup)
    }
}
impl<T: GFloat> AlmostEqual<T> for Interval<T> {
    fn almost_eq(self, other: T) -> bool {
        self.contains(other) && self.inf.fequals(other) && self.sup.fequals(other)
    }
}
impl<T: GFloat> From<T> for Interval<T> {
    #[inline]
    fn from(t: T) -> Interval<T> {
        Interval::new(t, t)
    }
}
impl<T: GFloat> From<(T, T)> for Interval<T> {
    #[inline]
    fn from(t: (T, T)) -> Interval<T> {
        Interval::new(t.0, t.1)
    }
}
// can't do this
// impl<T: GFloat> From<Interval<T>> for T {
//     fn from(t: Interval<T>) -> T {
//         t.approx()
//     }
// }

// or this
// impl<T: GFloat> Into<T> for Interval<T> {
//     fn into(self) -> T {
//         self.approx()
//     }
// }
impl From<Interval<f32>> for f32 {
    #[inline]
    fn from(t: Interval<f32>) -> f32 {
        t.approx()
    }
}
impl From<Interval<f64>> for f64 {
    #[inline]
    fn from(t: Interval<f64>) -> f64 {
        t.approx()
    }
}
impl<T: GFloat> std::ops::Add<Self> for Interval<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(
            (self.inf + rhs.inf).next_down(),
            (self.sup + rhs.sup).next_up(),
        )
    }
}
impl<T: GFloat> std::ops::Sub<Self> for Interval<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(
            (self.inf - rhs.sup).next_down(),
            (self.sup - rhs.inf).next_up(),
        )
    }
}
impl<T: GFloat> std::ops::Add<T> for Interval<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        Self::new((self.inf + rhs).next_down(), (self.sup + rhs).next_up())
    }
}
impl<T: GFloat> std::ops::Sub<T> for Interval<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: T) -> Self::Output {
        Self::new((self.inf - rhs).next_down(), (self.sup - rhs).next_up())
    }
}
impl<T: GFloat> std::ops::Mul<Self> for Interval<T> {
    type Output = Self;

    // adapted from CGAL
    // they also avoid NaNs if possible which is not done here
    // is using the straight forward min/max version better?
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let a = &self;
        let b = &rhs;

        // if a >= 0
        if a.is_nonnegative() {
            // if b >= 0 then result = [a.inf * b.inf, a.sup * b.sup]
            // if b <= 0 then result = [a.sup * b.inf, a.inf * b.sup]
            // if b ~= 0 then result = [a.sup * b.inf, a.sup * b.sup]
            let x = if b.is_nonnegative() { a.inf } else { a.sup };
            let y = if b.is_nonpositive() { a.inf } else { a.sup };
            Self::new((x * b.inf).next_down(), (y * b.sup).next_up())
        }
        // if a <= 0
        else if a.is_nonpositive() {
            // if b >= 0 then result = [a.inf * b.sup, a.sup * b.inf]
            // if b <= 0 then result = [a.sup * b.sup, a.inf * b.inf]
            // if b ~= 0 then result = [a.inf * b.sup, a.inf * b.inf]
            let x = if b.is_nonpositive() { a.sup } else { a.inf };
            let y = if b.is_nonnegative() { a.sup } else { a.inf };
            Self::new((x * b.sup).next_down(), (y * b.inf).next_up())
        }
        // this means a ~= 0
        else {
            // if b >= 0 then result = [a.inf * b.sup, a.sup * b.sup]
            // if b <= 0 then result = [a.sup * b.inf, a.inf * b.inf]
            if b.is_nonnegative() {
                Self::new((a.inf * b.sup).next_down(), (a.sup * b.sup).next_up())
            } else if b.is_nonpositive() {
                Self::new((a.sup * b.inf).next_down(), (a.inf * b.inf).next_up())
            } else {
                Self::new(
                    (a.inf * b.sup).min(a.sup * b.inf).next_down(),
                    (a.inf * a.inf).max(b.sup * b.sup).next_up(),
                )
            }
        }
    }
}
impl<T: GFloat> std::ops::Mul<T> for Interval<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        if rhs >= T::zero() {
            Self::new((self.inf * rhs).next_down(), (self.sup * rhs).next_up())
        } else {
            Self::new((self.sup * rhs).next_down(), (self.inf * rhs).next_up())
        }
    }
}
impl<T: GFloat> std::ops::MulAssign<T> for Interval<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        *self = *self * rhs;
    }
}
impl<T: GFloat> std::ops::Div<Self> for Interval<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        let a = &self;
        let b = &rhs;

        // if b is positive
        if b.is_positive() {
            // if a >= 0 then result = [a.inf / b.sup, a.sup / b.inf]
            // if a <= 0 then result = [a.inf / b.inf, a.sup / b.sup]
            // if a ~= 0 then result = [a.inf / b.inf, a.sup / b.inf]
            let x = if a.is_nonnegative() { b.sup } else { b.inf };
            let y = if a.is_nonpositive() { b.sup } else { b.inf };
            Self::new((a.inf / x).next_down(), (a.sup / y).next_up())
        }
        // if b is negative
        else if b.is_negative() {
            // if a >= 0 then result = [a.sup / b.sup, a.inf / b.inf]
            // if a <= 0 then result = [a.sup / b.inf, a.inf / b.sup]
            // if a ~= 0 then result = [a.sup / b.sup, a.inf / b.sup]
            let x = if a.is_nonpositive() { b.inf } else { b.sup };
            let y = if a.is_nonnegative() { b.inf } else { b.sup };
            Self::new((a.sup / x).next_down(), (a.inf / y).next_up())
        }
        // if 0 lies in b
        else {
            Self::new(T::neg_infinity(), T::infinity())
        }
    }
}
impl<T: GFloat> std::ops::DivAssign<T> for Interval<T> {
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        *self = *self / rhs;
    }
}
impl<T: GFloat> std::ops::Div<T> for Interval<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        if rhs > T::zero() {
            Self::new((self.inf / rhs).next_down(), (self.sup / rhs).next_up())
        } else if rhs < T::zero() {
            Self::new((self.sup / rhs).next_down(), (self.inf / rhs).next_up())
        } else {
            Self::new(T::neg_infinity(), T::infinity())
        }
    }
}
impl<T: GFloat> std::ops::AddAssign<Self> for Interval<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl<T: GFloat> std::ops::SubAssign<Self> for Interval<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl<T: GFloat> std::ops::MulAssign<Self> for Interval<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl<T: GFloat> std::ops::DivAssign<Self> for Interval<T> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}
impl<T: GFloat> std::ops::Neg for Interval<T> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::new(-self.sup, -self.inf)
    }
}
impl<T: GFloat> PartialEq<Self> for Interval<T> {
    #[inline]
    /// checks that the two intervals have the same bounds
    fn eq(&self, other: &Self) -> bool {
        self.is_exact() && other.is_exact() && self.inf == other.inf
    }
}
impl<T: GFloat> PartialOrd<Self> for Interval<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self == other {
            Some(Ordering::Equal)
        } else if self.sup < other.inf {
            Some(Ordering::Less)
        } else if other.sup < self.inf {
            Some(Ordering::Greater)
        } else {
            None
        }
    }
}
impl<T: GFloat> PartialEq<T> for Interval<T> {
    #[inline]
    fn eq(&self, other: &T) -> bool {
        self.is_exact() && self.inf == *other
    }
}
impl<T: GFloat> PartialOrd<T> for Interval<T> {
    #[inline]
    fn partial_cmp(&self, other: &T) -> Option<Ordering> {
        if self == other {
            Some(Ordering::Equal)
        } else if self.sup < *other {
            Some(Ordering::Less)
        } else if *other < self.inf {
            Some(Ordering::Greater)
        } else {
            None
        }
    }
}
impl<T: GFloat> std::ops::Rem<Self> for Interval<T> {
    type Output = Self;
    fn rem(self, _rhs: Self) -> Self::Output {
        std::unimplemented!();
    }
}
impl<T: GFloat> std::ops::RemAssign<Self> for Interval<T> {
    fn rem_assign(&mut self, _rhs: Self) {
        std::unimplemented!();
    }
}

impl<T: GFloat> Scalar for Interval<T> {
    #[inline]
    fn zero() -> Self {
        Self::new(T::zero(), T::zero())
    }
    #[inline]
    fn one() -> Self {
        Self::new(T::one(), T::one())
    }
    #[inline]
    fn lowest() -> Self {
        Self::new(T::lowest(), T::lowest())
    }
    #[inline]
    fn highest() -> Self {
        Self::new(T::highest(), T::highest())
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        Self::new(self.inf.max(other.inf), self.sup.max(other.sup))
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        Self::new(self.inf.min(other.inf), self.sup.min(other.sup))
    }
}
impl<T: GFloat> SignedScalar for Interval<T> {
    #[inline]
    fn abs(self) -> Self {
        if self.inf.is_negative() {
            Self::new(T::zero(), self.inf.abs().max(self.sup.abs()))
        } else {
            Self::from(T::min_max(self.inf, self.sup))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spiral_test() {
        // taken from CGAL
        let mut x_i = Interval::<f64>::from(1.0);
        let mut y_i = Interval::<f64>::from(0.0);

        let mut i = 0;
        while i < 500 {
            i += 1;

            let x_ip1 = x_i - y_i / Interval::<f64>::from(i as f64).sqrt();
            let y_ip1 = y_i + x_i / Interval::<f64>::from(i as f64).sqrt();
            x_i = x_ip1;
            y_i = y_ip1;

            if x_i.contains(0.0) || y_i.contains(0.0) {
                break;
            }
        }
        // if we changed rounding modes instead of adding/subtracting ulps, this
        // should go up to 396
        assert_eq!(i, 365);
    }

    #[test]
    fn square_root_test() {
        // taken from CGAL
        use more_asserts::*;

        let mut i = 0;
        let mut a = Interval::<f64>::new(0.5, 1.5);

        while i < 500 {
            i += 1;
            let b = a.sqrt();

            if b.is_same(a) {
                break;
            }
            a = b;
        }
        a -= Interval::<f64>::from(1.0);
        assert_eq!(i, 54);

        // ideally we should converge to 1 as we repeatedly perform square root
        assert_lt!(a.sup, 3. / ((1 << 30) as f64 * (1 << 22) as f64));
        assert_gt!(a.inf, -3. / ((1 << 30) as f64 * (1 << 22) as f64));
    }

    #[test]
    fn division_test() {
        // taken from CGAL
        let mut a = Interval::<f64>::from(1.0);
        let mut b = Interval::<f64>::from(0.0);
        let mut d = Interval::<f64>::new(-1.0, 1.0) / -2.0 + Interval::<f64>::from(1.0); // [0.5, 1.5]
        let mut e = -d;

        let c = a / b;
        assert!(Interval::<f64>::new(f64::NEG_INFINITY, f64::INFINITY).is_same(c));

        let mut i = 0;
        while i < 100 {
            i += 1;

            b = (Interval::<f64>::from(1.0) / d + d) / 4.0 + Interval::<f64>::from(0.5);
            a = (Interval::<f64>::from(-1.0) / e - e * 1.0) / -4.0 - Interval::<f64>::from(0.5); // make it complicated to test more cases.

            if b.is_same(d) && a.is_same(e) {
                break;
            }
            d = b;
            e = a;
        }
        assert_eq!(i, 54);
    }
}
