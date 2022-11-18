use crate::scalar::*;

// ideally we should change floating point rounding mode to positive and negative
// infinity before doing operations
// but we always extend the bounds by one ulp instead

// the invariant is inf >= sup
#[derive(Debug, Clone, Copy)]
pub struct Interval<T: GFloat> {
    pub inf: T,
    pub sup: T,
}

pub type Intervalf = Interval<Float>;

impl<T: GFloat> Interval<T> {
    pub fn new(inf: T, sup: T) -> Self {
        assert!(sup >= inf);
        Self { inf, sup }
    }
    pub fn is_positive(&self) -> bool {
        self.inf > T::zero()
    }
    pub fn is_nonnegative(&self) -> bool {
        self.inf >= T::zero()
    }
    pub fn is_negative(&self) -> bool {
        self.sup < T::zero()
    }
    pub fn square(&self) -> Self {
        // adapted from CGAL
        if self.inf >= T::zero() {
            Self::new(
                (self.inf * self.inf).next_down(),
                (self.sup * self.sup).next_up(),
            )
        } else if self.sup < T::zero() {
            Self::new(
                (self.sup * self.sup).next_down(),
                (self.inf * self.inf).next_up(),
            )
        } else {
            let abs_max = max(-self.inf, self.sup);
            Self::new(T::zero(), (abs_max * abs_max).next_up())
        }
    }
    pub fn sqrt(&self) -> Option<Self> {
        if self.is_positive() {
            Some(Self::new(
                self.inf.sqrt().next_down(),
                self.sup.sqrt().next_up(),
            ))
        } else {
            None
        }
    }
    pub fn approx(&self) -> T {
        (self.inf + self.sup) * T::from(0.5).unwrap()
    }
    pub fn in_range(&self, t0: T, t1: T) -> bool {
        self.inf >= t0 && self.sup < t1
    }
    pub fn is_exact(&self) -> bool {
        // should we use fequals instead?
        self.inf == self.sup
    }
}

impl<T: GFloat> From<T> for Interval<T> {
    fn from(t: T) -> Interval<T> {
        Interval::new(t, t)
    }
}
// can't do this
// impl<T: GFloat> From<Interval<f32>> for f32 {
//     fn from(t: Interval<f32>) -> f32 {
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
    fn from(t: Interval<f32>) -> f32 {
        t.approx()
    }
}
impl From<Interval<f64>> for f64 {
    fn from(t: Interval<f64>) -> f64 {
        t.approx()
    }
}
impl<T: GFloat> std::ops::Add<Self> for Interval<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(
            (self.inf + rhs.inf).next_down(),
            (self.sup + rhs.sup).next_up(),
        )
    }
}
impl<T: GFloat> std::ops::Sub<Self> for Interval<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(
            (self.inf - rhs.sup).next_down(),
            (self.sup - rhs.inf).next_up(),
        )
    }
}
impl<T: GFloat> std::ops::Add<T> for Interval<T> {
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        Self::new((self.inf + rhs).next_down(), (self.sup + rhs).next_up())
    }
}
impl<T: GFloat> std::ops::Sub<T> for Interval<T> {
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        Self::new((self.inf - rhs).next_down(), (self.sup - rhs).next_up())
    }
}
impl<T: GFloat> std::ops::Mul<Self> for Interval<T> {
    type Output = Self;

    // adapted from CGAL
    // they also avoid NaNs if possible which is not done here
    // is using the straight forward min/max version better?
    fn mul(self, rhs: Self) -> Self::Output {
        let a = &self;
        let b = &rhs;

        // if a is nonnegative
        if a.is_nonnegative() {
            // if b >= 0 then result = [a.inf * b.inf, a.sup * b.sup]
            // if b < 0  then result = [a.sup * b.inf, a.inf * b.sup]
            // if b ~= 0 then result = [a.sup * b.inf, a.sup * b.sup]
            let x = if b.inf >= T::zero() { a.inf } else { a.sup };
            let y = if b.sup <= T::zero() { a.inf } else { a.sup };
            Self::new((x * b.inf).next_down(), (y * b.sup).next_up())
        }
        // if a is negitive
        else if a.is_negative() {
            // if b >= 0 then result = [a.inf * b.sup, a.sup * b.inf]
            // if b < 0  then result = [a.sup * b.sup, a.inf * b.inf]
            // if b ~= 0 then result = [a.inf * b.sup, a.inf * b.inf]
            let x = if b.sup <= T::zero() { a.sup } else { a.inf };
            let y = if b.inf >= T::zero() { a.sup } else { a.inf };
            Self::new((x * b.sup).next_down(), (y * b.inf).next_up())
        }
        // this means a ~= 0
        else {
            // if b >= 0 then result = [a.inf * b.sup, a.sup * b.sup]
            // if b < 0  then result = [a.sup * b.inf, a.inf * b.inf]
            if b.inf >= T::zero() {
                Self::new((a.inf * b.sup).next_down(), (a.sup * b.sup).next_up())
            } else if b.sup <= T::zero() {
                Self::new((a.sup * b.inf).next_down(), (a.inf * b.inf).next_up())
            } else {
                Self::new(
                    min(a.inf * b.sup, a.sup * b.inf).next_down(),
                    max(a.inf * a.inf, b.sup * b.sup).next_up(),
                )
            }
        }
    }
}
impl<T: GFloat> std::ops::Mul<T> for Interval<T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self::new((self.inf * rhs).next_down(), (self.sup * rhs).next_up())
    }
}
impl<T: GFloat> std::ops::Div<Self> for Interval<T> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let a = &self;
        let b = &rhs;

        // if b is positive
        if b.inf > T::zero() {
            // if a >= 0 then result = [a.inf / b.sup, a.sup / b.inf]
            // if a < 0  then result = [a.inf / b.inf, a.sup / b.sup]
            // if a ~ 0  then result = [a.inf / b.inf, a.sup / b.inf]
            let x = if a.inf >= T::zero() { b.sup } else { b.inf };
            let y = if a.sup < T::zero() { b.sup } else { b.inf };
            Self::new((a.inf / x).next_down(), (a.sup / y).next_up())
        }
        // if b is negative
        else if b.sup < T::zero() {
            // if a >= 0 then result = [a.sup / b.sup, a.inf / b.inf]
            // if a < 0  then result = [a.sup / b.inf, a.inf / b.sup]
            // if a ~ 0  then result = [a.sup / b.sup, a.inf / b.sup]
            let x = if a.sup < T::zero() { b.inf } else { b.sup };
            let y = if a.inf >= T::zero() { b.inf } else { b.sup };
            Self::new((a.sup / x).next_down(), (a.inf / y).next_up())
        }
        // if 0 lies in b
        else {
            Self::new(T::neg_infinity(), T::infinity())
        }
    }
}
impl<T: GFloat> std::ops::Div<T> for Interval<T> {
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        Self::new((self.inf / rhs).next_down(), (self.sup / rhs).next_up())
    }
}
impl<T: GFloat> std::ops::AddAssign<Self> for Interval<T> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl<T: GFloat> std::ops::SubAssign<Self> for Interval<T> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl<T: GFloat> std::ops::MulAssign<Self> for Interval<T> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl<T: GFloat> std::ops::DivAssign<Self> for Interval<T> {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}
impl<T: GFloat> std::ops::Neg for Interval<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.sup, -self.inf)
    }
}
impl<T: GFloat> std::cmp::PartialEq<Self> for Interval<T> {
    fn eq(&self, other: &Self) -> bool {
        self.inf == other.sup && self.sup == other.inf
    }
}
impl<T: GFloat> std::cmp::PartialOrd<Self> for Interval<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self == other {
            Some(std::cmp::Ordering::Equal)
        } else if self.sup < other.inf {
            Some(std::cmp::Ordering::Less)
        } else if other.sup < self.inf {
            Some(std::cmp::Ordering::Greater)
        } else {
            None
        }
    }
}
impl<T: GFloat> std::cmp::PartialEq<T> for Interval<T> {
    fn eq(&self, other: &T) -> bool {
        self.is_exact() && self.inf == *other
    }
}
impl<T: GFloat> std::cmp::PartialOrd<T> for Interval<T> {
    fn partial_cmp(&self, other: &T) -> Option<std::cmp::Ordering> {
        if self == other {
            Some(std::cmp::Ordering::Equal)
        } else if self.sup < *other {
            Some(std::cmp::Ordering::Less)
        } else if *other < self.inf {
            Some(std::cmp::Ordering::Greater)
        } else {
            None
        }
    }
}
impl<T: GFloat> num_traits::Zero for Interval<T> {
    fn zero() -> Self {
        Self::new(T::zero(), T::zero())
    }
    fn is_zero(&self) -> bool {
        self.inf == T::zero() && self.sup == T::zero()
    }
}
impl<T: GFloat> num_traits::One for Interval<T> {
    fn one() -> Self {
        Self::new(T::one(), T::one())
    }
}
impl<T: GFloat> LowestHighest for Interval<T> {
    fn lowest() -> Self {
        Self::from(T::lowest())
    }
    fn highest() -> Self {
        Self::from(T::highest())
    }
}
impl<T: GFloat> num_traits::Num for Interval<T> {
    type FromStrRadixErr = T::FromStrRadixErr;
    fn from_str_radix(s: &str, r: u32) -> std::result::Result<Self, Self::FromStrRadixErr> {
        match T::from_str_radix(s, r) {
            Err(e) => Err(e),
            Ok(s) => Ok(Self::from(s)),
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
    fn rem_assign(&mut self, _rhs: Self) -> () {
        std::unimplemented!();
    }
}

impl<T: GFloat> GFloatBits for Interval<T> {
    fn next_up(self) -> Self {
        Self::new(self.inf.next_up(), self.sup.next_up())
    }
    fn next_down(self) -> Self {
        Self::new(self.inf.next_down(), self.sup.next_down())
    }
}
