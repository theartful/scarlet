use std::{
    fmt::Debug,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign},
};

pub type Float = f32;
pub type Int = i32;
pub type UInt = u32;

pub trait NumOps<Rhs = Self, Output = Self>:
    Add<Rhs, Output = Output>
    + Sub<Rhs, Output = Output>
    + Mul<Rhs, Output = Output>
    + Div<Rhs, Output = Output>
    + Rem<Rhs, Output = Output>
{
}

pub trait NumAssignOps<Rhs = Self>:
    AddAssign<Rhs> + SubAssign<Rhs> + MulAssign<Rhs> + DivAssign<Rhs> + RemAssign<Rhs>
{
}

pub trait Scalar: PartialEq + PartialOrd + NumOps + NumAssignOps + Copy + Debug {
    fn zero() -> Self;
    fn one() -> Self;
    fn lowest() -> Self;
    fn highest() -> Self;

    #[inline]
    fn two() -> Self {
        Self::one() + Self::one()
    }

    #[inline]
    fn is_one(self) -> bool {
        self == Self::one()
    }
    #[inline]
    fn is_zero(self) -> bool {
        self == Self::zero()
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        if self < other {
            self
        } else {
            other
        }
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        if self > other {
            self
        } else {
            other
        }
    }
    #[inline]
    fn min_max(self, other: Self) -> (Self, Self) {
        if self < other {
            (self, other)
        } else {
            (other, self)
        }
    }
    #[inline]
    fn clamp(self, lo: Self, hi: Self) -> Self {
        if self < lo {
            lo
        } else if self > hi {
            hi
        } else {
            self
        }
    }
    #[inline]
    fn in_range(self, lo: Self, hi: Self) -> bool {
        self >= lo && self <= hi
    }
    #[inline]
    fn square(self) -> Self {
        self * self
    }
}

pub trait SignedScalar: Scalar + Neg<Output = Self> {
    #[inline]
    fn is_positive(self) -> bool {
        self > Self::zero()
    }
    #[inline]
    fn is_negative(self) -> bool {
        self < Self::zero()
    }
    #[inline]
    fn is_nonpositive(self) -> bool {
        self <= Self::zero()
    }
    #[inline]
    fn is_nonnegative(self) -> bool {
        self >= Self::zero()
    }
    #[inline]
    fn signum(self) -> Self {
        if self.is_positive() {
            Self::one()
        } else if self.is_negative() {
            -Self::one()
        } else {
            Self::zero()
        }
    }
    #[inline]
    fn abs(self) -> Self {
        if self.is_negative() {
            -self
        } else {
            self
        }
    }
}

pub trait GFloat: SignedScalar {
    fn next_up(self) -> Self;
    fn next_down(self) -> Self;

    // casting
    fn to_i64(self) -> i64;
    fn to_f32(self) -> f32;
    fn to_f64(self) -> f64;

    fn from_i64(rhs: i64) -> Self;
    fn from_f32(rhs: f32) -> Self;
    fn from_f64(rhs: f64) -> Self;

    fn nan() -> Self;
    fn infinity() -> Self;
    fn neg_infinity() -> Self;
    fn neg_zero() -> Self;
    fn min_positive_value() -> Self;
    fn is_nan(self) -> bool;
    fn is_finite(self) -> bool;
    fn is_infinite(self) -> bool;

    // consts
    fn half() -> Self;
    fn pi() -> Self;
    fn tau() -> Self;
    fn frac_pi_2() -> Self;
    fn frac_pi_4() -> Self;
    fn abs_epsilon() -> Self;
    fn rel_epsilon() -> Self;

    // to facilitate interval arithmetic
    fn pi_lower() -> Self {
        Self::pi().next_down()
    }
    fn pi_upper() -> Self {
        Self::pi().next_up()
    }
    fn tau_lower() -> Self {
        Self::tau().next_down()
    }
    fn tau_upper() -> Self {
        Self::tau().next_up()
    }
    fn frac_pi_2_lower() -> Self {
        Self::frac_pi_2().next_down()
    }
    fn frac_pi_2_upper() -> Self {
        Self::frac_pi_2().next_up()
    }
    fn frac_pi_4_lower() -> Self {
        Self::frac_pi_4().next_down()
    }
    fn frac_pi_4_upper() -> Self {
        Self::frac_pi_4().next_up()
    }

    // math
    #[inline]
    fn recip(self) -> Self {
        Self::one() / self
    }
    fn sqrt(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn sin_cos(self) -> (Self, Self);
    fn tan(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan(self) -> Self;
    fn atan2(self, rhs: Self) -> Self;

    fn fequals(self, other: Self) -> bool {
        // https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
        // https://floating-point-gui.de/errors/comparison/
        let diff = (self - other).abs();
        if diff < Self::abs_epsilon() {
            return true;
        }

        let largest = self.abs().max(other.abs());

        diff < largest * Self::rel_epsilon()
    }
}

pub trait AlmostEqual<U = Self> {
    fn almost_eq(self, other: U) -> bool;
}

// ----------------------------  impls ---------------------------------------
impl<Rhs, Output, T> NumOps<Rhs, Output> for T where
    T: Add<Rhs, Output = Output>
        + Sub<Rhs, Output = Output>
        + Mul<Rhs, Output = Output>
        + Div<Rhs, Output = Output>
        + Rem<Rhs, Output = Output>
{
}

impl<Rhs, T> NumAssignOps<Rhs> for T where
    T: AddAssign<Rhs> + SubAssign<Rhs> + MulAssign<Rhs> + DivAssign<Rhs> + RemAssign<Rhs>
{
}

impl Scalar for u8 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn lowest() -> Self {
        0
    }
    fn highest() -> Self {
        255
    }
}
impl Scalar for u16 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn lowest() -> Self {
        0
    }
    fn highest() -> Self {
        65535
    }
}
impl Scalar for u32 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn lowest() -> Self {
        0
    }
    fn highest() -> Self {
        4294967295
    }
}
impl Scalar for u64 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn lowest() -> Self {
        0
    }
    fn highest() -> Self {
        18446744073709551615
    }
}
impl Scalar for i8 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn lowest() -> Self {
        -128
    }
    fn highest() -> Self {
        127
    }
}
impl Scalar for i16 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn lowest() -> Self {
        -32768
    }
    fn highest() -> Self {
        32767
    }
}
impl Scalar for i32 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn lowest() -> Self {
        -2147483648
    }
    fn highest() -> Self {
        2147483647
    }
}
impl Scalar for i64 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn lowest() -> Self {
        -9223372036854775808
    }
    fn highest() -> Self {
        9223372036854775807
    }
}
impl Scalar for f32 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn lowest() -> Self {
        f32::MIN
    }
    fn highest() -> Self {
        f32::MAX
    }
}
impl Scalar for f64 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn lowest() -> Self {
        f64::MIN
    }
    fn highest() -> Self {
        f64::MAX
    }
}
impl SignedScalar for i8 {}
impl SignedScalar for i16 {}
impl SignedScalar for i32 {}
impl SignedScalar for i64 {}
impl SignedScalar for f32 {}
impl SignedScalar for f64 {}

impl GFloat for f32 {
    // https://github.com/rust-lang/rust/pull/100578
    fn next_up(self) -> Self {
        // We must use strictly integer arithmetic to prevent denormals from
        // flushing to zero after an arithmetic operation on some platforms.
        const TINY_BITS: u32 = 0x1; // Smallest positive f32.
        const CLEAR_SIGN_MASK: u32 = 0x7fff_ffff;

        let bits = self.to_bits();
        if self.is_nan() || bits == Self::INFINITY.to_bits() {
            return self;
        }

        let abs = bits & CLEAR_SIGN_MASK;
        let next_bits = if abs == 0 {
            TINY_BITS
        } else if bits == abs {
            bits + 1
        } else {
            bits - 1
        };
        Self::from_bits(next_bits)
    }
    fn next_down(self) -> Self {
        // We must use strictly integer arithmetic to prevent denormals from
        // flushing to zero after an arithmetic operation on some platforms.
        const NEG_TINY_BITS: u32 = 0x8000_0001; // Smallest (in magnitude) negative f32.
        const CLEAR_SIGN_MASK: u32 = 0x7fff_ffff;

        let bits = self.to_bits();
        if self.is_nan() || bits == Self::NEG_INFINITY.to_bits() {
            return self;
        }

        let abs = bits & CLEAR_SIGN_MASK;
        let next_bits = if abs == 0 {
            NEG_TINY_BITS
        } else if bits == abs {
            bits - 1
        } else {
            bits + 1
        };
        Self::from_bits(next_bits)
    }

    // casting
    fn to_i64(self) -> i64 {
        self as i64
    }
    fn to_f32(self) -> f32 {
        self as f32
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn from_i64(rhs: i64) -> Self {
        rhs as f32
    }
    fn from_f32(rhs: f32) -> Self {
        rhs as f32
    }
    fn from_f64(rhs: f64) -> Self {
        rhs as f32
    }

    fn nan() -> Self {
        f32::NAN
    }
    fn infinity() -> Self {
        f32::INFINITY
    }
    fn neg_infinity() -> Self {
        f32::NEG_INFINITY
    }
    fn neg_zero() -> Self {
        -0.0
    }
    fn min_positive_value() -> Self {
        f32::MIN_POSITIVE
    }
    fn is_nan(self) -> bool {
        self.is_nan()
    }
    fn is_finite(self) -> bool {
        self.is_finite()
    }
    fn is_infinite(self) -> bool {
        self.is_finite()
    }

    // consts
    #[inline]
    fn half() -> Self {
        0.5
    }
    #[inline]
    fn pi() -> Self {
        std::f32::consts::PI
    }
    #[inline]
    fn tau() -> Self {
        std::f32::consts::TAU
    }
    #[inline]
    fn frac_pi_2() -> Self {
        std::f32::consts::FRAC_PI_2
    }
    #[inline]
    fn frac_pi_4() -> Self {
        std::f32::consts::FRAC_PI_4
    }
    #[inline]
    fn abs_epsilon() -> Self {
        1e-6_f32
    }
    #[inline]
    fn rel_epsilon() -> Self {
        1e-6_f32
    }

    #[inline]
    fn pi_lower() -> Self {
        // from boost interval
        13176794.0f32 / ((1 << 22) as f32)
    }
    #[inline]
    fn pi_upper() -> Self {
        13176794.0f32 / ((1 << 22) as f32)
    }
    #[inline]
    fn tau_lower() -> Self {
        Self::pi_lower() * 2.0f32
    }
    #[inline]
    fn tau_upper() -> Self {
        Self::pi_upper() * 2.0f32
    }
    #[inline]
    fn frac_pi_2_lower() -> Self {
        Self::pi_lower() / 2.0f32
    }
    #[inline]
    fn frac_pi_2_upper() -> Self {
        Self::pi_upper() / 2.0f32
    }
    #[inline]
    fn frac_pi_4_lower() -> Self {
        Self::pi_lower() / 4.0f32
    }
    #[inline]
    fn frac_pi_4_upper() -> Self {
        Self::pi_upper() / 4.0f32
    }

    // math
    #[inline]
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    #[inline]
    fn sin(self) -> Self {
        self.sin()
    }
    #[inline]
    fn cos(self) -> Self {
        self.cos()
    }
    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        self.sin_cos()
    }
    #[inline]
    fn tan(self) -> Self {
        self.tan()
    }
    #[inline]
    fn asin(self) -> Self {
        self.asin()
    }
    #[inline]
    fn acos(self) -> Self {
        self.acos()
    }
    #[inline]
    fn atan(self) -> Self {
        self.atan()
    }
    #[inline]
    fn atan2(self, rhs: Self) -> Self {
        self.atan2(rhs)
    }
}

impl GFloat for f64 {
    // https://github.com/rust-lang/rust/pull/100578
    fn next_up(self) -> Self {
        // We must use strictly integer arithmetic to prevent denormals from
        // flushing to zero after an arithmetic operation on some platforms.
        const TINY_BITS: u64 = 0x1; // Smallest positive f64.
        const CLEAR_SIGN_MASK: u64 = 0x7fff_ffff_ffff_ffff;

        let bits = self.to_bits();
        if self.is_nan() || bits == Self::INFINITY.to_bits() {
            return self;
        }

        let abs = bits & CLEAR_SIGN_MASK;
        let next_bits = if abs == 0 {
            TINY_BITS
        } else if bits == abs {
            bits + 1
        } else {
            bits - 1
        };
        Self::from_bits(next_bits)
    }
    fn next_down(self) -> Self {
        // We must use strictly integer arithmetic to prevent denormals from
        // flushing to zero after an arithmetic operation on some platforms.
        const NEG_TINY_BITS: u64 = 0x8000_0000_0000_0001; // Smallest (in magnitude) negative f64.
        const CLEAR_SIGN_MASK: u64 = 0x7fff_ffff_ffff_ffff;

        let bits = self.to_bits();
        if self.is_nan() || bits == Self::NEG_INFINITY.to_bits() {
            return self;
        }

        let abs = bits & CLEAR_SIGN_MASK;
        let next_bits = if abs == 0 {
            NEG_TINY_BITS
        } else if bits == abs {
            bits - 1
        } else {
            bits + 1
        };
        Self::from_bits(next_bits)
    }

    // casting
    fn to_i64(self) -> i64 {
        self as i64
    }
    fn to_f32(self) -> f32 {
        self as f32
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn from_i64(rhs: i64) -> Self {
        rhs as f64
    }
    fn from_f32(rhs: f32) -> Self {
        rhs as f64
    }
    fn from_f64(rhs: f64) -> Self {
        rhs as f64
    }

    fn nan() -> Self {
        f64::NAN
    }
    fn infinity() -> Self {
        f64::INFINITY
    }
    fn neg_infinity() -> Self {
        f64::NEG_INFINITY
    }
    fn neg_zero() -> Self {
        -0.0
    }
    fn min_positive_value() -> Self {
        f64::MIN_POSITIVE
    }
    fn is_nan(self) -> bool {
        self.is_nan()
    }
    fn is_finite(self) -> bool {
        self.is_finite()
    }
    fn is_infinite(self) -> bool {
        self.is_finite()
    }

    // consts
    #[inline]
    fn half() -> Self {
        0.5
    }
    #[inline]
    fn pi() -> Self {
        std::f64::consts::PI
    }
    #[inline]
    fn tau() -> Self {
        std::f64::consts::TAU
    }
    #[inline]
    fn frac_pi_2() -> Self {
        std::f64::consts::FRAC_PI_2
    }
    #[inline]
    fn frac_pi_4() -> Self {
        std::f64::consts::FRAC_PI_4
    }
    #[inline]
    fn abs_epsilon() -> Self {
        1e-6_f64
    }
    #[inline]
    fn rel_epsilon() -> Self {
        1e-6_f64
    }

    #[inline]
    fn pi_lower() -> Self {
        // from boost interval
        (3373259426.0 + 273688.0 / ((1 << 21) as f64)) / ((1 << 30) as f64)
    }
    #[inline]
    fn pi_upper() -> Self {
        (3373259426.0 + 273689.0 / ((1 << 21) as f64)) / ((1 << 30) as f64)
    }
    #[inline]
    fn tau_lower() -> Self {
        Self::pi_lower() * 2.0f64
    }
    #[inline]
    fn tau_upper() -> Self {
        Self::pi_upper() * 2.0f64
    }
    #[inline]
    fn frac_pi_2_lower() -> Self {
        Self::pi_lower() / 2.0f64
    }
    #[inline]
    fn frac_pi_2_upper() -> Self {
        Self::pi_upper() / 2.0f64
    }
    #[inline]
    fn frac_pi_4_lower() -> Self {
        Self::pi_lower() / 4.0f64
    }
    #[inline]
    fn frac_pi_4_upper() -> Self {
        Self::pi_upper() / 4.0f64
    }

    // math
    #[inline]
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    #[inline]
    fn sin(self) -> Self {
        self.sin()
    }
    #[inline]
    fn cos(self) -> Self {
        self.cos()
    }
    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        self.sin_cos()
    }
    #[inline]
    fn tan(self) -> Self {
        self.tan()
    }
    #[inline]
    fn asin(self) -> Self {
        self.asin()
    }
    #[inline]
    fn acos(self) -> Self {
        self.acos()
    }
    #[inline]
    fn atan(self) -> Self {
        self.atan()
    }
    #[inline]
    fn atan2(self, rhs: Self) -> Self {
        self.atan2(rhs)
    }
}

impl AlmostEqual for u8 {
    fn almost_eq(self, other: Self) -> bool {
        self == other
    }
}
impl AlmostEqual for u16 {
    fn almost_eq(self, other: Self) -> bool {
        self == other
    }
}
impl AlmostEqual for u32 {
    fn almost_eq(self, other: Self) -> bool {
        self == other
    }
}
impl AlmostEqual for u64 {
    fn almost_eq(self, other: Self) -> bool {
        self == other
    }
}
impl AlmostEqual for i8 {
    fn almost_eq(self, other: Self) -> bool {
        self == other
    }
}
impl AlmostEqual for i16 {
    fn almost_eq(self, other: Self) -> bool {
        self == other
    }
}
impl AlmostEqual for i32 {
    fn almost_eq(self, other: Self) -> bool {
        self == other
    }
}
impl AlmostEqual for i64 {
    fn almost_eq(self, other: Self) -> bool {
        self == other
    }
}
impl AlmostEqual for f32 {
    fn almost_eq(self, other: Self) -> bool {
        self.fequals(other)
    }
}
impl AlmostEqual for f64 {
    fn almost_eq(self, other: Self) -> bool {
        self.fequals(other)
    }
}
