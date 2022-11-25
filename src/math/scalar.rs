pub use num_traits::Float as NumTraitsGFloat;
pub use num_traits::{Num, NumAssignOps, One, Zero};
use std::fmt::Debug;
use std::ops::Neg;

pub type Float = f32;
pub type Int = i32;
pub type UInt = u32;

pub trait AlmostEqual {
    fn almost_eq(&self, other: &Self) -> bool;
}

pub trait LowestHighest {
    fn lowest() -> Self;
    fn highest() -> Self;
}

pub trait GFloatBits {
    fn next_up(self) -> Self;
    fn next_down(self) -> Self;
}

pub trait Consts {
    fn pi() -> Self;
    fn tau() -> Self;
    fn frac_pi_2() -> Self;
    fn frac_pi_3() -> Self;
    fn frac_pi_4() -> Self;
    fn frac_pi_6() -> Self;
    fn frac_pi_8() -> Self;
    fn frac_1_pi() -> Self;
    fn frac_2_pi() -> Self;
    fn frac_2_sqrt_pi() -> Self;
    fn sqrt_2() -> Self;
    fn frac_1_sqrt_2() -> Self;
    fn e() -> Self;
    fn log2_e() -> Self;
    fn log2_10() -> Self;
    fn log10_e() -> Self;
    fn log10_2() -> Self;
    fn ln_2() -> Self;
    fn ln_10() -> Self;
    fn abs_epsilon() -> Self;
    fn rel_epsilon() -> Self;
}

// signed scalar
pub trait Scalar:
    Num + NumAssignOps + PartialOrd<Self> + LowestHighest + Copy + Debug + AlmostEqual
{
}
pub trait SignedScalar: Scalar + std::ops::Neg<Output = Self> {}

pub trait GFloat: Scalar + NumTraitsGFloat + Consts + GFloatBits {}
impl<
        T: num_traits::Num + NumAssignOps + PartialOrd + LowestHighest + Copy + Debug + AlmostEqual,
    > Scalar for T
{
}

impl AlmostEqual for u8 {
    fn almost_eq(&self, other: &Self) -> bool {
        *self == *other
    }
}
impl AlmostEqual for u16 {
    fn almost_eq(&self, other: &Self) -> bool {
        *self == *other
    }
}
impl AlmostEqual for u32 {
    fn almost_eq(&self, other: &Self) -> bool {
        *self == *other
    }
}
impl AlmostEqual for u64 {
    fn almost_eq(&self, other: &Self) -> bool {
        *self == *other
    }
}
impl AlmostEqual for i8 {
    fn almost_eq(&self, other: &Self) -> bool {
        *self == *other
    }
}
impl AlmostEqual for i16 {
    fn almost_eq(&self, other: &Self) -> bool {
        *self == *other
    }
}
impl AlmostEqual for i32 {
    fn almost_eq(&self, other: &Self) -> bool {
        *self == *other
    }
}
impl AlmostEqual for i64 {
    fn almost_eq(&self, other: &Self) -> bool {
        *self == *other
    }
}
impl AlmostEqual for f32 {
    fn almost_eq(&self, other: &Self) -> bool {
        fequals(*self, *other)
    }
}
impl AlmostEqual for f64 {
    fn almost_eq(&self, other: &Self) -> bool {
        fequals(*self, *other)
    }
}

impl<T: Scalar + Neg<Output = Self>> SignedScalar for T {}

impl<T: Scalar + NumTraitsGFloat + Consts + GFloatBits + Debug> GFloat for T {}

impl LowestHighest for i32 {
    fn lowest() -> Self {
        std::i32::MIN
    }
    fn highest() -> Self {
        std::i32::MAX
    }
}

impl LowestHighest for i64 {
    fn lowest() -> Self {
        std::i64::MIN
    }
    fn highest() -> Self {
        std::i64::MAX
    }
}

impl LowestHighest for u8 {
    fn lowest() -> Self {
        0
    }
    fn highest() -> Self {
        std::u8::MAX
    }
}

impl LowestHighest for u32 {
    fn lowest() -> Self {
        0
    }
    fn highest() -> Self {
        std::u32::MAX
    }
}

impl LowestHighest for u64 {
    fn lowest() -> Self {
        0
    }
    fn highest() -> Self {
        std::u64::MAX
    }
}

impl LowestHighest for f32 {
    fn lowest() -> Self {
        f32::MIN
    }
    fn highest() -> Self {
        f32::MAX
    }
}

impl LowestHighest for f64 {
    fn lowest() -> Self {
        f64::MIN
    }
    fn highest() -> Self {
        f64::MAX
    }
}

impl Consts for f32 {
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
    fn frac_pi_3() -> Self {
        std::f32::consts::FRAC_PI_3
    }
    #[inline]
    fn frac_pi_4() -> Self {
        std::f32::consts::FRAC_PI_4
    }
    #[inline]
    fn frac_pi_6() -> Self {
        std::f32::consts::FRAC_PI_6
    }
    #[inline]
    fn frac_pi_8() -> Self {
        std::f32::consts::FRAC_PI_8
    }
    #[inline]
    fn frac_1_pi() -> Self {
        std::f32::consts::FRAC_1_PI
    }
    #[inline]
    fn frac_2_pi() -> Self {
        std::f32::consts::FRAC_2_PI
    }
    #[inline]
    fn frac_2_sqrt_pi() -> Self {
        std::f32::consts::FRAC_2_SQRT_PI
    }
    #[inline]
    fn sqrt_2() -> Self {
        std::f32::consts::SQRT_2
    }
    #[inline]
    fn frac_1_sqrt_2() -> Self {
        std::f32::consts::FRAC_1_SQRT_2
    }
    #[inline]
    fn e() -> Self {
        std::f32::consts::E
    }
    #[inline]
    fn log2_e() -> Self {
        std::f32::consts::LOG2_E
    }
    #[inline]
    fn log2_10() -> Self {
        std::f32::consts::LOG2_10
    }
    #[inline]
    fn log10_e() -> Self {
        std::f32::consts::LOG10_E
    }
    #[inline]
    fn log10_2() -> Self {
        std::f32::consts::LOG10_2
    }
    #[inline]
    fn ln_2() -> Self {
        std::f32::consts::LN_2
    }
    #[inline]
    fn ln_10() -> Self {
        std::f32::consts::LN_10
    }
    #[inline]
    fn abs_epsilon() -> Self {
        1e-6_f32
    }
    #[inline]
    fn rel_epsilon() -> Self {
        1e-6_f32
    }
}

impl Consts for f64 {
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
    fn frac_pi_3() -> Self {
        std::f64::consts::FRAC_PI_3
    }
    #[inline]
    fn frac_pi_4() -> Self {
        std::f64::consts::FRAC_PI_4
    }
    #[inline]
    fn frac_pi_6() -> Self {
        std::f64::consts::FRAC_PI_6
    }
    #[inline]
    fn frac_pi_8() -> Self {
        std::f64::consts::FRAC_PI_8
    }
    #[inline]
    fn frac_1_pi() -> Self {
        std::f64::consts::FRAC_1_PI
    }
    #[inline]
    fn frac_2_pi() -> Self {
        std::f64::consts::FRAC_2_PI
    }
    #[inline]
    fn frac_2_sqrt_pi() -> Self {
        std::f64::consts::FRAC_2_SQRT_PI
    }
    #[inline]
    fn sqrt_2() -> Self {
        std::f64::consts::SQRT_2
    }
    #[inline]
    fn frac_1_sqrt_2() -> Self {
        std::f64::consts::FRAC_1_SQRT_2
    }
    #[inline]
    fn e() -> Self {
        std::f64::consts::E
    }
    #[inline]
    fn log2_e() -> Self {
        std::f64::consts::LOG2_E
    }
    #[inline]
    fn log2_10() -> Self {
        std::f64::consts::LOG2_10
    }
    #[inline]
    fn log10_e() -> Self {
        std::f64::consts::LOG10_E
    }
    #[inline]
    fn log10_2() -> Self {
        std::f64::consts::LOG10_2
    }
    #[inline]
    fn ln_2() -> Self {
        std::f64::consts::LN_2
    }
    #[inline]
    fn ln_10() -> Self {
        std::f64::consts::LN_10
    }
    #[inline]
    fn abs_epsilon() -> Self {
        1e-6_f64
    }
    #[inline]
    fn rel_epsilon() -> Self {
        1e-6_f64
    }
}

// https://github.com/rust-lang/rust/pull/100578
impl GFloatBits for f32 {
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
}
impl GFloatBits for f64 {
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
}

// if a is NaN return b
pub fn min<T: PartialOrd<T>>(a: T, b: T) -> T {
    if a < b {
        a
    } else {
        b
    }
}

// if a is NaN return b
pub fn max<T: PartialOrd<T>>(a: T, b: T) -> T {
    if a > b {
        a
    } else {
        b
    }
}

pub fn min_max<T: PartialOrd<T>>(a: T, b: T) -> (T, T) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

pub fn clamp<T: PartialOrd<T>>(v: T, min: T, max: T) -> T {
    if v < min {
        min
    } else if v > max {
        max
    } else {
        v
    }
}

pub trait Sign {
    fn sign<U: Scalar>(x: U) -> Self;
}

impl<T: Scalar> Sign for T {
    fn sign<U: Scalar>(x: U) -> Self {
        if x < U::zero() {
            Self::zero()
        } else {
            Self::one()
        }
    }
}

pub fn fequals<T: GFloat>(a: T, b: T) -> bool {
    // https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
    // https://floating-point-gui.de/errors/comparison/

    let diff = (a - b).abs();
    if diff < T::abs_epsilon() {
        return true;
    }

    let largest = max(a.abs(), b.abs());

    diff < largest * T::rel_epsilon()
}
