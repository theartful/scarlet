pub use num_traits::Float as NumTraitsGFloat;
pub use num_traits::{Num, NumAssignOps, One, Zero};
use std::ops::Neg;

pub type Float = f32;
pub type Int = i32;
pub type UInt = u32;

pub trait LowestHighest {
    fn lowest() -> Self;
    fn highest() -> Self;
}

pub trait GFloatBits {
    fn next_up(self) -> Self;
    fn next_down(self) -> Self;
}

pub trait Consts {
    const PI: Self;
    const TAU: Self;
    const FRAC_PI_2: Self;
    const FRAC_PI_3: Self;
    const FRAC_PI_4: Self;
    const FRAC_PI_6: Self;
    const FRAC_PI_8: Self;
    const FRAC_1_PI: Self;
    const FRAC_2_PI: Self;
    const FRAC_2_SQRT_PI: Self;
    const SQRT_2: Self;
    const FRAC_1_SQRT_2: Self;
    const E: Self;
    const LOG2_E: Self;
    const LOG2_10: Self;
    const LOG10_E: Self;
    const LOG10_2: Self;
    const LN_2: Self;
    const LN_10: Self;
    const EPSILON: Self;
    const FEQUALS_EPSILON: Self;
}

// signed scalar
pub trait Scalar: Num + NumAssignOps + PartialOrd<Self> + LowestHighest + Copy {}
pub trait SignedScalar: Scalar + std::ops::Neg<Output = Self> {}

pub trait GFloat: Scalar + NumTraitsGFloat + Consts + GFloatBits + std::fmt::Debug {}
impl<T: num_traits::Num + NumAssignOps + PartialOrd + LowestHighest + Copy> Scalar for T {}
impl<T: Scalar + Neg<Output = Self>> SignedScalar for T {}

impl<T: Scalar + NumTraitsGFloat + Consts + GFloatBits + std::fmt::Debug> GFloat for T {}

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
    const PI: Self = std::f32::consts::PI;
    const TAU: Self = std::f32::consts::TAU;
    const FRAC_PI_2: Self = std::f32::consts::FRAC_PI_2;
    const FRAC_PI_3: Self = std::f32::consts::FRAC_PI_3;
    const FRAC_PI_4: Self = std::f32::consts::FRAC_PI_4;
    const FRAC_PI_6: Self = std::f32::consts::FRAC_PI_6;
    const FRAC_PI_8: Self = std::f32::consts::FRAC_PI_8;
    const FRAC_1_PI: Self = std::f32::consts::FRAC_1_PI;
    const FRAC_2_PI: Self = std::f32::consts::FRAC_2_PI;
    const FRAC_2_SQRT_PI: Self = std::f32::consts::FRAC_2_SQRT_PI;
    const SQRT_2: Self = std::f32::consts::SQRT_2;
    const FRAC_1_SQRT_2: Self = std::f32::consts::FRAC_1_SQRT_2;
    const E: Self = std::f32::consts::E;
    const LOG2_E: Self = std::f32::consts::LOG2_E;
    const LOG2_10: Self = std::f32::consts::LOG2_10;
    const LOG10_E: Self = std::f32::consts::LOG10_E;
    const LOG10_2: Self = std::f32::consts::LOG10_2;
    const LN_2: Self = std::f32::consts::LN_2;
    const LN_10: Self = std::f32::consts::LN_10;
    const EPSILON: Self = f32::EPSILON;
    const FEQUALS_EPSILON: Self = 1e-6_f32;
}

impl Consts for f64 {
    const PI: Self = std::f64::consts::PI;
    const TAU: Self = std::f64::consts::TAU;
    const FRAC_PI_2: Self = std::f64::consts::FRAC_PI_2;
    const FRAC_PI_3: Self = std::f64::consts::FRAC_PI_3;
    const FRAC_PI_4: Self = std::f64::consts::FRAC_PI_4;
    const FRAC_PI_6: Self = std::f64::consts::FRAC_PI_6;
    const FRAC_PI_8: Self = std::f64::consts::FRAC_PI_8;
    const FRAC_1_PI: Self = std::f64::consts::FRAC_1_PI;
    const FRAC_2_PI: Self = std::f64::consts::FRAC_2_PI;
    const FRAC_2_SQRT_PI: Self = std::f64::consts::FRAC_2_SQRT_PI;
    const SQRT_2: Self = std::f64::consts::SQRT_2;
    const FRAC_1_SQRT_2: Self = std::f64::consts::FRAC_1_SQRT_2;
    const E: Self = std::f64::consts::E;
    const LOG2_E: Self = std::f64::consts::LOG2_E;
    const LOG2_10: Self = std::f64::consts::LOG2_10;
    const LOG10_E: Self = std::f64::consts::LOG10_E;
    const LOG10_2: Self = std::f64::consts::LOG10_2;
    const LN_2: Self = std::f64::consts::LN_2;
    const LN_10: Self = std::f64::consts::LN_10;
    const EPSILON: Self = f64::EPSILON;
    const FEQUALS_EPSILON: Self = 1e-12_f64;
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
    let largest = max(a.abs(), b.abs());

    diff < largest * T::FEQUALS_EPSILON
}
