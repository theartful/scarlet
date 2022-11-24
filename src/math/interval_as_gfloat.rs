use crate::math::{
    interval::Interval,
    scalar::{min_max, Consts, GFloat, GFloatBits, NumTraitsGFloat, One},
};

impl<T: GFloat> GFloatBits for Interval<T> {
    #[inline]
    fn next_up(self) -> Self {
        Self::new(self.inf.next_up(), self.sup.next_up())
    }
    #[inline]
    fn next_down(self) -> Self {
        Self::new(self.inf.next_down(), self.sup.next_down())
    }
}

impl<T: GFloat> num_traits::ToPrimitive for Interval<T> {
    fn to_i64(&self) -> Option<i64> {
        self.approx().to_i64()
    }
    fn to_u64(&self) -> Option<u64> {
        self.approx().to_u64()
    }
    fn to_f32(&self) -> Option<f32> {
        self.approx().to_f32()
    }
    fn to_f64(&self) -> Option<f64> {
        self.approx().to_f64()
    }
}

impl<T: GFloat> num_traits::NumCast for Interval<T> {
    fn from<U: num_traits::ToPrimitive>(n: U) -> Option<Self> {
        T::from(n).map(|v| Self::new(v, v))
    }
}

impl<T: GFloat> NumTraitsGFloat for Interval<T> {
    fn nan() -> Self {
        Self::new(T::nan(), T::nan())
    }
    fn infinity() -> Self {
        Self::new(T::infinity(), T::infinity())
    }
    fn neg_infinity() -> Self {
        Self::new(T::neg_infinity(), T::neg_infinity())
    }
    fn neg_zero() -> Self {
        Self::new(T::neg_zero(), T::neg_zero())
    }
    fn min_value() -> Self {
        Self::new(T::min_value(), T::min_value())
    }
    fn min_positive_value() -> Self {
        Self::new(T::min_positive_value(), T::min_positive_value())
    }
    fn max_value() -> Self {
        Self::new(T::max_value(), T::max_value())
    }
    fn is_nan(self) -> bool {
        self.inf.is_nan() || self.sup.is_nan()
    }
    fn is_infinite(self) -> bool {
        self.inf.is_infinite() && self.sup.is_infinite() && self.inf.signum() == self.sup.signum()
    }
    fn is_finite(self) -> bool {
        self.inf.is_finite() && self.sup.is_finite()
    }
    fn is_normal(self) -> bool {
        self.inf.is_normal() && self.sup.is_normal()
    }
    fn classify(self) -> std::num::FpCategory {
        std::unimplemented!()
    }
    fn floor(self) -> Self {
        std::unimplemented!()
    }
    fn ceil(self) -> Self {
        std::unimplemented!()
    }
    fn round(self) -> Self {
        std::unimplemented!()
    }
    fn trunc(self) -> Self {
        std::unimplemented!()
    }
    fn fract(self) -> Self {
        std::unimplemented!()
    }
    fn abs(self) -> Self {
        let (inf, sup) = min_max(self.inf.abs(), self.sup.abs());
        Self::new(inf, sup)
    }
    fn signum(self) -> Self {
        std::unimplemented!()
    }
    fn is_sign_positive(self) -> bool {
        self.is_positive()
    }
    fn is_sign_negative(self) -> bool {
        self.is_negative()
    }
    fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
    }
    fn recip(self) -> Self {
        Self::one() / self
    }
    fn powi(self, _n: i32) -> Self {
        std::unimplemented!()
    }
    fn powf(self, _n: Self) -> Self {
        std::unimplemented!()
    }
    fn sqrt(self) -> Self {
        Self::new(self.inf.sqrt().next_down(), self.sup.sqrt().next_up())
    }
    fn exp(self) -> Self {
        Self::new(self.inf.exp().next_down(), self.sup.exp().next_up())
    }
    fn exp2(self) -> Self {
        Self::new(self.inf.exp2().next_down(), self.sup.exp2().next_up())
    }
    fn ln(self) -> Self {
        Self::new(self.inf.ln().next_down(), self.sup.ln().next_up())
    }
    fn log(self, _base: Self) -> Self {
        std::unimplemented!()
    }
    fn log2(self) -> Self {
        Self::new(self.inf.log2().next_down(), self.sup.log2().next_up())
    }
    fn log10(self) -> Self {
        Self::new(self.inf.log10().next_down(), self.sup.log10().next_up())
    }
    fn max(self, other: Self) -> Self {
        Self::new(self.inf.max(other.inf), self.sup.max(other.sup))
    }
    fn min(self, other: Self) -> Self {
        Self::new(self.inf.min(other.inf), self.sup.min(other.sup))
    }
    fn abs_sub(self, other: Self) -> Self {
        (self - other).abs()
    }
    fn cbrt(self) -> Self {
        Self::new(self.inf.cbrt().next_down(), self.sup.cbrt().next_up())
    }
    fn hypot(self, _other: Self) -> Self {
        std::unimplemented!();
    }
    fn sin(self) -> Self {
        std::unimplemented!();
    }
    fn cos(self) -> Self {
        std::unimplemented!();
    }
    fn tan(self) -> Self {
        std::unimplemented!();
    }
    fn asin(self) -> Self {
        std::unimplemented!();
    }
    fn acos(self) -> Self {
        std::unimplemented!();
    }
    fn atan(self) -> Self {
        std::unimplemented!();
    }
    fn atan2(self, _other: Self) -> Self {
        std::unimplemented!();
    }
    fn sin_cos(self) -> (Self, Self) {
        std::unimplemented!();
    }
    fn exp_m1(self) -> Self {
        std::unimplemented!();
    }
    fn ln_1p(self) -> Self {
        std::unimplemented!();
    }
    fn sinh(self) -> Self {
        std::unimplemented!();
    }
    fn cosh(self) -> Self {
        std::unimplemented!();
    }
    fn tanh(self) -> Self {
        std::unimplemented!();
    }
    fn asinh(self) -> Self {
        std::unimplemented!();
    }
    fn acosh(self) -> Self {
        std::unimplemented!();
    }
    fn atanh(self) -> Self {
        std::unimplemented!();
    }
    fn integer_decode(self) -> (u64, i16, i8) {
        std::unimplemented!();
    }
}

impl<T: GFloat> Consts for Interval<T> {
    const PI: Self = Self {
        inf: T::PI,
        sup: T::PI,
    };
    const TAU: Self = Self {
        inf: T::TAU,
        sup: T::TAU,
    };
    const FRAC_PI_2: Self = Self {
        inf: T::FRAC_PI_2,
        sup: T::FRAC_PI_2,
    };
    const FRAC_PI_3: Self = Self {
        inf: T::FRAC_PI_3,
        sup: T::FRAC_PI_3,
    };
    const FRAC_PI_4: Self = Self {
        inf: T::FRAC_PI_4,
        sup: T::FRAC_PI_4,
    };
    const FRAC_PI_6: Self = Self {
        inf: T::FRAC_PI_6,
        sup: T::FRAC_PI_6,
    };
    const FRAC_PI_8: Self = Self {
        inf: T::FRAC_PI_8,
        sup: T::FRAC_PI_8,
    };
    const FRAC_1_PI: Self = Self {
        inf: T::FRAC_1_PI,
        sup: T::FRAC_1_PI,
    };
    const FRAC_2_PI: Self = Self {
        inf: T::FRAC_2_PI,
        sup: T::FRAC_2_PI,
    };
    const FRAC_2_SQRT_PI: Self = Self {
        inf: T::FRAC_2_SQRT_PI,
        sup: T::FRAC_2_SQRT_PI,
    };
    const SQRT_2: Self = Self {
        inf: T::SQRT_2,
        sup: T::SQRT_2,
    };
    const FRAC_1_SQRT_2: Self = Self {
        inf: T::FRAC_1_SQRT_2,
        sup: T::FRAC_1_SQRT_2,
    };
    const E: Self = Self {
        inf: T::E,
        sup: T::E,
    };
    const LOG2_E: Self = Self {
        inf: T::LOG2_E,
        sup: T::LOG2_E,
    };
    const LOG2_10: Self = Self {
        inf: T::LOG2_10,
        sup: T::LOG2_10,
    };
    const LOG10_E: Self = Self {
        inf: T::LOG10_E,
        sup: T::LOG10_E,
    };
    const LOG10_2: Self = Self {
        inf: T::LOG10_2,
        sup: T::LOG10_2,
    };
    const LN_2: Self = Self {
        inf: T::LN_2,
        sup: T::LN_2,
    };
    const LN_10: Self = Self {
        inf: T::LN_10,
        sup: T::LN_10,
    };
    const EPSILON: Self = Self {
        inf: T::EPSILON,
        sup: T::EPSILON,
    };
    const FEQUALS_EPSILON: Self = Self {
        inf: T::FEQUALS_EPSILON,
        sup: T::FEQUALS_EPSILON,
    };
}
