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
        let diff = self.sup - self.inf;
        if diff >= T::tau() {
            return Self::new(-T::one(), T::one());
        }

        let (cos_inf, cos_sup) = (self.inf.cos(), self.sup.cos());

        // same side
        if cos_inf.signum() == cos_sup.signum() {
            let (t0, t1) = min_max(self.inf.sin(), self.sup.sin());
            Self::new(t0.next_down(), t1.next_up())
        } else {
            // minima
            if cos_inf.is_sign_negative() {
                Self::new(-T::one(), self.inf.sin().max(self.sup.sin()).next_up())
            }
            // maxima
            else {
                Self::new(self.inf.sin().min(self.sup.sin()).next_down(), T::one())
            }
        }
    }
    fn cos(self) -> Self {
        let diff = self.sup - self.inf;
        if diff >= T::tau() {
            return Self::new(-T::one(), T::one());
        }

        let (sin_inf, sin_sup) = (self.inf.sin(), self.sup.sin());

        // same side
        if sin_inf.signum() == sin_sup.signum() {
            let (t0, t1) = min_max(self.inf.cos(), self.sup.cos());
            Self::new(t0.next_down(), t1.next_up())
        } else {
            // minima
            if sin_inf.is_sign_positive() {
                Self::new(-T::one(), self.inf.cos().max(self.sup.cos()).next_up())
            }
            // maxima
            else {
                Self::new(self.inf.cos().min(self.sup.cos()).next_down(), T::one())
            }
        }
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
        (self.sin(), self.cos())
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

// TODO: get tighter bounds
impl<T: GFloat> Consts for Interval<T> {
    #[inline]
    fn pi() -> Self {
        Self {
            inf: T::pi().next_down(),
            sup: T::pi().next_up(),
        }
    }
    #[inline]
    fn tau() -> Self {
        Self {
            inf: T::tau().next_down(),
            sup: T::tau().next_up(),
        }
    }
    #[inline]
    fn frac_pi_2() -> Self {
        Self {
            inf: T::frac_pi_2().next_down(),
            sup: T::frac_pi_2().next_up(),
        }
    }
    #[inline]
    fn frac_pi_3() -> Self {
        Self {
            inf: T::frac_pi_3().next_down(),
            sup: T::frac_pi_3().next_up(),
        }
    }
    #[inline]
    fn frac_pi_4() -> Self {
        Self {
            inf: T::frac_pi_4().next_down(),
            sup: T::frac_pi_4().next_up(),
        }
    }
    #[inline]
    fn frac_pi_6() -> Self {
        Self {
            inf: T::frac_pi_6().next_down(),
            sup: T::frac_pi_6().next_up(),
        }
    }
    #[inline]
    fn frac_pi_8() -> Self {
        Self {
            inf: T::frac_pi_8().next_down(),
            sup: T::frac_pi_8().next_up(),
        }
    }
    #[inline]
    fn frac_1_pi() -> Self {
        Self {
            inf: T::frac_1_pi().next_down(),
            sup: T::frac_1_pi().next_up(),
        }
    }
    #[inline]
    fn frac_2_pi() -> Self {
        Self {
            inf: T::frac_2_pi().next_down(),
            sup: T::frac_2_pi().next_up(),
        }
    }
    #[inline]
    fn frac_2_sqrt_pi() -> Self {
        Self {
            inf: T::frac_2_sqrt_pi().next_down(),
            sup: T::frac_2_sqrt_pi().next_up(),
        }
    }
    #[inline]
    fn sqrt_2() -> Self {
        Self {
            inf: T::sqrt_2().next_down(),
            sup: T::sqrt_2().next_up(),
        }
    }
    #[inline]
    fn frac_1_sqrt_2() -> Self {
        Self {
            inf: T::frac_1_sqrt_2().next_down(),
            sup: T::frac_1_sqrt_2().next_up(),
        }
    }
    #[inline]
    fn e() -> Self {
        Self {
            inf: T::e().next_down(),
            sup: T::e().next_up(),
        }
    }
    #[inline]
    fn log2_e() -> Self {
        Self {
            inf: T::log2_e().next_down(),
            sup: T::log2_e().next_up(),
        }
    }
    #[inline]
    fn log2_10() -> Self {
        Self {
            inf: T::log2_10().next_down(),
            sup: T::log2_10().next_up(),
        }
    }
    #[inline]
    fn log10_e() -> Self {
        Self {
            inf: T::log10_e().next_down(),
            sup: T::log10_e().next_up(),
        }
    }
    #[inline]
    fn log10_2() -> Self {
        Self {
            inf: T::log10_2().next_down(),
            sup: T::log10_2().next_up(),
        }
    }
    #[inline]
    fn ln_2() -> Self {
        Self {
            inf: T::ln_2().next_down(),
            sup: T::ln_2().next_up(),
        }
    }
    #[inline]
    fn ln_10() -> Self {
        Self {
            inf: T::ln_10().next_down(),
            sup: T::ln_10().next_up(),
        }
    }
    #[inline]
    fn rel_epsilon() -> Self {
        Self {
            inf: T::rel_epsilon(),
            sup: T::rel_epsilon(),
        }
    }
    #[inline]
    fn abs_epsilon() -> Self {
        Self {
            inf: T::abs_epsilon(),
            sup: T::abs_epsilon(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::scalar::AlmostEqual;
    use more_asserts::{assert_ge, assert_gt, assert_le, assert_lt};
    use rand::{
        distributions::{Distribution, Uniform},
        rngs::SmallRng,
        SeedableRng,
    };

    #[test]
    fn test_interval_sin_cos() {
        let seed: [u8; 32] = [123; 32];
        let mut rng = SmallRng::from_seed(seed);
        let dist_angle = Uniform::new_inclusive(0.0, std::f64::consts::PI);
        let dist_origin = Uniform::new_inclusive(-1e10, 1e10);

        let num_tests = 10000;
        for _ in 0..num_tests {
            let inf = dist_origin.sample(&mut rng);
            let sup = inf + dist_angle.sample(&mut rng);
            let interval = Interval::<f64>::new(inf, sup);

            let (sin_interval, cos_interval) = interval.sin_cos();

            let mut estimated_sin_inf = f64::infinity();
            let mut estimated_sin_sup = -f64::infinity();

            let mut estimated_cos_inf = f64::infinity();
            let mut estimated_cos_sup = -f64::infinity();

            let num_samples = 2000;
            for j in 0..=num_samples {
                let sample = (inf + (sup - inf) * (j as f64) / (num_samples as f64)).min(sup);

                let (sin_sample, cos_sample) = sample.sin_cos();

                assert!(
                    sin_interval.contains(sin_sample),
                    "sin_interval = {:?} sin_sample = {:?}",
                    sin_interval,
                    sin_sample
                );
                assert!(
                    cos_interval.contains(cos_sample),
                    "cos_interval = {:?} cos_sample = {:?}",
                    cos_interval,
                    cos_sample
                );

                estimated_sin_inf = estimated_sin_inf.min(sin_sample);
                estimated_sin_sup = estimated_sin_sup.max(sin_sample);

                estimated_cos_inf = estimated_cos_inf.min(cos_sample);
                estimated_cos_sup = estimated_cos_sup.max(cos_sample);
            }

            let estimated_sin_interval = Interval::new(estimated_sin_inf, estimated_sin_sup);
            let estimated_cos_interval = Interval::new(estimated_cos_inf, estimated_cos_sup);

            if sin_interval.inf != -1.0 {
                assert_lt!(sin_interval.inf, estimated_sin_interval.inf);
            } else {
                assert_le!(sin_interval.inf, estimated_sin_interval.inf);
            }
            if sin_interval.sup != 1.0 {
                assert_gt!(sin_interval.sup, estimated_sin_interval.sup);
            } else {
                assert_ge!(sin_interval.sup, estimated_sin_interval.sup);
            }

            if cos_interval.inf != -1.0 {
                assert_lt!(cos_interval.inf, estimated_cos_interval.inf);
            } else {
                assert_le!(cos_interval.inf, estimated_cos_interval.inf);
            }
            if cos_interval.sup != 1.0 {
                assert_gt!(cos_interval.sup, estimated_cos_interval.sup);
            } else {
                assert_ge!(cos_interval.sup, estimated_cos_interval.sup);
            }

            assert!(sin_interval.almost_eq(&estimated_sin_interval));
            assert!(cos_interval.almost_eq(&estimated_cos_interval));
        }
    }
}
