use crate::math::{
    interval::Interval,
    scalar::{GFloat, Scalar},
};

impl<T: GFloat> GFloat for Interval<T> {
    #[inline]
    fn next_up(self) -> Self {
        Self::new(self.inf.next_up(), self.sup.next_up())
    }
    #[inline]
    fn next_down(self) -> Self {
        Self::new(self.inf.next_down(), self.sup.next_down())
    }
    #[inline]
    fn to_i64(self) -> i64 {
        self.approx().to_i64()
    }
    #[inline]
    fn to_f32(self) -> f32 {
        self.approx().to_f32()
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self.approx().to_f64()
    }
    #[inline]
    fn from_i64(rhs: i64) -> Self {
        Self::from(T::from_i64(rhs))
    }
    #[inline]
    fn from_f32(rhs: f32) -> Self {
        Self::from(T::from_f32(rhs))
    }
    #[inline]
    fn from_f64(rhs: f64) -> Self {
        Self::from(T::from_f64(rhs))
    }
    #[inline]
    fn nan() -> Self {
        Self::new(T::nan(), T::nan())
    }

    #[inline]
    fn infinity() -> Self {
        Self::new(T::infinity(), T::infinity())
    }
    #[inline]
    fn neg_infinity() -> Self {
        Self::new(T::neg_infinity(), T::neg_infinity())
    }
    #[inline]
    fn neg_zero() -> Self {
        Self::new(T::neg_zero(), T::neg_zero())
    }
    #[inline]
    fn min_positive_value() -> Self {
        Self::new(T::min_positive_value(), T::min_positive_value())
    }
    #[inline]
    fn is_nan(self) -> bool {
        self.inf.is_nan() || self.sup.is_nan()
    }
    #[inline]
    fn is_infinite(self) -> bool {
        self.inf.is_infinite() && self.sup.is_infinite() && self.inf.signum() == self.sup.signum()
    }
    #[inline]
    fn is_finite(self) -> bool {
        self.inf.is_finite() && self.sup.is_finite()
    }
    #[inline]
    fn sqrt(self) -> Self {
        Self::from(T::min_max(
            self.inf.sqrt().next_down(),
            self.sup.sqrt().next_up(),
        ))
    }

    fn sin(self) -> Self {
        let diff = self.sup - self.inf;
        if diff >= T::tau() {
            return Self::new(-T::one(), T::one());
        }

        let (cos_inf, cos_sup) = (self.inf.cos(), self.sup.cos());

        // same side
        if cos_inf.signum() == cos_sup.signum() {
            let (t0, t1) = T::min_max(self.inf.sin(), self.sup.sin());
            Self::new(t0.next_down(), t1.next_up())
        } else {
            // minima
            if cos_inf.is_negative() {
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
            let (t0, t1) = T::min_max(self.inf.cos(), self.sup.cos());
            Self::new(t0.next_down(), t1.next_up())
        } else {
            // minima
            if sin_inf.is_positive() {
                Self::new(-T::one(), self.inf.cos().max(self.sup.cos()).next_up())
            }
            // maxima
            else {
                Self::new(self.inf.cos().min(self.sup.cos()).next_down(), T::one())
            }
        }
    }
    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }
    #[inline]
    fn tan(self) -> Self {
        // TODO
        self.sin() / self.cos()
    }
    #[inline]
    fn asin(self) -> Self {
        self.map_monotonic_inc(T::asin)
            .intersect(Interval::new(T::zero(), T::frac_pi_2_upper()))
    }
    #[inline]
    fn acos(self) -> Self {
        self.map_monotonic_dec(T::acos)
            .intersect(Interval::new(T::zero(), T::frac_pi_2_upper()))
    }
    #[inline]
    fn atan(self) -> Self {
        self.map_monotonic_inc(T::atan)
    }
    #[inline]
    fn atan2(self, rhs: Self) -> Self {
        let y = self;
        let x = rhs;

        if x.is_zero() || y.is_zero() {
            Interval::zero()
        }
        // X >= 0  -> first and fourth quadrants
        // atan2 is the same as atan at this point
        else if x.is_nonnegative() {
            (y / x)
                .map_monotonic_inc(T::atan)
                .intersect(Interval::new(-T::frac_pi_2_upper(), T::frac_pi_2_upper()))
        }
        // X <= 0 -> second and third quadrants
        // result can be in [-pi, -pi/2] union [pi/2, pi] = [-pi/2, pi/2]
        else if x.is_nonpositive() {
            // X <= 0 && Y >= 0 -> second quadrant
            if y.is_nonnegative() {
                ((y / x).map_monotonic_inc(T::atan) + T::pi())
                    .intersect(Interval::new(T::frac_pi_2_lower(), T::pi_upper()))
            }
            // X <= 0 && Y <= 0 -> third quadrant
            else if y.is_negative() {
                ((y / x).map_monotonic_inc(T::atan) - T::pi())
                    .intersect(Interval::new(-T::pi_upper(), -T::frac_pi_2_lower()))
            }
            // sign is uncertain. if y > 0 we get pi and if y < 0 we get -pi
            else {
                Self::new(-T::pi_upper(), T::pi_upper())
            }
        }
        // we don't know if we're in the right half or the left half
        else {
            Self::atan2(y, Interval::new(x.inf, T::zero()))
                .union(Self::atan2(y, Interval::new(T::zero(), x.sup)))
        }
    }

    #[inline]
    fn half() -> Self {
        Self::from(T::half())
    }
    #[inline]
    fn pi() -> Self {
        Self {
            inf: T::pi_lower(),
            sup: T::pi_upper(),
        }
    }
    #[inline]
    fn tau() -> Self {
        Self {
            inf: T::tau_lower(),
            sup: T::tau_upper(),
        }
    }
    #[inline]
    fn frac_pi_2() -> Self {
        Self {
            inf: T::frac_pi_2_lower(),
            sup: T::frac_pi_2_upper(),
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

            assert!(sin_interval.almost_eq(estimated_sin_interval));
            assert!(cos_interval.almost_eq(estimated_cos_interval));
        }
    }

    #[test]
    fn test_interval_atan2() {
        let seed: [u8; 32] = [123; 32];
        let mut rng = SmallRng::from_seed(seed);
        let dist = Uniform::new_inclusive(-1000.0, 1000.0);
        let num_tests = 10000;

        for _ in 0..num_tests {
            let y = Interval::from(dist.sample(&mut rng));
            let x = Interval::from(dist.sample(&mut rng));

            let atan2_interval = Interval::atan2(y, x);
            let atan2_scalar = f64::atan2(y.to_f64(), x.to_f64());
            assert!(
                atan2_interval.almost_eq(atan2_scalar),
                "y = {:?} x = {:?} atan2_interval = {:?} atan2_scalar = {:?}",
                y,
                x,
                atan2_interval,
                atan2_scalar
            );
        }
    }
}
