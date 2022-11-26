use crate::math::scalar::GFloat;

pub trait InnerScalar {
    type ScalarType;
}

pub trait Dimension {
    const DIM: usize;
}

pub trait VectorTraits:
    InnerScalar
    + Dimension
    + std::ops::Mul<<Self as InnerScalar>::ScalarType, Output = Self>
    + std::ops::Div<<Self as InnerScalar>::ScalarType, Output = Self>
    + std::ops::Add<Self, Output = Self>
    + std::ops::Sub<Self, Output = Self>
    + std::ops::Neg
    + Copy
    + Sized
{
}

impl<T> VectorTraits for T where
    T: InnerScalar
        + Dimension
        + std::ops::Mul<<Self as InnerScalar>::ScalarType, Output = Self>
        + std::ops::Div<<Self as InnerScalar>::ScalarType, Output = Self>
        + std::ops::Add<Self, Output = Self>
        + std::ops::Sub<Self, Output = Self>
        + std::ops::Neg
        + Copy
        + Sized
{
}

pub trait InnerProduct<Rhs>: InnerScalar + Sized + Copy {
    fn dot(self, rhs: Rhs) -> Self::ScalarType;
}

pub trait Norm: InnerProduct<Self> {
    #[inline]
    fn square_norm(self) -> Self::ScalarType {
        self.dot(self)
    }

    #[inline]
    fn norm(self) -> Self::ScalarType
    where
        Self::ScalarType: GFloat,
    {
        self.square_norm().sqrt()
    }

    #[inline]
    fn square_distance(self, vec: Self) -> Self::ScalarType
    where
        Self: std::ops::Sub<Self, Output = Self>,
    {
        (self - vec).square_norm()
    }

    #[inline]
    fn distance(self, vec: Self) -> Self::ScalarType
    where
        Self::ScalarType: GFloat,
        Self: std::ops::Sub<Self, Output = Self>,
    {
        self.square_distance(vec).sqrt()
    }

    #[inline]
    fn normalize(self) -> Self
    where
        Self::ScalarType: GFloat,
        Self: std::ops::Div<Self::ScalarType, Output = Self>,
    {
        self / self.norm()
    }
}

// trait specialization is not stable yet, and we would need it to provide
// optimized implementations for interval types
// impl<T> Norm for T where T: InnerProduct<T> {}
