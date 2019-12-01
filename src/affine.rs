pub mod vector_3;
pub mod n_vector;

///
/// Represent n-vector or n-point
///
pub trait Affine {
    ///
    /// Return zero  for n-dimension space
    ///
    fn zero() -> Self;

    ///
    /// Return polar coordinates(length, phi and theta in degree)
    ///
    fn polar(&self) -> (f64,f64,f64);

    ///
    /// Return negate affine object
    ///
    fn reverse(&self) -> Self;

    ///
    /// Return magnitude of vector
   ///
    fn length(&self) -> f64;

    ///
    /// Return product value on vector
    ///
    fn scalar_product(&self, scalar: f64) -> Self;

    ///
    /// Return 1-length vector parallel to input vector
    ///
    fn normalize(&self) -> Self;

    ///
    /// Return dot product of two vectors as f64
    ///
    fn dot_product(&self, right: Self) -> f64;

    ///
    /// Return cross product as new vector
    ///
    fn cross_product(&self, right: Self) -> Self;
}

