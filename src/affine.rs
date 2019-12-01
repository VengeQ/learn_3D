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
    fn polar(&self) -> (f64, f64, f64);

    ///
    /// Return negate affine object
    ///
    fn reverse(&self) -> Self;

    ///
    /// Return magnitude of vector or normalize distance between point and Origin
    ///
    fn length(&self) -> f64;
}

