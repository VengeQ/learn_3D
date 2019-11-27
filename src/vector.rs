pub mod vector_3;
pub mod n_vector;

///
/// Represent n-vector
///
pub trait Vector{
    ///
    /// Return zero vector for n-dimension space
    ///
    fn zero() -> Self;

    ///
    /// Return -vector
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