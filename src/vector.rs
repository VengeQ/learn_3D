mod vector_3;

pub trait Vector{
    fn zero() -> Self;

    fn reverse(&self) -> Self;

    fn length(&self) -> f64;

    fn scalar_product(&self, scalar: f64) -> Self;

    fn normalize(&self) -> Self;

    fn dot_product(&self, right: Self) -> f64;

    fn cross_product(&self, right: Self) -> Self;


}