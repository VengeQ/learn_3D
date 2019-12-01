use crate::affine::Affine;
use std::f64::consts::PI;

pub trait Affine3d {
    fn x(&self) -> f64;
    fn y(&self) -> f64;
    fn z(&self) -> f64;
    fn new(x: f64, y: f64, z: f64) -> Self;
}

impl<T> Affine for T where T: Affine3d + Sized {
    fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    ///Represent polar coordinates in degree
    fn polar(&self) -> (f64, f64, f64) {
        let (x, y, z) = (self.x(), self.y(), self.z());
        let po = (x * x + y * y + z * z).sqrt();
        let phi = (x * x + y * y).sqrt().atan2(z) * 180.0 / PI;
        let theta = y.atan2(x) * 180.0 / PI;
        (po, phi, theta)
    }

    fn reverse(&self) -> Self {
        Self::new(self.x(),self.y(),self.z())
    }

    fn length(&self) -> f64 {
        unimplemented!()
    }

    fn scalar_product(&self, scalar: f64) -> Self {
        unimplemented!()
    }

    fn normalize(&self) -> Self {
        unimplemented!()
    }

    fn dot_product(&self, right: Self) -> f64 {
        unimplemented!()
    }

    fn cross_product(&self, right: Self) -> Self {
        unimplemented!()
    }
}