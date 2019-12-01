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
    ///# Example
    /// ```
    /// use learn_3D::object_3d::point3d::Point;
    /// use learn_3D::object_3d::affine3d::Affine3d;
    /// use learn_3D::affine::Affine;
    /// let p1 =Point::new(1.0,1.0,1.0);
    /// assert_eq!(p1.polar().2, 45.0); //theta = 45 degree for (x,x,x)
    /// let p2 = Point::new(1.0,0.0,0.0);
    /// assert_eq!(p2.polar().1, 90.0); //theta = 90 degree for i
    /// assert_eq!(p2.polar().2, 0.0); //phi = 0 degree for i
    /// ```
    fn polar(&self) -> (f64, f64, f64) {
        let (x, y, z) = (self.x(), self.y(), self.z());
        let po = (x * x + y * y + z * z).sqrt();
        let phi = (x * x + y * y).sqrt().atan2(z) * 180.0 / PI;
        let theta = y.atan2(x) * 180.0 / PI;
        (po, phi, theta)
    }

    fn reverse(&self) -> Self {
        Self::new(self.x(), self.y(), self.z())
    }

    fn length(&self) -> f64 {
        unimplemented!()
    }
}