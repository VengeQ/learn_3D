use crate::vector::vector_3::Vector3;
use std::f64::consts::PI;

#[derive(Default, Debug, PartialOrd, PartialEq, Copy, Clone)]
pub struct Line3 {
    origin: Point3,
    direction: Vector3,
}

impl Line3 {
    pub fn new() -> Self {
        unimplemented!()
    }
}


#[derive(Default, Debug, PartialOrd, PartialEq, Copy, Clone)]
pub struct Point3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Point3 {
    pub fn from_cartesian(x: f64, y: f64, z: f64) -> Self {
        Self {
            x,
            y,
            z,
        }
    }

    pub fn xyz(&self) -> (f64, f64, f64) {
        (self.x, self.y, self.z)
    }

    ///Represent polar coordinates in degree
    pub fn polar(&self) -> (f64, f64, f64) {
        let (x, y, z) = (self.x, self.y, self.z);
        let po = (x * x + y * y + z * z).sqrt();
        let phi = (x * x + y * y).sqrt().atan2(z) * 180.0 / PI;
        let theta = y.atan2(x) * 180.0 / PI;
        (po, phi, theta)
    }
}

#[cfg(test)]
mod tests {
    const DIFF_ERR: f64 = 10e-7;

    use super::*;

    #[test]
    fn point_from_cartesian_test() {
        let p = Point3::from_cartesian(1.0, 2.0, 3.0);
    }

    #[test]
    fn xyz_test() {
        let p = Point3::from_cartesian(1.0, 2.0, 3.0);
        assert_eq!(p.xyz(), (1.0, 2.0, 3.0));
    }

    #[test]
    fn polar_test() {
        let p = Point3::from_cartesian(3.0, 3.0, 3.0);
        assert_eq!(p.polar().2, 45.0)
    }
}