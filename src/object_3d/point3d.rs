use crate::object_3d::affine3d::Affine3d;

#[derive(Debug, PartialOrd, PartialEq, Copy, Clone)]
struct Point {
    x: f64,
    y: f64,
    z: f64,
}

impl Affine3d for Point {
    fn x(&self) -> f64 {
        self.x
    }

    fn y(&self) -> f64 {
        self.y
    }

    fn z(&self) -> f64 {
        self.z
    }
    fn new(x: f64, y: f64, z: f64) -> Self {
        Point {
            x,
            y,
            z,
        }
    }
}

impl Point {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::affine::Affine;

    #[test]
    fn point_new_test() {
        let p1 = Point::new(1.0, 3.0, 4.0);
        unsafe {
            let type_of_p1 = std::intrinsics::type_name::<Point>();
            assert_eq!(type_of_p1, "learn_3D::object_3d::point3d::Point");
        }
    }


    #[test]
    fn point_equal_test() {
        let p1 = Point::new(1.0, 3.0, 4.0);
        let p2 = Point::new(1.0, 3.0, 4.0);
        assert_eq!(p1, p2);
    }

    #[test]
    fn point_polar_simple_test() {
        let p1 = Point::new(4.0, 4.0, 4.0);
        let p1_in_polar = p1.polar();
        assert_eq!(p1.polar().2, 45.0)
    }
}

