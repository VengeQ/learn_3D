use crate::object_3d::affine3d::Affine3d;
use crate::affine::vector_3::Vector3;


#[derive(Debug, PartialOrd, PartialEq, Copy, Clone)]
pub struct Point3d {
    x: f64,
    y: f64,
    z: f64,
}

impl Affine3d for Point3d {
    fn x(&self) -> f64 {
        self.x
    }

    fn y(&self) -> f64 {
        self.y
    }

    fn z(&self) -> f64 {
        self.z
    }

    /// Create new Point in affine space from cartesian.
    /// # Example
    /// ```
    /// use learn_3D::object_3d::point3d::Point;
    /// use learn_3D::object_3d::affine3d::Affine3d;
    /// let point = Point::new(1.0,2.0,3.0);
    /// ```
    fn new(x: f64, y: f64, z: f64) -> Self {
        Point3d {
            x,
            y,
            z,
        }
    }

}

impl Point3d {
    #[inline]
    ///Return squared distance between two points.
    ///# Example
    /// ```
    /// use learn_3D::object_3d::point3d::Point3d;
    /// use learn_3D::object_3d::affine3d::Affine3d;
    /// let p1 = Point3d::new(2.0,0.0,0.0);
    /// let p2 = Point3d::new(4.0,0.0,0.0);
    /// let expected_squared_distance = (4.0_f64-2.0).powi(2).sqrt();
    /// assert_eq!(expected_squared_distance,Point3d::distance(p1,p2))
    /// ```
    pub fn distance(v1: Point3d, v2: Point3d) -> f64 {
        Self::distance_squared(v1, v2).sqrt()
    }

    /// Return squared distance between two points.
    /// # Example
    /// ```
    /// use learn_3D::object_3d::point3d::Point3d;
    /// use learn_3D::object_3d::affine3d::Affine3d;
    /// let p1 = Point3d::new(2.0,0.0,0.0);
    /// let p2 = Point3d::new(4.0,0.0,0.0);
    /// let expected_squared_distance = (4.0_f64-2.0).powi(2);
    /// assert_eq!(expected_squared_distance,Point3d::distance_squared(p1,p2))
    /// ```
    pub fn distance_squared(v1: Point3d, v2: Point3d) -> f64 {
        let x = v1.x - v2.x;
        let y = v1.y - v2.y;
        let z = v1.z - v2.z;
        x * x + y * y + z * z
    }

    ///Create vector between to point
    pub fn create_vector(start:Self, finish:Self) ->Vector3{
        Vector3::new(finish.x-start.x,finish.y-start.y,finish.z-start.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::affine::Affine;
    use rand::Rng;

    const DIFF_ERR: f64 = 10e-7;

    #[test]
    fn point_new_test() {
        let p1 = Point3d::new(1.0, 3.0, 4.0);
        unsafe {
            let type_of_p1 = std::intrinsics::type_name::<Point3d>();
            assert_eq!(type_of_p1, "learn_3D::object_3d::point3d::Point3d");
        }
    }

    #[test]
    fn point_equal_test() {
        let p1 = Point3d::new(1.0, 3.0, 4.0);
        let p2 = Point3d::new(1.0, 3.0, 4.0);
        assert_eq!(p1, p2);
    }

    #[test]
    fn point_polar_simple_test() {
        let p1 = Point3d::new(4.0, 4.0, 4.0);
        let p1_in_polar = p1.polar();
        assert_eq!(p1.polar().2, 45.0)
    }

    #[test]
    fn point_distance_test() {
        let mut rng = rand::thread_rng();
        for _ in 0..1000 {
            let x1 = rng.gen_range(0, 1000) as f64 / 10.0;
            let y1 = rng.gen_range(0, 1000) as f64 / 10.0;
            let z1 = rng.gen_range(0, 1000) as f64 / 10.0;
            let x2 = rng.gen_range(0, 1000) as f64 / 10.0;
            let y2 = rng.gen_range(0, 1000) as f64 / 10.0;
            let z2 = rng.gen_range(0, 1000) as f64 / 10.0;

            let p1 = Point3d::new(x1, y1, z1);
            let p2 = Point3d::new(x2, y2, z2);
            let fact = Point3d::distance(p1, p2);
            let x = p1.x - p2.x;
            let y = p1.y - p2.y;
            let z = p1.z - p2.z;
            let expect = (x * x + y * y + z * z).sqrt();

            assert!(expect - fact <= DIFF_ERR);
        }
    }

    #[test]
    fn point_distance_squared_test() {
        let mut rng = rand::thread_rng();
        for _ in 0..1000 {
            let x1 = rng.gen_range(0, 1000) as f64 / 10.0;
            let y1 = rng.gen_range(0, 1000) as f64 / 10.0;
            let z1 = rng.gen_range(0, 1000) as f64 / 10.0;
            let x2 = rng.gen_range(0, 1000) as f64 / 10.0;
            let y2 = rng.gen_range(0, 1000) as f64 / 10.0;
            let z2 = rng.gen_range(0, 1000) as f64 / 10.0;

            let p1 = Point3d::new(x1, y1, z1);
            let p2 = Point3d::new(x2, y2, z2);

            let fact = Point3d::distance_squared(p1, p2);
            let x = p1.x - p2.x;
            let y = p1.y - p2.y;
            let z = p1.z - p2.z;
            let expect = x * x + y * y + z * z;
            assert!(expect - fact <= DIFF_ERR);
        }
    }
}

