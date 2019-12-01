use crate::affine::vector_3::Vector3;
use crate::object_3d::point3d::Point3d;


struct Triangle3d {
    point0: Point3d,
    point1: Point3d,
    point2: Point3d,
}

impl Triangle3d {
    pub fn new(point0: Point3d, point1: Point3d, point2: Point3d) -> Self {
        Self {
            point0,
            point1,
            point2,
        }
    }

    //Todo Need to improve
    pub fn check_point(&self, point: Point3d) -> bool {
        let p0 = self.point0;
        let p1 = self.point1;
        let p2 = self.point2;
        let v0: Vector3 = crate::object_3d::point3d::Point3d::create_vector(p1, p0);
        let v1: Vector3 = crate::object_3d::point3d::Point3d::create_vector(p2, p1);
        let v2: Vector3 = crate::object_3d::point3d::Point3d::create_vector(p0, p2);
        let normal = v0.cross_product(v1).normalize();
        let w0: Vector3 = crate::object_3d::point3d::Point3d::create_vector(point, p0);
        let w1: Vector3 = crate::object_3d::point3d::Point3d::create_vector(point, p1);
        let w2: Vector3 = crate::object_3d::point3d::Point3d::create_vector(point, p2);

        let crosses = (
            v0.cross_product(w0).normalize(),
            v1.cross_product(w1).normalize(),
            v2.cross_product(w2).normalize()
        );

        let check = (
            crosses.0.dot_product(normal),
            crosses.1.dot_product(normal),
            crosses.2.dot_product(normal)
        );

        //ToDo Need to improve
        check==(1.0,1.0,1.0)
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::affine::Affine;
    use crate::object_3d::affine3d::Affine3d;

    #[test]
    fn new_test() {
        let p1 = Point3d::new(1.0, 2.0, 3.0);
        let p2 = Point3d::new(3.0, 4.0, 7.0);
        let p3 = Point3d::new(5.0, 1.0, 9.0);
        let t = Triangle3d::new(p1, p2, p3);
    }


    #[test]
    fn lying_point_test() {
        let p0 = Point3d::new(10.0, 0.0, 4.0);
        let p1 = Point3d::new(10.0, 10.0, 4.0);
        let p2 = Point3d::new(20.0, 10.0, 4.0);
        let t = Triangle3d::new(p0, p1, p2);
        let p = Point3d::new(11.0, 9.0, 4.0);

        let v0: Vector3 = crate::object_3d::point3d::Point3d::create_vector(p1, p0);
        let v1: Vector3 = crate::object_3d::point3d::Point3d::create_vector(p2, p1);
        let v2: Vector3 = crate::object_3d::point3d::Point3d::create_vector(p0, p2);
        let normal = v0.cross_product(v1).normalize();
        let w0: Vector3 = crate::object_3d::point3d::Point3d::create_vector(p, p0);
        let w1: Vector3 = crate::object_3d::point3d::Point3d::create_vector(p, p1);
        let w2: Vector3 = crate::object_3d::point3d::Point3d::create_vector(p, p2);

        let crosses = (
            v0.cross_product(w0).normalize(),
            v1.cross_product(w1).normalize(),
            v2.cross_product(w2).normalize()
        );

        let check = (
            crosses.0.dot_product(normal),
            crosses.1.dot_product(normal),
            crosses.2.dot_product(normal)
        );
        println!("{:?}", normal);
        println!("{:?}", crosses);
        println!("{:?}", check);
    }
}