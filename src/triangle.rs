use crate::affine::vector_3::Vector3;

type Point = (f64, f64, f64);

struct Triangle {
    point0: Point,
    point1: Point,
    point2: Point,
}

impl Triangle {
    pub fn new(point0: Point, point1: Point, point2: Point) -> Self {
        Self {
            point0,
            point1,
            point2,
        }
    }
}


impl From<Vector3> for Point {
    fn from(v: Vector3) -> Self {
        v.xyz()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::affine::Affine;

    #[test]
    fn new_test() {
        let p1 = (1.0, 2.0, 3.0);
        let p2 = (3.0, 4.0, 7.0);
        let p3 = (5.0, 1.0, 9.0);
        let t = Triangle::new(p1, p2, p3);
    }

    #[test]
    fn into_vector3_test() {
        let x = 1.0;
        let y = -2.0;
        let z = 3.5;
        let p1 = (x, y, z);
        let vector: Vector3 = p1.into();
        assert_eq!(vector, Vector3::new(x, y, z))
    }

    #[test]
    fn lying_point_test() {
        let p0: Point = (10.0, 0.0, 4.0);
        let p1: Point = (10.0, 10.0, 4.0);
        let p2: Point = (20.0, 10.0, 4.0);
        let t = Triangle::new(p0, p1, p2);
        let p = (11.0, 9.0, 4.0);

        let v0: Vector3 = Vector3::from(p1) - p0.into();
        let v1: Vector3 = Vector3::from(p2) - p1.into();
        let v2: Vector3 = Vector3::from(p0) - p2.into();
        let normal = v0.cross_product(v1).normalize();
        let w0: Vector3 = Vector3::from(p) - p0.into();
        let w1: Vector3 = Vector3::from(p) - p1.into();
        let w2: Vector3 = Vector3::from(p) - p2.into();

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